"""Train binary dog whistle disambiguator (coded vs non-coded).

Trains on silent_signals positives + mined silver negatives.
Evaluates on: (1) grouped held-out test split, (2) locked detection set (101 rows),
(3) locked disambiguation set (124 rows).
"""
import os, json, gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score, confusion_matrix
from utils import parse_args, set_seed, print_diagnostics, content_hash, get_content_hashes

def build_binary_input(row, input_format="term_enriched_def",
                       defn_fallback="No standard definition available"):
    """Format one row for the binary disambiguator.

    Two arms (see DESIGN_DEFENSE.md D6):
      - "term":              Candidate term + Text (no definition line)
      - "term_enriched_def": Candidate term + Candidate meaning (enriched glossary) + Text

    The enriched glossary is read from the ``definition_enriched`` column,
    pre-joined into every parquet in ``final_data/rq_a_binary/``.
    """
    dw = str(row["dog_whistle"])
    content = str(row["content"])
    if input_format == "term":
        return (f"Candidate term: {dw}\n"
                f"Text: {content}\n"
                f"Question: Is this candidate used as a coded dog whistle here?")
    if input_format == "term_enriched_def":
        defn_raw = row.get("definition_enriched", None)
        defn = str(defn_raw) if pd.notna(defn_raw) else defn_fallback
        return (f"Candidate term: {dw}\n"
                f"Candidate meaning: {defn}\n"
                f"Text: {content}\n"
                f"Question: Is this candidate used as a coded dog whistle here?")
    raise ValueError(f"Unknown input_format: {input_format!r} "
                     "(expected 'term' or 'term_enriched_def')")

def load_binary_split(data_dir, split_name, seed, input_format="term_enriched_def"):
    """Read RQ-A data from final_data/rq_a_binary/{split}.parquet (positives + negatives
    pre-balanced and label-tagged). Falls back to the legacy paired layout, then to
    the un-paired layout, in that order, if final_data/ is not present yet.

    Always returns a frame with a ``text`` column built via ``build_binary_input``
    using ``input_format``.
    """
    final_path = f"{data_dir}/final_data/rq_a_binary/{split_name}.parquet"
    if os.path.exists(final_path):
        df = pd.read_parquet(final_path)
        if "label" not in df.columns:
            raise KeyError(f"{final_path} missing `label` column")
        print(f"[{split_name}] using final_data/rq_a_binary: rows={len(df)}")
    else:
        pos_balanced = f"{data_dir}/processed/positives_balanced_{split_name}.parquet"
        neg_balanced = f"{data_dir}/processed/negatives_balanced_{split_name}.parquet"
        if os.path.exists(pos_balanced) and os.path.exists(neg_balanced):
            pos = pd.read_parquet(pos_balanced); pos["label"] = 1
            neg = pd.read_parquet(neg_balanced); neg["label"] = 0
            df = pd.concat([pos, neg], ignore_index=True)
            print(f"[{split_name}] FALLBACK balanced parquets pos={len(pos)} neg={len(neg)}")
        else:
            pos = pd.read_parquet(f"{data_dir}/splits/{split_name}.parquet"); pos["label"] = 1
            neg = pd.read_parquet(f"{data_dir}/processed/negatives_{split_name}.parquet"); neg["label"] = 0
            df = pd.concat([pos, neg], ignore_index=True)
            print(f"[{split_name}] FALLBACK legacy splits/ + negatives_{split_name}")

    # Build the model input text once. Always rebuild here — the column may
    # have been written by an earlier run with a different input_format.
    df["text"] = df.apply(lambda r: build_binary_input(r, input_format=input_format), axis=1)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    pr_auc = average_precision_score(labels, probs)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f, "pr_auc": pr_auc}

def eval_on_dataset(trainer, tokenizer, df, dataset_name, max_length, dump_path=None):
    """Run binary eval on an arbitrary labeled dataframe.

    If ``dump_path`` is given, also writes a per-row predictions parquet that the
    notebook's error-analysis section consumes.
    """
    def tok_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=max_length)
    ds = HFDataset.from_pandas(df[["text", "label"]]).map(tok_fn, batched=True, remove_columns=["text"])
    preds_out = trainer.predict(ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    probs = torch.softmax(torch.tensor(preds_out.predictions, dtype=torch.float32), dim=-1)[:, 1].numpy()
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    pr_auc = average_precision_score(labels, probs) if len(set(labels)) > 1 else 0.0
    result = {"dataset": dataset_name, "n": len(labels),
              "accuracy": float(accuracy_score(labels, preds)),
              "precision": float(p), "recall": float(r), "f1": float(f), "pr_auc": float(pr_auc),
              "confusion_matrix": confusion_matrix(labels, preds).tolist()}
    print(f"  {dataset_name}: F1={f:.4f} P={p:.4f} R={r:.4f} PR-AUC={pr_auc:.4f}")

    if dump_path is not None:
        keep = [c for c in ["dog_whistle", "dog_whistle_root", "ingroup", "content", "definition", "subreddit", "type"] if c in df.columns]
        out = df[keep].copy().reset_index(drop=True)
        out["label"] = labels
        out["pred"] = preds
        out["prob_coded"] = probs
        out["correct"] = (out["label"] == out["pred"]).astype(int)
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        out.to_parquet(dump_path, index=False)
        print(f"    wrote {len(out)} per-row predictions to {dump_path}")
    return result

def main():
    cfg, args = parse_args(extra_args=[
        {"flag": "--input_format",
         "kwargs": {"type": str, "default": None,
                    "choices": ["term", "term_enriched_def"],
                    "help": "Override input_format from binary.yaml (D6 ablation arm)."}},
    ])
    if args.input_format:
        cfg["input_format"] = args.input_format
    input_format = cfg.get("input_format", "term_enriched_def")
    if input_format not in ("term", "term_enriched_def"):
        raise ValueError(f"binary.yaml has unknown input_format={input_format!r}")
    seed = cfg["seed"]
    set_seed(seed)
    print_diagnostics()
    print(f"Input format (D6 arm): {input_format}")

    out_dir = os.path.join(cfg["output_dir"], f"format_{input_format}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = cfg["data_dir"]

    df_train = load_binary_split(data_dir, "train", seed, input_format=input_format)
    df_val = load_binary_split(data_dir, "val", seed, input_format=input_format)
    df_test = load_binary_split(data_dir, "test", seed, input_format=input_format)
    print(f"Data: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    max_len = cfg["max_length"]
    def tok_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=max_len)

    ds_train = HFDataset.from_pandas(df_train[["text", "label"]]).map(tok_fn, batched=True, remove_columns=["text"])
    ds_val = HFDataset.from_pandas(df_val[["text", "label"]]).map(tok_fn, batched=True, remove_columns=["text"])
    ds_test = HFDataset.from_pandas(df_test[["text", "label"]]).map(tok_fn, batched=True, remove_columns=["text"])
    del df_train, df_val; gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

    # Re-initialise newly-added pooler/classifier with small Gaussian weights.
    # Was added for DeBERTa-v3-large; harmless on RoBERTa. Walks all
    # nn.Linear submodules under model.pooler / model.classifier so it
    # handles both flat heads (DeBERTa: classifier is a Linear) and
    # multi-layer heads (RoBERTa: classifier has .dense + .out_proj).
    init_std = cfg.get("init_head_std", None)
    if init_std is not None:
        import torch.nn as nn
        n_inited = 0
        for head_name in ("pooler", "classifier"):
            head = getattr(model, head_name, None)
            if head is None:
                continue
            for sub in head.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.normal_(sub.weight, mean=0.0, std=init_std)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)
                    n_inited += 1
        print(f"[init] re-initialised {n_inited} Linear head module(s) with N(0, {init_std}^2)")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        label_smoothing_factor=cfg.get("label_smoothing_factor", 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=cfg.get("fp16", False),
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=cfg.get("num_workers", 4),
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg.get("report_to", "none"),
        seed=seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    trainer.train()

    # --- Evaluate on all three datasets ---
    all_results = {}

    # 1) Internal test split (dump per-row preds for Section 5c error analysis)
    all_results["test_split"] = eval_on_dataset(
        trainer, tokenizer, df_test, "test_split", max_len,
        dump_path=os.path.join(out_dir, "predictions", "test_split_preds.parquet"),
    )

    def _ensure_int_label(df):
        """The locked eval files already carry integer labels. The legacy code path
        assumed string labels and silently zeroed the frame. Detect dtype and only
        coerce when needed."""
        if pd.api.types.is_integer_dtype(df["label"]):
            return df.reset_index(drop=True)
        # legacy fallback for any "coded"/"non-coded" string columns
        m = {"coded": 1, "non-coded": 0, "literal": 0, "non_coded": 0}
        df = df.copy()
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(m)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        return df

    # 2) Locked detection set (101 rows). Schema: dog_whistle, content, definition[_enriched], label (int).
    print("Evaluating on locked detection set...")
    df_det = pd.read_parquet(f"{cfg['data_dir']}/final_data/rq_a_binary/eval_detection.parquet")
    df_det = _ensure_int_label(df_det)
    df_det["text"] = df_det.apply(lambda r: build_binary_input(r, input_format=input_format), axis=1)
    all_results["detection_101"] = eval_on_dataset(
        trainer, tokenizer, df_det, "detection_101", max_len,
        dump_path=os.path.join(out_dir, "predictions", "detection_101_preds.parquet"),
    )

    # 3) Locked disambiguation set (124 rows). Schema: dog_whistle, content, definition[_enriched], label (int).
    print("Evaluating on locked disambiguation set...")
    df_dis = pd.read_parquet(f"{cfg['data_dir']}/final_data/rq_a_binary/eval_disambiguation.parquet")
    df_dis = _ensure_int_label(df_dis)
    df_dis["text"] = df_dis.apply(lambda r: build_binary_input(r, input_format=input_format), axis=1)
    all_results["disambiguation_124"] = eval_on_dataset(
        trainer, tokenizer, df_dis, "disambiguation_124", max_len,
        dump_path=os.path.join(out_dir, "predictions", "disambiguation_124_preds.parquet"),
    )

    # Flatten for report_metrics.py compatibility
    flat_results = {}
    for ds_name, metrics in all_results.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                flat_results[f"{ds_name}_{k}"] = float(v)
    flat_results["_full"] = all_results

    flat_results["_input_format"] = input_format
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(flat_results, f, indent=2)
    trainer.save_model(os.path.join(out_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "best_model"))
    print(f"Done [{input_format}, seed {seed}]. Test F1={all_results['test_split']['f1']:.4f} | "
          f"detection_101 F1={all_results['detection_101']['f1']:.4f} | "
          f"disambiguation_124 F1={all_results['disambiguation_124']['f1']:.4f}")

if __name__ == "__main__":
    main()
