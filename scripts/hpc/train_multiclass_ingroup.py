"""Train multiclass ingroup classifier (RQ-B).

Two input arms (see DESIGN_DEFENSE.md / RQ-B section):
  - "term":      Text + matched dog-whistle term
  - "text_only": Text only (ablation: how much of the ingroup signal is
                 recoverable from context without seeing the surface form?)

Methodological notes:
- The label space is built from TRAIN ONLY. Val/test rows whose ingroup
  is not in train are dropped (they are unlearnable from the train fold
  and would silently inject zero-support classes into macro-F1). On the
  current grouped split that drops 10 val rows ("misogynistic"); test
  is unaffected.
- macro-F1 is reported over the labels actually present in
  y_true U y_pred (sklearn default when `labels=` is not passed).
  Headline metric is macro-F1; weighted-F1 + per-class support are
  reported alongside because the grouped split produces a strong prior
  shift on `anti-liberal` between train (~6%) and test (~34%).
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
from utils import parse_args, set_seed, print_diagnostics


class WeightedLossTrainer(Trainer):
    """Trainer subclass that uses CE loss weighted by `class_weights` (Run D)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights  # 1-D float tensor, len == n_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_mc_input(row, input_format="term"):
    """Format one row for the multiclass classifier."""
    content = str(row["content"]) if row["content"] is not None else ""
    if input_format == "term":
        dw = str(row["dog_whistle"]) if row["dog_whistle"] is not None else ""
        return (f"Text: {content}\n"
                f"Matched term: {dw}\n"
                f"Predict the ingroup category.")
    if input_format == "text_only":
        return (f"Text: {content}\n"
                f"Predict the ingroup category.")
    raise ValueError(f"Unknown input_format: {input_format!r} "
                     "(expected 'term' or 'text_only')")


def load_mc_split(data_dir, split_name, data_subdir="rq_b_multiclass"):
    """Read RQ-B data from final_data/<data_subdir>/, falling back to splits/."""
    final_path = f"{data_dir}/final_data/{data_subdir}/{split_name}.parquet"
    if os.path.exists(final_path):
        df = pd.read_parquet(final_path)
        print(f"[{split_name}] using final_data/{data_subdir}: rows={len(df)}")
    else:
        df = pd.read_parquet(f"{data_dir}/splits/{split_name}.parquet")
        print(f"[{split_name}] FALLBACK splits/{split_name}.parquet: rows={len(df)}")
    return df


def main():
    cfg, args = parse_args(extra_args=[
        {"flag": "--input_format",
         "kwargs": {"type": str, "default": None,
                    "choices": ["term", "text_only"],
                    "help": "Override input_format from multiclass.yaml."}},
        {"flag": "--data_subdir",
         "kwargs": {"type": str, "default": None,
                    "help": "Override final_data/<subdir>; default 'rq_b_multiclass'. "
                            "Use 'rq_b_multiclass_random' for the alt-split sanity run (Run C)."}},
        {"flag": "--class_weighted_loss",
         "kwargs": {"action": "store_true",
                    "help": "Use sklearn-balanced class weights in CE loss (Run D)."}},
        {"flag": "--run_tag",
         "kwargs": {"type": str, "default": None,
                    "help": "Optional tag inserted into the output dir to keep runs separate "
                            "(e.g. 'altsplit', 'weighted'). Default: empty."}},
    ])
    if args.input_format:
        cfg["input_format"] = args.input_format
    if args.data_subdir:
        cfg["data_subdir"] = args.data_subdir
    if args.class_weighted_loss:
        cfg["class_weighted_loss"] = True
    input_format = cfg.get("input_format", "term")
    data_subdir = cfg.get("data_subdir", "rq_b_multiclass")
    class_weighted_loss = bool(cfg.get("class_weighted_loss", False))
    run_tag = args.run_tag or cfg.get("run_tag", "")
    if input_format not in ("term", "text_only"):
        raise ValueError(f"multiclass.yaml has unknown input_format={input_format!r}")
    seed = cfg["seed"]
    set_seed(seed)
    print_diagnostics()
    print(f"Input format: {input_format} | data_subdir: {data_subdir} | "
          f"class_weighted_loss: {class_weighted_loss} | run_tag: {run_tag!r}")

    fmt_dir = f"format_{input_format}" if not run_tag else f"format_{input_format}_{run_tag}"
    out_dir = os.path.join(cfg["output_dir"], fmt_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = cfg["data_dir"]

    # Load raw splits.
    dfs = {s: load_mc_split(data_dir, s, data_subdir=data_subdir) for s in ["train", "val", "test"]}

    # ---- Train-only label space (Fix #1) -------------------------------
    train_ingroups = sorted(dfs["train"]["ingroup"].unique().tolist())
    label2id = {name: i for i, name in enumerate(train_ingroups)}
    id2label = {i: name for name, i in label2id.items()}
    n_labels = len(label2id)
    print(f"Label space (train only): {n_labels} ingroups")

    # Drop val/test rows whose ingroup is not in train (Fix #1 b).
    drop_report = {}
    for s in ("val", "test"):
        before = len(dfs[s])
        unseen_mask = ~dfs[s]["ingroup"].isin(label2id)
        unseen_counts = dfs[s].loc[unseen_mask, "ingroup"].value_counts().to_dict()
        dfs[s] = dfs[s].loc[~unseen_mask].reset_index(drop=True)
        after = len(dfs[s])
        drop_report[s] = {"before": int(before), "after": int(after),
                          "dropped": int(before - after),
                          "unseen_label_counts": {k: int(v) for k, v in unseen_counts.items()}}
        if before != after:
            print(f"[{s}] dropped {before - after} rows with unseen-in-train ingroups: {unseen_counts}")
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id,
                   "id2label": {str(k): v for k, v in id2label.items()},
                   "drop_report": drop_report}, f, indent=2)

    # Build text + integer label.
    for s in dfs:
        dfs[s]["text"] = dfs[s].apply(lambda r: build_mc_input(r, input_format=input_format), axis=1)
        dfs[s]["label"] = dfs[s]["ingroup"].map(label2id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    max_len = cfg["max_length"]

    def tok_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=max_len)

    datasets = {}
    for s in ["train", "val", "test"]:
        datasets[s] = HFDataset.from_pandas(dfs[s][["text", "label"]]).map(
            tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=n_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # No `labels=` kwarg -> macro-F1 over labels present in
        # y_true U y_pred (sklearn default). This avoids 0-support classes
        # from dragging the macro mean toward 0.
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=cfg.get("num_workers", 4),
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg.get("report_to", "none"),
        seed=seed,
    )

    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    if class_weighted_loss:
        # sklearn-balanced weights: w_c = N / (K * count_c). Equalises the
        # contribution of each class to the empirical loss.
        train_label_ids = dfs["train"]["label"].to_numpy()
        counts = np.bincount(train_label_ids, minlength=n_labels).astype(np.float64)
        weights_np = len(train_label_ids) / (n_labels * np.maximum(counts, 1.0))
        weights_t = torch.tensor(weights_np, dtype=torch.float32)
        print(f"[weighted-CE] class weights (min={weights_np.min():.3f} "
              f"max={weights_np.max():.3f}): "
              f"{ {id2label[i]: round(float(w), 3) for i, w in enumerate(weights_np)} }")
        trainer = WeightedLossTrainer(class_weights=weights_t, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # ---- Test eval ------------------------------------------------------
    preds_out = trainer.predict(datasets["test"])
    logits = preds_out.predictions
    preds = np.argmax(logits, axis=-1)
    labels = preds_out.label_ids

    present_label_ids = sorted(set(labels.tolist()) | set(preds.tolist()))
    present_label_names = [id2label[i] for i in present_label_ids]
    cls_report = classification_report(
        labels, preds,
        labels=present_label_ids,
        target_names=present_label_names,
        zero_division=0,
        output_dict=True,
    )

    results = {
        "input_format": input_format,
        "data_subdir": data_subdir,
        "class_weighted_loss": class_weighted_loss,
        "run_tag": run_tag,
        "seed": seed,
        "n_test": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "classification_report": cls_report,
        "confusion_matrix": confusion_matrix(labels, preds, labels=present_label_ids).tolist(),
        "confusion_matrix_label_order": present_label_names,
        "drop_report": drop_report,
    }
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Per-row predictions for error analysis (mirrors RQ-A layout).
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    keep = [c for c in ["dog_whistle", "dog_whistle_root", "ingroup", "content", "subreddit", "type"]
            if c in dfs["test"].columns]
    out_df = dfs["test"][keep].copy().reset_index(drop=True)
    out_df["label"] = labels
    out_df["pred"] = preds
    out_df["pred_label"] = [id2label[int(p)] for p in preds]
    out_df["correct"] = (out_df["label"] == out_df["pred"]).astype(int)
    out_df["pred_prob"] = probs.max(axis=1)
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_df.to_parquet(os.path.join(pred_dir, "test_preds.parquet"), index=False)
    print(f"Wrote {len(out_df)} per-row predictions to {pred_dir}/test_preds.parquet")

    trainer.save_model(os.path.join(out_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "best_model"))
    print(f"Done [{input_format}, seed {seed}]. "
          f"Test macro-F1={results['f1_macro']:.4f}  weighted-F1={results['f1_weighted']:.4f}  "
          f"accuracy={results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
