"""Train structured explanation generator with Flan-T5 + LoRA.

Saves LoRA adapter weights + tokenizer (not a merged full checkpoint).
To run inference, load the base model and apply the adapter.
"""
import os, json, gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate as hf_evaluate
from utils import parse_args, set_seed, print_diagnostics
from eval_generation import compute_metrics as eval_compute_metrics

def main():
    cfg, args = parse_args()
    seed = cfg["seed"]
    set_seed(seed)
    print_diagnostics()

    out_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = cfg["data_dir"]

    def _load_gen_split_hpc(data_dir, split_name):
        """Prefer final_data/rq_c_generation/ + build (input_text, target_json) inline; fall back to data/processed/generation_{split}.parquet.
        Carries through dog_whistle, dog_whistle_root, ingroup, definition, content, type so downstream error analysis can group on them."""
        final_p = f"{data_dir}/final_data/rq_c_generation/{split_name}.parquet"
        proc_p = f"{data_dir}/processed/generation_{split_name}.parquet"
        if os.path.exists(final_p):
            df_raw = pd.read_parquet(final_p)
            inputs, targets, dws, roots, ingroups, defns, contents = [], [], [], [], [], [], []
            for _, r in df_raw.iterrows():
                dw = str(r["dog_whistle"]); content = str(r["content"])
                root = str(r["dog_whistle_root"]); ingroup = str(r["ingroup"])
                defn = str(r["definition"]) if (pd.notna(r.get("definition")) and r.get("definition")) else "No standard definition available"
                inputs.append(f"Text: {content}\nMatched term: {dw}\nReturn JSON explaining the coded use.")
                expl = f"In this text, \'{dw}\' is used as a coded reference to {ingroup} ideology. {defn}"
                targets.append(json.dumps({"dog_whistle_root": root, "ingroup": ingroup, "definition": defn, "explanation": expl}))
                dws.append(dw); roots.append(root); ingroups.append(ingroup); defns.append(defn); contents.append(content)
            df_out = pd.DataFrame({
                "input_text": inputs, "target_json": targets,
                "dog_whistle": dws, "dog_whistle_root": roots, "ingroup": ingroups,
                "definition": defns, "content": contents,
            })
            if "type" in df_raw.columns:
                df_out["type"] = df_raw["type"].astype(str).values
            return df_out
        # Fallback: trust the processed parquet to already carry the needed columns
        return pd.read_parquet(proc_p)
    df_train = _load_gen_split_hpc(data_dir, "train")
    df_val = _load_gen_split_hpc(data_dir, "val")
    df_test = _load_gen_split_hpc(data_dir, "test")
    print(f"Data: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Optional: oversample under-represented ingroups in train to attack the
    # class-prior bias (anti-liberal is 5.8% of train but 34% of test). Only
    # touches train; val and test stay untouched.
    cb = cfg.get("class_balance_train") or "none"
    if cb != "none" and "ingroup" in df_train.columns:
        before = df_train["ingroup"].value_counts().to_dict()
        if cb == "match_max":
            target = int(df_train["ingroup"].value_counts().max())
        elif cb == "median":
            target = int(df_train["ingroup"].value_counts().median())
        else:
            try:
                target = int(cb)
            except (TypeError, ValueError):
                raise ValueError(f"class_balance_train must be 'none', 'match_max', 'median' or an int — got {cb!r}")
        parts = []
        for ig, sub in df_train.groupby("ingroup"):
            if len(sub) >= target:
                parts.append(sub.sample(n=target, random_state=seed, replace=False))
            else:
                parts.append(sub.sample(n=target, random_state=seed, replace=True))
        df_train = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        after = df_train["ingroup"].value_counts().to_dict()
        print(f"class_balance_train={cb} -> target={target} per ingroup; train rows {sum(before.values())} -> {len(df_train)}")
        print(f"  before: {before}")
        print(f"  after:  {after}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    max_input = cfg.get("max_input_len", 512)
    max_target = cfg.get("max_target_len", 128)

    def tok_fn(examples):
        inputs = [str(t) for t in examples["input_text"]]
        targets = [str(t) for t in examples["target_json"]]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_train = HFDataset.from_pandas(df_train[["input_text", "target_json"]]).map(tok_fn, batched=True, remove_columns=["input_text", "target_json"])
    ds_val = HFDataset.from_pandas(df_val[["input_text", "target_json"]]).map(tok_fn, batched=True, remove_columns=["input_text", "target_json"])
    ds_test = HFDataset.from_pandas(df_test[["input_text", "target_json"]]).map(tok_fn, batched=True, remove_columns=["input_text", "target_json"])

    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    rouge = hf_evaluate.load("rouge")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        predict_with_generate=True,
        generation_max_length=max_target,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=False,
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=cfg.get("num_workers", 4),
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg.get("report_to", "none"),
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    trainer.train()

    # Evaluate: generate on test
    gen_preds = trainer.predict(ds_test)
    pred_ids = gen_preds.predictions
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    targets = df_test["target_json"].tolist()

    # Build per-row predictions frame; eval_generation handles brace-repair, JSON parse,
    # field-level classification metrics (L3) + TF-IDF cosine on explanation (L5) + ROUGE-L.
    pred_df = df_test[[c for c in ["dog_whistle", "dog_whistle_root", "ingroup", "definition", "content", "type", "input_text", "target_json"] if c in df_test.columns]].copy()
    pred_df["pred_text"] = decoded_preds

    results, pred_df = eval_compute_metrics(pred_df)
    results["sample_outputs"] = [
        {
            "input": str(df_test.iloc[i].get("input_text", ""))[:200],
            "predicted": decoded_preds[i][:300],
            "target": targets[i][:300],
        }
        for i in range(min(5, len(decoded_preds)))
    ]

    pred_df.to_parquet(os.path.join(out_dir, "test_predictions.parquet"), index=False)
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    model.save_pretrained(os.path.join(out_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "best_model"))

    ig = results.get("ingroup", {})
    print(
        f"Done. parse={results['json_parse_rate']:.3f}  "
        f"ingroup acc={ig.get('accuracy', 0):.3f} macro_f1={ig.get('macro_f1', 0):.3f}  "
        f"expl_cos={results.get('explanation_cosine_mean', 0):.3f}  "
        f"expl_rougeL={results.get('explanation_rouge_l_mean', 0):.3f}"
    )

if __name__ == "__main__":
    main()
