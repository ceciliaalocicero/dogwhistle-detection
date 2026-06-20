"""Stage 2 — LLM-as-judge over Stage-1 candidates using vLLM.

Loads negatives_stage1_raw.parquet, batches the disambiguation prompt through
vLLM (default: meta-llama/Llama-3.1-8B-Instruct), parses 'coded' / 'non-coded',
writes negatives_stage2_judged.parquet with judge_label + judge_confidence.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

JUDGE_PROMPT = (
    "You are an expert linguist labelling whether a phrase in a Reddit comment "
    "is being used as a coded dog whistle.\n\n"
    "A dog whistle is a phrase that, on its surface, has an innocuous meaning "
    "but is also used to signal a hidden, coded meaning to an in-group audience.\n\n"
    "Phrase: \"{dog_whistle}\"\n"
    "The coded (in-group) meaning of this phrase is: {definition}\n\n"
    "Reddit comment:\n\"\"\"\n{content}\n\"\"\"\n\n"
    "Question: In this Reddit comment, is the phrase \"{dog_whistle}\" being used "
    "with the coded meaning above, or with its ordinary literal meaning?\n\n"
    "Answer with exactly one word: 'coded' or 'non-coded'."
)


def parse(text):
    head = (text or "").strip().lower()[:60]
    if "non-coded" in head or "non coded" in head or "noncoded" in head:
        return "non-coded", 0.9
    if re.search(r"\bcoded\b", head):
        return "coded", 0.9
    return "unparseable", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_content_chars", type=int, default=2000)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    data = Path(args.data_dir)
    cand = pd.read_parquet(data / "processed" / "negatives_stage1_raw.parquet")
    print(f"adjudicating {len(cand):,} candidates with {args.model}")

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="auto",
              gpu_memory_utilization=0.9, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = [
        JUDGE_PROMPT.format(
            dog_whistle=r.dog_whistle,
            definition=r.definition or "no in-group meaning provided",
            content=r.content[:args.max_content_chars],
        )
        for r in cand.itertuples(index=False)
    ]
    # Use chat template if the tokenizer supports it
    tok = llm.get_tokenizer()
    if hasattr(tok, "apply_chat_template"):
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]

    outputs = llm.generate(prompts, sp)
    parsed = [parse(o.outputs[0].text) for o in outputs]
    cand["judge_label"] = [p[0] for p in parsed]
    cand["judge_confidence"] = [p[1] for p in parsed]
    cand["judge_model"] = args.model

    out = data / "processed" / "negatives_stage2_judged.parquet"
    cand.to_parquet(out)
    print(f"wrote {out}")
    print(cand["judge_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
