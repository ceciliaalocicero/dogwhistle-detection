"""Shared utilities for HPC training scripts."""
import os, sys, json, hashlib, random, gc, yaml, argparse
import numpy as np
import pandas as pd
import torch

def load_config(yaml_path, overrides=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    if cfg.get("bf16") == "auto":
        cfg["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    elif cfg.get("bf16") == "false" or cfg.get("bf16") is False:
        cfg["bf16"] = False
    return cfg

def parse_args(extra_args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--model_path", type=str, default=None)
    if extra_args:
        for arg in extra_args:
            p.add_argument(arg["flag"], **arg["kwargs"])
    args = p.parse_args()
    cfg = load_config(args.config, {
        "seed": args.seed,
        "sample_size": args.sample_size,
    })
    if args.model_path:
        cfg["model_path"] = args.model_path
    return cfg, args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_diagnostics():
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16: {torch.cuda.is_bf16_supported()}")
    print(f"CPUs: {os.cpu_count()}")

def sanitize_text_column(df, col):
    before = len(df)
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  sanitize {col}: dropped {dropped:,} rows")
    return df

def content_hash(text):
    normalized = " ".join(str(text).lower().strip().split())
    return hashlib.sha256(normalized.encode()).hexdigest()

def get_content_hashes(df, text_col="content"):
    return set(df[text_col].apply(content_hash))
