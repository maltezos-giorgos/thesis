#!/usr/bin/env bash
# Download auxiliary datasets for Phase 2 hard-case mining.
# Idempotent: each section skips if target already exists.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW="$REPO_ROOT/data/raw"
mkdir -p "$DATA_RAW"

# ---------------------------------------------------------------------------
# 1. EntityQuestions (Princeton NLP) — test set
# Dataset is distributed as a zip (not in the repo itself).
# ---------------------------------------------------------------------------
EQ_DIR="$DATA_RAW/entity_questions"
EQ_DATASET="$EQ_DIR/dataset/test"
if [ -d "$EQ_DATASET" ]; then
    echo "[skip] EntityQuestions already at $EQ_DIR"
else
    mkdir -p "$EQ_DIR"
    echo "==> Downloading EntityQuestions dataset.zip..."
    wget -q --show-progress -O "$EQ_DIR/dataset.zip" \
        https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
    echo "==> Unzipping..."
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$EQ_DIR/dataset.zip" "$EQ_DIR"
    rm "$EQ_DIR/dataset.zip"
    N=$(find "$EQ_DATASET" -name '*.json' | wc -l)
    echo "    Done: $N relation test files"
fi

# ---------------------------------------------------------------------------
# 2. BEIR-NQ queries (2000) and 3. PopQA dev split — via HuggingFace datasets
# ---------------------------------------------------------------------------
echo "==> Downloading HuggingFace datasets..."
DATA_RAW="$DATA_RAW" python3 - <<'PYEOF'
import json, os, pathlib, sys

data_raw = pathlib.Path(os.environ["DATA_RAW"])

# Torchvision stub — required before sentence_transformers / transformers imports
import torch
try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from datasets import load_dataset

# ---- BEIR-NQ queries -------------------------------------------------------
nq_dir = data_raw / "beir_nq"
nq_file = nq_dir / "queries.jsonl"
if nq_file.exists():
    print(f"[skip] BEIR-NQ queries already at {nq_file}")
else:
    nq_dir.mkdir(parents=True, exist_ok=True)
    print("Loading BeIR/nq queries (first 2000)...")
    try:
        ds = load_dataset("BeIR/nq", "queries", trust_remote_code=True)
        queries_split = ds["queries"] if "queries" in ds else next(iter(ds.values()))
        n = min(2000, len(queries_split))
        rows = list(queries_split.select(range(n)))
    except Exception as exc:
        print(f"    BeIR/nq load failed ({exc}); falling back to natural_questions validation")
        ds = load_dataset("natural_questions", split="validation[:2000]", trust_remote_code=True)
        rows = [
            {"_id": str(r["id"]), "text": r["question"]["text"],
             "answers": [a["value"] for a in r["annotations"][0]["short_answers"] if a.get("value")]}
            for r in ds
        ]
    with open(nq_file, "w") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")
    print(f"    Saved {len(rows)} queries -> {nq_file}")

# ---- PopQA dev split -------------------------------------------------------
popqa_dir = data_raw / "popqa"
popqa_file = popqa_dir / "dev.jsonl"
if popqa_file.exists():
    print(f"[skip] PopQA dev already at {popqa_file}")
else:
    popqa_dir.mkdir(parents=True, exist_ok=True)
    print("Loading akariasai/PopQA (test split)...")
    ds = load_dataset("akariasai/PopQA", split="test", trust_remote_code=True)
    with open(popqa_file, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")
    print(f"    Saved {len(ds)} examples -> {popqa_file}")

print("==> All downloads complete.")
PYEOF
