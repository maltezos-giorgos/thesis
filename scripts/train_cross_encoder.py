"""Fine-tune cross-encoder/ms-marco-MiniLM-L-12-v2 on hard-cases training data.

Run data preparation first:
    python scripts/prepare_training_data.py

Then train:
    python scripts/train_cross_encoder.py [options]

Works on both GPU (Colab) and CPU (local, slower).
Best model is selected by validation accuracy and saved to models/cross_encoder_best/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Workaround for torchvision/NMS registration conflict (same fix as eval_llm_judge_full.py)
try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_OUTPUT = "models/cross_encoder_best"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder for CRAG retrieval evaluation.")
    parser.add_argument("--train",         default="data/training_v2/cross_encoder_train.jsonl")
    parser.add_argument("--val",           default="data/training_v2/cross_encoder_val.jsonl")
    parser.add_argument("--model",         default=BASE_MODEL)
    parser.add_argument("--output",        default=DEFAULT_OUTPUT)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1)
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--eval-steps",    type=int,   default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    logger.info("Device: %s  |  AMP: %s", device, use_amp)

    train_rows = load_jsonl(Path(args.train))
    val_rows   = load_jsonl(Path(args.val))
    logger.info("Train pairs: %d  |  Val pairs: %d", len(train_rows), len(val_rows))

    model = CrossEncoder(args.model, num_labels=1, max_length=512)

    train_examples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in train_rows
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Evaluator needs hard binary labels; smooth labels → hard for eval only
    val_examples = [
        InputExample(texts=[r["query"], r["passage"]], label=1 if r["label"] > 0.5 else 0)
        for r in val_rows
    ]
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name="val")

    total_steps  = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info("Total steps: %d  |  Warmup steps: %d", total_steps, warmup_steps)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        loss_fct=torch.nn.BCEWithLogitsLoss(),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        evaluation_steps=args.eval_steps,
        output_path=str(output_path),
        save_best_model=True,
        use_amp=use_amp,
        show_progress_bar=True,
    )

    logger.info("Training complete. Best model saved to %s", output_path)


if __name__ == "__main__":
    main()
