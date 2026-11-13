"""End-to-end ERM pipeline driver."""
from __future__ import annotations
import argparse
from pathlib import Path
from erm.config import MODEL_DIR
from erm.evaluate import evaluate_on_test
from erm.saliency import generate_saliency_for_checkpoint
from erm.train_erm import train_erm

def main(skip_train=False,device="cuda"):
    ckpt=Path(MODEL_DIR/"erm_best.pt")
    if not skip_train:
        info=train_erm(device=device); ckpt=Path(info["best_checkpoint"])
    if not ckpt.is_file(): raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    evaluate_on_test(ckpt,device=device)
    generate_saliency_for_checkpoint(ckpt,device=device,per_group=1)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--skip-train",action="store_true"); p.add_argument("--device",default="cuda")
    a=p.parse_args(); main(skip_train=a.skip_train,device=a.device)
