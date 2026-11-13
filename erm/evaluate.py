"""Subgroup-aware test evaluation; reports sample-weighted and macro-group accuracy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from erm.config import GROUP_NAMES, NUM_GROUPS, RESULT_DIR
from erm.data import accuracy_per_group, collect_group_predictions, load_waterbirds, make_loader
from erm.model import build_model

def load_checkpoint(path):
    path=Path(path)
    model=build_model()
    ckpt=torch.load(path,map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model

def evaluate_on_test(checkpoint, device="cuda", save_report=True):
    checkpoint=Path(checkpoint)
    device=torch.device(device if torch.cuda.is_available() else "cpu")
    model=load_checkpoint(checkpoint).to(device)
    subset=load_waterbirds("test"); loader=make_loader(subset,shuffle=False)
    buckets=collect_group_predictions(model,loader,device)
    per=accuracy_per_group(buckets)
    valid={k:v for k,v in per.items() if not np.isnan(v)}
    if not valid: raise RuntimeError("No test subgroup observations were collected")
    total=sum(len(items) for items in buckets.values())
    correct=sum(sum(int(p==t) for p,t in items) for items in buckets.values())
    sample_weighted=correct/total if total else float("nan")
    macro=float(np.mean(list(valid.values())))
    worst_name=min(valid,key=valid.get); worst=valid[worst_name]
    counts={GROUP_NAMES[g]:len(buckets[g]) for g in range(NUM_GROUPS)}
    print("\n=== Test Subgroup Accuracy ===")
    for name,acc in per.items(): print(f"  {name:<22s} {acc:.4f}")
    print(f"  {'MACRO GROUP AVG':<22s} {macro:.4f}")
    print(f"  {'SAMPLE-WEIGHTED':<22s} {sample_weighted:.4f}")
    print(f"  {'WORST GROUP':<22s} {worst:.4f} ({worst_name})")
    report={"checkpoint":str(checkpoint),"overall_accuracy":sample_weighted,
        "sample_weighted_accuracy":sample_weighted,"macro_group_accuracy":macro,
        "num_test_examples":total,"num_correct":correct,"worst_group_name":worst_name,
        "worst_group_accuracy":worst,"per_group_accuracy":per,"per_group_counts":counts}
    if save_report:
        out=RESULT_DIR/f"{checkpoint.stem}_test_report.json"
        out.write_text(json.dumps(report,indent=2)); print(f"[report] -> {out}")
    return report

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--device",default="cuda")
    a=p.parse_args(); evaluate_on_test(a.checkpoint,device=a.device)
