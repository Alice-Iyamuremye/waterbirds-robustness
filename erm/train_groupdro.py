"""Group DRO training with absent-minibatch groups handled explicitly."""
from __future__ import annotations
import json, random
from dataclasses import asdict, dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from erm.config import LEARNING_RATE, MODEL_DIR, NUM_EPOCHS, NUM_GROUPS, SEED, WEIGHT_DECAY
from erm.data import accuracy_per_group, collect_group_predictions, load_waterbirds, make_loader
from erm.model import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class GroupDROLoss:
    """Exponentiated-gradient Group DRO; absent groups are not assigned zero loss."""
    def __init__(self, num_groups, robust_step_size=0.01):
        if num_groups < 1 or robust_step_size < 0: raise ValueError("Invalid Group DRO configuration")
        self.num_groups = num_groups
        self.robust_step_size = robust_step_size
        self.q = torch.full((num_groups,), 1.0 / num_groups)

    def __call__(self, logits, labels, group_ids):
        if not (len(logits) == len(labels) == len(group_ids)):
            raise ValueError("logits, labels, and group_ids must have equal batch size")
        if torch.any((group_ids < 0) | (group_ids >= self.num_groups)):
            raise ValueError("group_ids contains an out-of-range index")
        if self.q.device != logits.device: self.q = self.q.to(logits.device)
        losses = F.cross_entropy(logits, labels, reduction="none")
        means = torch.zeros(self.num_groups, device=logits.device, dtype=losses.dtype)
        present = torch.zeros(self.num_groups, device=logits.device, dtype=torch.bool)
        for gid in range(self.num_groups):
            mask = group_ids == gid
            if mask.any():
                means[gid] = losses[mask].mean()
                present[gid] = True
        with torch.no_grad():
            updated = self.q.clone()
            updated[present] *= torch.exp(self.robust_step_size * means[present].detach())
            self.q = updated / updated.sum().clamp_min(torch.finfo(updated.dtype).tiny)
        active_q = self.q[present]
        active_q = active_q / active_q.sum().clamp_min(torch.finfo(active_q.dtype).tiny)
        return torch.sum(active_q * means[present])

@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_avg_acc: float
    val_worst_group_acc: float
    val_per_group: dict

def train_groupdro(device="cuda"):
    set_seed(SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train = load_waterbirds("train"); val = load_waterbirds("val")
    train_loader = make_loader(train, shuffle=True); val_loader = make_loader(val, shuffle=False)
    model = build_model().to(device)
    criterion = GroupDROLoss(NUM_GROUPS, robust_step_size=0.01)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    history = []; best_worst = -1.0; best_path = MODEL_DIR / "groupdro_best.pt"
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); running = 0.0; batches = 0
        for x, y, metadata in tqdm(train_loader, desc=f"epoch {epoch:02d} train"):
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); metadata=metadata.to(device)
            groups = metadata[:,1].long()*2 + metadata[:,0].long()
            optimizer.zero_grad(set_to_none=True); loss=criterion(model(x), y, groups)
            loss.backward(); optimizer.step(); running += float(loss.detach()); batches += 1
        scheduler.step()
        buckets=collect_group_predictions(model,val_loader,device); per=accuracy_per_group(buckets)
        valid=[v for v in per.values() if not np.isnan(v)]
        if not valid: raise RuntimeError("Validation contains no observed groups")
        avg=float(np.mean(valid)); worst=float(min(valid))
        history.append(EpochLog(epoch,running/max(batches,1),avg,worst,per))
        print(f"[epoch {epoch:02d}] train_loss={running/max(batches,1):.4f} val_macro_group_acc={avg:.4f} val_worst_group={worst:.4f}")
        if worst > best_worst:
            best_worst=worst
            torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                "scheduler_state":scheduler.state_dict(),"group_weights":criterion.q.detach().cpu(),
                "epoch":epoch,"seed":SEED,"val_worst_group_acc":worst,
                "val_macro_group_acc":avg,"val_per_group":per,
                "robust_step_size":criterion.robust_step_size},best_path)
    history_path=MODEL_DIR/"groupdro_history.json"
    history_path.write_text(json.dumps([asdict(x) for x in history],indent=2))
    return {"best_checkpoint":str(best_path),"best_worst_group_val_acc":best_worst,"history_path":str(history_path)}

if __name__ == "__main__": train_groupdro()
