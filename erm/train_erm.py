"""Empirical Risk Minimization training baseline."""
from __future__ import annotations
import json,random
from dataclasses import asdict,dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from erm.config import LEARNING_RATE,MODEL_DIR,NUM_EPOCHS,SEED,WEIGHT_DECAY
from erm.data import accuracy_per_group,collect_group_predictions,load_waterbirds,make_loader
from erm.model import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
@dataclass
class EpochLog:
    epoch:int; train_loss:float; val_avg_acc:float; val_worst_group_acc:float; val_per_group:dict

def train_erm(device="cuda"):
    set_seed(SEED); device=torch.device(device if torch.cuda.is_available() else "cpu")
    train=load_waterbirds("train"); val=load_waterbirds("val")
    train_loader=make_loader(train,True); val_loader=make_loader(val,False)
    model=build_model().to(device); criterion=nn.CrossEntropyLoss()
    optimizer=AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS)
    history=[]; best=-1.0; path=MODEL_DIR/"erm_best.pt"
    for epoch in range(1,NUM_EPOCHS+1):
        model.train(); running=0.; batches=0
        for x,y,_ in tqdm(train_loader,desc=f"epoch {epoch:02d} train"):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); optimizer.step()
            running+=float(loss.detach()); batches+=1
        scheduler.step(); buckets=collect_group_predictions(model,val_loader,device); per=accuracy_per_group(buckets)
        valid=[v for v in per.values() if not np.isnan(v)]
        if not valid: raise RuntimeError("Validation contains no observed groups")
        avg=float(np.mean(valid)); worst=float(min(valid)); history.append(EpochLog(epoch,running/max(batches,1),avg,worst,per))
        print(f"[epoch {epoch:02d}] train_loss={running/max(batches,1):.4f} val_macro_group_acc={avg:.4f} val_worst_group={worst:.4f}")
        if worst>best:
            best=worst; torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"epoch":epoch,"seed":SEED,"val_worst_group_acc":worst,"val_macro_group_acc":avg,"val_per_group":per},path)
    hist=MODEL_DIR/"erm_history.json"; hist.write_text(json.dumps([asdict(x) for x in history],indent=2))
    return {"best_checkpoint":str(path),"best_worst_group_val_acc":best,"history_path":str(hist)}
if __name__=="__main__": train_erm()
