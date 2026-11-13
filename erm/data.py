"""WILDS Waterbirds loading and bird/background subgroup bookkeeping."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset
from erm.config import BATCH_SIZE,DATA_ROOT,GROUP_NAMES,NUM_GROUPS,NUM_WORKERS

def group_id_from_metadata(row):
    background=int(row[0]); bird=int(row[1]); return bird*2+background

def train_transform():
    return transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.7,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

def eval_transform():
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

def load_waterbirds(split):
    if split not in {"train","val","test"}: raise ValueError("split must be train, val, or test")
    ds=get_dataset(dataset="waterbirds",root_dir=str(DATA_ROOT),download=True)
    return ds.get_subset(split,transform=train_transform() if split=="train" else eval_transform())

def make_loader(subset,shuffle):
    return DataLoader(subset,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),drop_last=False)

def collect_group_predictions(model,loader,device):
    model.eval(); buckets={g:[] for g in range(NUM_GROUPS)}
    with torch.no_grad():
        for batch in loader:
            x,y,metadata=batch; x=x.to(device,non_blocking=True)
            preds=model(x).argmax(1).cpu().numpy(); labels=y.cpu().numpy(); meta=metadata.cpu().numpy()
            for p,t,m in zip(preds,labels,meta): buckets[group_id_from_metadata(m)].append((int(p),int(t)))
    return buckets

def accuracy_per_group(buckets):
    return {GROUP_NAMES[g]:(sum(p==t for p,t in items)/len(items) if items else float("nan")) for g,items in buckets.items()}
