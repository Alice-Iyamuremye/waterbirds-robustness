"""Grad-CAM visualizations for qualitative model inspection (ResNet-18)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from erm.config import GROUP_NAMES, SALiency_DIR
from erm.data import group_id_from_metadata, load_waterbirds
from erm.evaluate import load_checkpoint
from erm.model import get_gradcam_target_layer

class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model; self.activations=None; self.gradients=None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)
    def _fwd_hook(self,module,inputs,output): self.activations=output.detach()
    def _bwd_hook(self,module,grad_input,grad_output): self.gradients=grad_output[0].detach()
    def __call__(self,x,class_idx):
        self.model.eval(); logits=self.model(x); score=logits[:,class_idx].sum()
        self.model.zero_grad(); score.backward()
        weights=self.gradients.mean(dim=(2,3),keepdim=True)
        cam=F.relu((weights*self.activations).sum(dim=1,keepdim=True)).squeeze().cpu().numpy()
        return (cam-cam.min())/(cam.max()-cam.min()+1e-8)

def denormalize(x):
    mean=torch.tensor([.485,.456,.406]).view(3,1,1); std=torch.tensor([.229,.224,.225]).view(3,1,1)
    return (x*std+mean).clamp(0,1)

def overlay_heatmap(rgb,heatmap):
    import matplotlib.cm as cm
    if heatmap.shape!=rgb.shape[:2]:
        t=torch.tensor(heatmap).float()[None,None]
        heatmap=F.interpolate(t,size=rgb.shape[:2],mode="bilinear",align_corners=False).squeeze().numpy()
    return np.clip(.5*rgb+.5*cm.jet(heatmap)[:,:,:3],0,1)

def _find_example_indices(dataset,target_group,n=1):
    found=[]
    for i in range(len(dataset)):
        _,_,meta=dataset[i]
        if group_id_from_metadata(meta)==target_group:
            found.append(i)
            if len(found)>=n: break
    return found

def generate_saliency_for_checkpoint(checkpoint,device="cuda",per_group=1):
    checkpoint=Path(checkpoint)
    device=torch.device(device if torch.cuda.is_available() else "cpu")
    model=load_checkpoint(checkpoint).to(device)
    cam=GradCAM(model,get_gradcam_target_layer(model)); subset=load_waterbirds("test")
    fig,axes=plt.subplots(per_group,len(GROUP_NAMES)*3,figsize=(3*len(GROUP_NAMES),3*per_group),squeeze=False)
    for col,gid in enumerate(range(len(GROUP_NAMES))):
        for row,idx in enumerate(_find_example_indices(subset,gid,n=per_group)):
            x,y,_=subset[idx]; rgb=denormalize(x).permute(1,2,0).cpu().numpy()
            heat=cam(x.unsqueeze(0).to(device),int(y)); overlay=overlay_heatmap(rgb,heat)
            for ax,img,title in [(axes[row,col*3],rgb,f"{GROUP_NAMES[gid]}\nlabel={int(y)}"),
                (axes[row,col*3+1],heat,"Grad-CAM"),(axes[row,col*3+2],overlay,"Overlay")]:
                ax.imshow(img,cmap="jet" if title=="Grad-CAM" else None); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"Grad-CAM — {checkpoint.stem}")
    fig.tight_layout(); out=SALiency_DIR/f"{checkpoint.stem}_saliency.png"
    fig.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)
    return str(out)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--per-group",type=int,default=1); p.add_argument("--device",default="cuda")
    a=p.parse_args(); generate_saliency_for_checkpoint(a.checkpoint,a.device,a.per_group)
