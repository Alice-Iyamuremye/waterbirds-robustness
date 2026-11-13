"""Print explicit Waterbirds split and subgroup counts for auditability."""
from erm.config import GROUP_NAMES
from erm.data import group_id_from_metadata,load_waterbirds

def inspect(split="train"):
    ds=load_waterbirds(split); counts=[0]*len(GROUP_NAMES)
    for i in range(len(ds)):
        _,_,meta=ds[i]; counts[group_id_from_metadata(meta)]+=1
    total=sum(counts); print(f"{split}: n={total}")
    for name,count in zip(GROUP_NAMES,counts): print(f"{name}: {count} ({count/total:.1%})")
if __name__=="__main__": inspect()
