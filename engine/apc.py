"""APC: calibrated, adaptation-aware prototype confusion control."""
import math
import numpy as np
import torch
import torch.nn.functional as F


class AdaptationAwarePrototypeController:
    def __init__(self,alpha=.05,calibration_trials=256,seed=1):
        self.alpha,self.calibration_trials,self.seed=float(alpha),int(calibration_trials),int(seed)
        if not 0<self.alpha<1 or self.calibration_trials<1: raise ValueError("Invalid APC calibration settings.")

    def calibrate(self,features_by_block,n_new_classes,k_shot):
        if not features_by_block: raise ValueError("APC needs base block features.")
        block=min(int(k) for k in features_by_block)
        item=features_by_block[block]
        x=F.normalize(torch.as_tensor(item["cls_in"]).float(),dim=-1).cpu()
        y=torch.as_tensor(item["labels"]).long().cpu()
        src=torch.as_tensor(item["source_ids"]).long().cpu()
        classes=np.unique(y.numpy())
        if len(classes)<2: raise ValueError("APC needs at least two base classes.")
        way=min(max(1,int(n_new_classes)),len(classes)-1)
        shot=len(torch.unique(src)) if int(k_shot)<=0 else int(k_shot)
        rows={}
        for c in classes:
            by_source={}
            for i in np.flatnonzero(y.numpy()==int(c)): by_source.setdefault(int(src[i]),int(i))
            rows[int(c)]=np.asarray(list(by_source.values()),dtype=np.int64)
        rng=np.random.default_rng(self.seed+block); scores=[]
        for _ in range(self.calibration_trials):
            protos={}
            for c in classes:
                candidates=rows[int(c)]
                chosen=rng.choice(candidates,size=min(shot,len(candidates)),replace=False)
                protos[int(c)]=F.normalize(x[torch.as_tensor(chosen)].mean(0),dim=0)
            new=rng.choice(classes,size=way,replace=False); ns={int(c) for c in new}
            old=[int(c) for c in classes if int(c) not in ns]
            scores.append((torch.stack([protos[int(c)] for c in new])@
                           torch.stack([protos[c] for c in old]).t()).flatten())
        scores=torch.cat(scores).sort().values
        if not torch.isfinite(scores).all(): raise FloatingPointError("Invalid APC calibration scores.")
        return {"version":1,"stable_block":block,"alpha":self.alpha,"pair_scores":scores.cpu(),
                "calibration_trials":self.calibration_trials,"seed":self.seed,
                "n_new_classes":way,"k_shot":shot,"num_base_classes":len(classes),"feature_dim":x.size(1)}

    @staticmethod
    def threshold(state,n_new,n_old):
        scores=torch.as_tensor(state["pair_scores"]).float().flatten().sort().values
        if not len(scores): raise ValueError("Empty APC null distribution.")
        q=1-float(state["alpha"])/max(1,int(n_new)*int(n_old))
        return float(scores[min(len(scores)-1,max(0,math.ceil(q*len(scores))-1))])

    @staticmethod
    def prototypes(item,class_ids):
        x=F.normalize(torch.as_tensor(item["cls_in"]).float(),dim=-1)
        y=torch.as_tensor(item["labels"]).long().reshape(-1)
        rows=[]
        for c in torch.as_tensor(class_ids).long().tolist():
            m=y.eq(int(c))
            if not m.any(): raise ValueError(f"Missing APC class {c}.")
            rows.append(F.normalize(x[m].mean(0),dim=0))
        return torch.stack(rows)

    def decide(self,features,old_anchors,old_ids,new_ids,state):
        if not isinstance(state,dict) or state.get("version")!=1: raise ValueError("Invalid APC state.")
        block=int(state["stable_block"])
        if block not in features or str(block) not in old_anchors: raise KeyError("APC stable block missing.")
        old=torch.as_tensor(old_ids).long().flatten(); new=torch.as_tensor(new_ids).long().flatten()
        if not len(old) or not len(new) or torch.isin(new,old).any(): raise ValueError("APC class IDs invalid.")
        pnew=self.prototypes(features[block],new)
        pold=F.normalize(torch.as_tensor(old_anchors[str(block)]).float(),dim=-1)
        if pold.shape!=(len(old),pnew.size(1)): raise ValueError("APC old prototypes misaligned.")
        sim=pnew@pold.t(); risk=float(sim.max()); threshold=self.threshold(state,len(new),len(old))
        return [{"block_idx":block,"expand":risk>threshold,"risk":risk,"threshold":threshold,
                 "class_scores":{int(c):float(v) for c,v in zip(new.tolist(),sim.max(1).values.tolist())},
                 "comparison_count":int(len(new)*len(old))}]
