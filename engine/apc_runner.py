"""APC-controlled dynamic MoE FSCIL runner."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from engine.apc import AdaptationAwarePrototypeController
from engine.engine import Runner, _cfg_value


class APCRunner(Runner):
    def __init__(self,cfg):
        c=cfg.TRAINER.BiMC.APC
        self.apc_controller=AdaptationAwarePrototypeController(
            c.ALPHA,c.CALIBRATION_TRIALS,c.SEED if c.SEED>=0 else cfg.SEED)
        super().__init__(cfg)

    @staticmethod
    def _adapt_cls_anchors(adapter, normalized_cls):
        """Apply exactly the adapter's CLS residual to stored pre-adapter CLS inputs."""
        x=torch.as_tensor(normalized_cls,device=adapter.residual_scale.device,
                          dtype=adapter.residual_scale.dtype)
        _,base=adapter.router(x,adapter.base_expert_count)
        active=list(range(adapter.base_expert_count,adapter.num_experts))
        if active:
            gates=torch.cat([torch.sigmoid(adapter.router.columns[i](x)) for i in active],-1)
            inc=gates/(1.0+gates.sum(-1,keepdim=True))
            weights=torch.cat([base,inc],-1)
        else: weights=base
        experts=list(range(adapter.base_expert_count))+active
        residual=torch.zeros_like(x)
        for col,eid in enumerate(experts):
            residual=residual+adapter.experts[eid](x)*weights[:,col:col+1]
        return F.normalize((x+adapter.residual_scale.to(x.dtype)*residual).float(),dim=-1)

    @staticmethod
    def _cpu_class_metadata(class_ids,class_counts):
        """Normalize historical class IDs/counts to CPU before label expansion.

        These values are bookkeeping metadata used with CPU-cached anchor
        features; keep them on CPU even when checkpoint class_ids were saved
        from CUDA.
        """
        ids=torch.as_tensor(class_ids,dtype=torch.long).detach().cpu().reshape(-1)
        counts=torch.as_tensor(class_counts,dtype=torch.long).detach().cpu().reshape(-1)
        if ids.numel()==0 or counts.numel()!=ids.numel() or torch.any(counts<=0):
            raise ValueError("Historical class IDs and positive anchor counts must align.")
        labels=torch.repeat_interleave(ids,counts)
        return ids,counts,labels

    @staticmethod
    def _class_input_bank(item,class_ids):
        x=torch.as_tensor(item["cls_in"]).float().cpu()
        y=torch.as_tensor(item["labels"]).long().cpu()
        ids=torch.as_tensor(class_ids).long().cpu()
        counts=[]
        rows=[]
        for c in ids.tolist():
            selected=x[y.eq(c)]
            if selected.numel()==0: raise ValueError(f"Missing stable-block inputs for class {c}.")
            rows.append(selected)
            counts.append(len(selected))
        return torch.cat(rows,0),counts

    @staticmethod
    def _merge_block_input_anchors(states,key):
        """Merge per-class historical CLS anchors in global class-ID order.

        Class IDs, counts and cached anchors are bookkeeping/replay metadata.
        Normalize all three to CPU before concatenation and slicing, regardless
        of the device on which a task checkpoint was originally serialized.
        """
        ids_by_state=[]
        class_chunks=[]
        counts_by_class=[]
        feature_dim=None

        for state in states:
            ids=torch.as_tensor(state["class_ids"],dtype=torch.long).detach().cpu().reshape(-1)
            anchors=torch.as_tensor(
                state["block_input_anchors"][key],dtype=torch.float32
            ).detach().cpu()
            counts=torch.as_tensor(
                state["block_input_counts"][key],dtype=torch.long
            ).detach().cpu().reshape(-1)

            if anchors.ndim!=2:
                raise ValueError("Stable-block CLS anchors must be a two-dimensional tensor.")
            if feature_dim is None:
                feature_dim=anchors.size(1)
            elif anchors.size(1)!=feature_dim:
                raise ValueError("Stable-block CLS anchor feature dimensions differ.")
            if ids.numel()==0 or counts.numel()!=ids.numel() or torch.any(counts<=0):
                raise ValueError("Every class must have a positive, aligned anchor count.")
            if int(counts.sum().item())!=anchors.size(0):
                raise ValueError("Stable-block anchor counts do not match anchor rows.")

            ids_by_state.append(ids)
            offset=0
            for count in counts.tolist():
                class_chunks.append(anchors[offset:offset+count])
                counts_by_class.append(int(count))
                offset+=count

        if not ids_by_state:
            raise ValueError("Cannot merge an empty stable-block anchor bank.")
        all_ids=torch.cat(ids_by_state,dim=0)
        if torch.unique(all_ids).numel()!=all_ids.numel():
            raise ValueError("Duplicate class ids in stable-block anchor bank.")
        if len(class_chunks)!=all_ids.numel():
            raise ValueError("Stable-block anchor chunks do not align with class IDs.")

        order=all_ids.argsort()
        ordered_anchors=torch.cat([class_chunks[i] for i in order.tolist()],dim=0)
        ordered_counts=torch.tensor(
            [counts_by_class[i] for i in order.tolist()],dtype=torch.long
        )
        return all_ids[order],ordered_anchors,ordered_counts

    def merge_dicts(self,states):
        merged=super().merge_dicts(states)
        if not any("block_input_anchors" in s for s in states): return merged
        if not all("block_input_anchors" in s and "block_input_counts" in s for s in states):
            raise ValueError("Every seen class must carry stable-block input anchors.")
        key=str(self.apc_state["stable_block"])
        _,anchors,counts=self._merge_block_input_anchors(states,key)
        merged["block_input_anchors"]={key:anchors}
        merged["block_input_counts"]={key:counts}
        return merged

    def _ensure_apc_ready(self,states):
        if not states or not self._moe_adapters(): return
        if self.apc_state is not None and "block_input_anchors" in states[0]: return
        loader=self.data_manager.get_dataloader(0,source="train",mode="test",accumulate_past=False)
        self.model.eval()
        features=self.model_without_dp.collect_blockwise_cls_features(loader)
        stable=min(features)
        adapter=self._adapter_by_block(stable)
        adapted=dict(features[stable])
        adapted["cls_in"]=self._adapt_cls_anchors(adapter,adapted["cls_in"]).cpu()
        if self.apc_state is None:
            self.apc_state=self.apc_controller.calibrate(
                {stable:adapted},self.cfg.DATASET.NUM_INC_CLS,self.cfg.DATASET.NUM_INC_SHOT)
        if "block_input_anchors" not in states[0]:
            raw,counts=self._class_input_bank(features[stable],states[0]["class_ids"])
            states[0]["block_input_anchors"]={str(stable):raw}
            states[0]["block_input_counts"]={str(stable):counts}

    def train_incremental_task(self,task_id,train_loader,extract_loader,class_ids,class_names,
                               calibrate_visual,old_state):
        if self.apc_state is None: raise RuntimeError("APC must be calibrated on base session.")
        stable=int(self.apc_state["stable_block"])
        adapter=self._adapter_by_block(stable)
        self.model.eval()
        block_features=self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        current=dict(block_features[stable])
        # The earliest adapter's input is invariant across sessions. Transform
        # both current and historical inputs through the exact existing expert
        # pool before calculating residual prototype confusion.
        current["cls_in"]=self._adapt_cls_anchors(adapter,current["cls_in"]).cpu()
        old_input=torch.as_tensor(
            old_state["block_input_anchors"][str(stable)],
            device=self.device,dtype=torch.float32)
        old_ids,old_counts,old_labels=self._cpu_class_metadata(
            old_state["class_ids"],old_state["block_input_counts"][str(stable)])
        if len(old_input)!=int(old_counts.sum()):
            raise ValueError("Historical stable-block feature memory is inconsistent.")
        old_transformed=self._adapt_cls_anchors(adapter,old_input)
        old_proto=AdaptationAwarePrototypeController.prototypes(
            {"cls_in":old_transformed.detach().cpu(),"labels":old_labels,"source_ids":torch.arange(len(old_labels),dtype=torch.long)},
            old_ids)
        decisions=self.apc_controller.decide(
            {stable:current},{str(stable):old_proto.cpu()},old_ids,class_ids,self.apc_state)
        for _,a in self._moe_adapters(): a.freeze_all()
        if not decisions[0]["expand"]:
            return {"trained":False,"decisions":decisions,"expanded":[],
                    "epochs":[],"encoder_version":task_id-1}

        expert_id=self.model_without_dp.add_expert(stable,task_id)
        adapter=self._adapter_by_block(stable)
        cfg=self.cfg.TRAINER.BiMC
        text,description=self._text_prototypes(class_names,class_ids)
        old_fused=self.model_without_dp.fuse_prototypes(old_state).to(self.device).float()
        # newest_contribution expects the adapter input, not its transformed output.
        old_anchor=old_input
        groups=[
            {"params":list(adapter.experts[-1].parameters()),"lr":float(cfg.OPTIM.LR_INCREMENTAL_EXPERT)},
            {"params":list(adapter.router.columns[-1].parameters()),"lr":float(cfg.OPTIM.LR_INCREMENTAL_ROUTER)}
        ]
        params=adapter.newest_parameters()
        optimizer=torch.optim.AdamW(groups,weight_decay=float(cfg.OPTIM.WEIGHT_DECAY),
                                    betas=tuple(cfg.OPTIM.BETAS))
        epochs=int(cfg.INCREMENTAL.EPOCHS)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs))
        align_w=float(cfg.PROTECTION.ALIGN_WEIGHT)
        gate_w=float(cfg.PROTECTION.GATE_WEIGHT)
        residual_w=float(cfg.PROTECTION.RESIDUAL_WEIGHT)
        history=[]
        try:
            for epoch in range(epochs):
                bank=self._support_bank(extract_loader,class_ids,text,description,task_id,
                                        calibrate=bool(calibrate_visual))
                self.model.train()
                totals=torch.zeros(5,device=self.device); count=0
                for batch in tqdm(train_loader,desc=f"APC Session {task_id} Epoch {epoch+1}/{epochs}"):
                    images,labels=self.parse_batch(batch)
                    source=self._batch_source_ids(batch,len(labels),labels.device)
                    cached=torch.as_tensor(bank["source_ids"],device=self.device).reshape(-1)
                    match=source[:,None].eq(cached[None,:])
                    if not match.any(1).all(): raise KeyError("Missing source in LOO support bank.")
                    rows=match.long().argmax(1)
                    novel=bank["fused_loo"][rows].to(self.device).float()
                    prototypes=torch.cat([old_fused[None].expand(len(labels),-1,-1),novel],1)
                    ids=torch.cat([old_ids.to(self.device),
                                   torch.as_tensor(bank["classes"],device=self.device).long()])
                    feature,_=self.model_without_dp.extract_img_feature_train(images,max_session=task_id)
                    feature=F.normalize(feature.float(),dim=-1)
                    logits=float(cfg.INCREMENTAL.LOGIT_SCALE)*torch.einsum("bd,bcd->bc",feature,prototypes)
                    target=self._targets_to_columns(labels,{"class_ids":ids})
                    ce=F.cross_entropy(logits,target)
                    new_col=self._targets_to_columns(labels,{"class_ids":torch.as_tensor(
                        bank["classes"],device=self.device).long()})
                    text_proto=torch.as_tensor(bank["text_proto"],device=self.device).float()
                    align=1.0-F.cosine_similarity(feature,text_proto[new_col],dim=-1).mean()
                    gate,residual=adapter.newest_contribution(old_anchor)
                    old_gate=gate.square().mean()
                    old_residual=residual.square().sum(-1).mean()
                    loss=ce+align_w*align+gate_w*old_gate+residual_w*old_residual
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite APC loss.")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,1.0); optimizer.step()
                    n=len(labels); totals+=torch.stack([loss,ce,align,old_gate,old_residual]).detach()*n
                    count+=n
                if count==0: raise RuntimeError("Empty incremental training loader.")
                scheduler.step()
                values=(totals/count).cpu().tolist()
                history.append({"epoch":epoch+1,"loss":values[0],"ce":values[1],
                                "align":values[2],"old_gate":values[3],"old_residual":values[4]})
        except Exception:
            adapter.remove_newest_expert()
            for _,a in self._moe_adapters(): a.freeze_all()
            raise
        for _,a in self._moe_adapters(): a.freeze_all()
        self.model.eval()
        return {"trained":True,"decisions":decisions,
                "expanded":[{"block_idx":stable,"expert_id":expert_id}],
                "epochs":history,"encoder_version":task_id}

    def run(self):
        configured=int(_cfg_value(self.cfg.TRAINER.BiMC,"END_SESSION",-1))
        end=self.data_manager.num_tasks-1 if configured<0 else min(configured,self.data_manager.num_tasks-1)
        states=list(self._resume_state_dict_list); start=self._resume_completed_session+1
        if start>end: return
        if start>0: self._ensure_apc_ready(states)
        for task in range(start,end+1):
            self.model.eval()
            classes=self.data_manager.class_index_in_task[task]
            names=np.asarray(self.data_manager.class_names)[classes].tolist()
            extract=self.data_manager.get_dataloader(task,"train","test",False)
            train=self.data_manager.get_dataloader(task,"train","train",False)
            adaptation=None
            if task==0:
                state=self.model_without_dp.build_task_statistics(names,extract,classes,calibrate_novel_vision_proto=False)
                state["class_ids"]=torch.as_tensor(classes,device=self.device,dtype=torch.long)
                if self._moe_adapters():
                    self.train_base_task(state,train,extract); self.model.eval()
                    state=self.model_without_dp.build_task_statistics(names,extract,classes,calibrate_novel_vision_proto=False)
                    state["class_ids"]=torch.as_tensor(classes,device=self.device,dtype=torch.long)
                    features=self.model_without_dp.collect_blockwise_cls_features(extract)
                    stable=min(features); item=features[stable]
                    raw,counts=self._class_input_bank(item,classes)
                    state["block_input_anchors"]={str(stable):raw}
                    state["block_input_counts"]={str(stable):counts}
                    adapter=self._adapter_by_block(stable)
                    adapted=dict(item); adapted["cls_in"]=self._adapt_cls_anchors(adapter,item["cls_in"]).cpu()
                    self.apc_state=self.apc_controller.calibrate(
                        {stable:adapted},self.cfg.DATASET.NUM_INC_CLS,self.cfg.DATASET.NUM_INC_SHOT)
                state["creation_session"]=0; states.append(state)
            else:
                old=self.merge_dicts(states)
                if self._moe_adapters():
                    adaptation=self.train_incremental_task(task,train,extract,classes,names,
                        self.cfg.TRAINER.BiMC.VISION_CALIBRATION,old)
                self.model.eval()
                state=self.model_without_dp.build_task_statistics(
                    names,extract,classes,calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,
                    encoder_version=task if adaptation and adaptation["trained"] else task-1)
                state["class_ids"]=torch.as_tensor(classes,device=self.device,dtype=torch.long)
                if self._moe_adapters():
                    features=self.model_without_dp.collect_blockwise_cls_features(extract)
                    stable=int(self.apc_state["stable_block"])
                    raw,counts=self._class_input_bank(features[stable],classes)
                    state["block_input_anchors"]={str(stable):raw}
                    state["block_input_counts"]={str(stable):counts}
                state["creation_session"]=task; states.append(state)
            self.task_gate_biases=[0.0]*(task+1)
            merged=self.merge_dicts(states)
            start_time=time.perf_counter()
            result=self.inference_task_bilevel(task,merged)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed=time.perf_counter()-start_time
            result["AA"]=float(result["macro_task_acc"])
            result["PD"]=self._average_forgetting(task,result["task_acc"])
            self.acc_list.append(float(result["mean_acc"]))
            self.task_acc_list.append(list(result["task_acc"]))
            self.session_diagnostics.append({"session":task,"adaptation":self._cpu_copy(adaptation),
                "expert_counts":{int(i):int(a.num_experts) for i,a in self._moe_adapters()},
                "classifier":self._cpu_copy(result),"evaluation_seconds":elapsed})
            self._resume_state_dict_list=list(states); self._resume_completed_session=task
            self.save_checkpoint(task,states)
            print(f"=> Session {task}: Acc={result['mean_acc']:.3f} AA={result['AA']:.3f} PD={result['PD']:.3f}")
        print("Final accuracy:",self.acc_list)

    def _average_forgetting(self,task,current):
        if task<=0:return 0.0
        drops=[]
        for old in range(task):
            best=max(float(self.task_acc_list[s][old]) for s in range(old,task) if old<len(self.task_acc_list[s]))
            drops.append(max(0.0,best-float(current[old])))
        return float(np.mean(drops)) if drops else 0.0

    @torch.no_grad()
    def inference_task_bilevel(self,task,state):
        beta=float(self.cfg.DATASET.BETA)
        lam=float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        ids=torch.as_tensor(state["class_ids"],device=self.device,dtype=torch.long)
        image=F.normalize(torch.as_tensor(state["image_proto"],device=self.device).float(),dim=-1)
        text=F.normalize((1-lam)*torch.as_tensor(state["text_features"],device=self.device).float()
                         +lam*torch.as_tensor(state["description_proto"],device=self.device).float(),dim=-1)
        fused=F.normalize(beta*text+(1-beta)*image,dim=-1)
        if state.get("fused_proto") is not None:
            fused=F.normalize(torch.as_tensor(state["fused_proto"],device=self.device).float(),dim=-1)
        loader=self.data_manager.get_dataloader(task,"test","test")
        logits=[]; labels=[]; self.model.eval()
        for batch in tqdm(loader,desc=f"Global Eval {task}"):
            images,y=self.parse_batch(batch)
            z=F.normalize(self.model_without_dp.extract_img_feature(images).float(),dim=-1)
            logits.append(100.0*z@fused.t()); labels.append(y)
        logits=torch.cat(logits); labels=torch.cat(labels)
        result=self.evaluator.calc_accuracy(logits,labels,task,class_ids=ids)
        result["error_flow"]=self.evaluator.error_flow(logits,labels,task,class_ids=ids)
        return result
