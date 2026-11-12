
import json
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.clip.clip as clip


@dataclass
class LayerExpansionDecision:
    block_idx: int
    expand: bool
    class_scores: Dict[int, float]
    uncovered_classes: List[int]


def build_loo_cache_from_embeddings(features, labels, source_ids, class_index=None):
    """Build source-exclusive visual prototypes from already encoded features."""
    feature = F.normalize(torch.as_tensor(features).detach(), dim=-1)
    target = torch.as_tensor(labels, device=feature.device, dtype=torch.long).reshape(-1)
    source = torch.as_tensor(source_ids, device=feature.device).reshape(-1)
    if feature.ndim != 2 or feature.size(0) != target.numel() or source.numel() != target.numel():
        raise ValueError("features, labels and source_ids must align for visual LOO.")
    if not torch.isfinite(feature).all():
        raise ValueError("visual LOO features must be finite.")
    # A source may own multiple augmented views, but it cannot identify
    # different semantic classes: otherwise query-exclusive LOO is ambiguous.
    for source_id in torch.unique(source, sorted=True):
        source_labels = torch.unique(target[source.eq(source_id)])
        if source_labels.numel() != 1:
            raise ValueError(
                f"source id {int(source_id)} is associated with multiple classes."
            )
    classes = (
        torch.unique(target, sorted=True)
        if class_index is None
        else torch.unique(torch.as_tensor(class_index, device=feature.device, dtype=torch.long), sorted=True)
    )
    rows_per_query = []
    for sid in source:
        rows = []
        for cls in classes:
            mask = target.eq(cls) & source.ne(sid)
            rows.append(feature[mask].mean(0) if mask.any() else feature.new_zeros(feature.size(1)))
        rows_per_query.append(F.normalize(torch.stack(rows), dim=-1))
    loo_valid = torch.stack([
        (target.eq(target[index]) & source.ne(source[index])).any()
        for index in range(source.numel())
    ])
    return {
        "features": feature,
        "labels": target,
        "source_ids": source,
        "classes": classes,
        "loo_visual_prototypes": torch.stack(rows_per_query),
        "loo_valid": loo_valid,
    }


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), cfg=cfg)
    return model


class BiMC(nn.Module):
    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.BiMC.PREC in ["fp32", "amp"]:
            clip_model.float()
        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None
        self.base_vision_prototype = None
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if hasattr(self.cfg.TRAINER.BiMC, 'VISUAL_MOE') and self.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
            for name, param in self.clip_model.visual.named_parameters():
                # 只训练显式插入的视觉 MoE Adapter。不要使用 "proj" 模糊匹配，
                # 否则会误解冻 attention projection、MLP projection 和 visual.proj。
                if "moe_adapter" in name:
                    # Descriptors are opened only by their reconstruction-only
                    # training stage; classification never owns their gradient.
                    param.requires_grad = ".descriptors." not in name
        trainable = [
            name for name, param in self.clip_model.named_parameters()
            if param.requires_grad
        ]
        trainable_numel = sum(
            param.numel() for param in self.clip_model.parameters() if param.requires_grad
        )
        total_numel = sum(param.numel() for param in self.clip_model.parameters())
        print(
            f"[Freeze] trainable tensors: {len(trainable)}, "
            f"parameters: {trainable_numel:,}/{total_numel:,}"
        )
        for name in trainable:
            print(f"  - {name}")

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.clip_model.eval()
            if hasattr(self.cfg.TRAINER.BiMC, 'VISUAL_MOE') and self.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
                for block in self.clip_model.visual.transformer.resblocks:
                    if hasattr(block, 'moe_adapter') and block.moe_adapter is not None:
                        block.moe_adapter.train()
                        if not any(
                            param.requires_grad
                            for param in block.moe_adapter.router.parameters()
                        ):
                            block.moe_adapter.router.eval()
        return self

    @torch.no_grad()
    def collect_blockwise_cls_features(self, loader):
        """Collect current-path CLS inputs, labels and stable support source ids."""
        block_features, fallback_source_id = {}, 0
        moe_blocks = [
            idx for idx, block in enumerate(self.clip_model.visual.transformer.resblocks)
            if getattr(block, "moe_adapter", None) is not None
        ]
        if not moe_blocks:
            return block_features
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                images, labels = self.parse_batch(batch)
                raw = (
                    batch.get("source_ids", batch.get("source_id", batch.get("index")))
                    if isinstance(batch, dict) else None
                )
                source_ids = (
                    torch.arange(fallback_source_id, fallback_source_id + labels.numel())
                    if raw is None else torch.as_tensor(raw).reshape(-1)
                )
                fallback_source_id += labels.numel()
                if source_ids.numel() != labels.numel():
                    raise ValueError("Support source ids must align with labels.")
                _, aux_list = self.clip_model.encode_image(images, return_moe_aux=True)
                if len(aux_list) != len(moe_blocks):
                    raise RuntimeError("MoE auxiliary outputs do not match inserted blocks.")
                for block_idx, aux in zip(moe_blocks, aux_list):
                    cls_in = aux.get("cls_in")
                    if cls_in is None or cls_in.ndim != 2:
                        raise RuntimeError(f"Block {block_idx} has invalid CLS features.")
                    item = block_features.setdefault(block_idx, {
                        "cls_in": [], "labels": [], "source_ids": []
                    })
                    item["cls_in"].append(cls_in.detach().cpu())
                    item["labels"].append(labels.detach().cpu())
                    item["source_ids"].append(source_ids.detach().cpu())
        finally:
            self.train(was_training)
        for item in block_features.values():
            item["cls_in"] = torch.cat(item["cls_in"])
            item["labels"] = torch.cat(item["labels"])
            item["source_ids"] = torch.cat(item["source_ids"])
        return block_features

    @torch.no_grad()
    def build_leave_one_out_prototypes(self, loader, class_index=None):
        """Collect a support cache with stable source ids and per-query LOO means.

        ``loo_visual_prototypes[i, j]`` is class ``j``'s support mean after
        removing all views having query ``i``'s source id.  This prevents a
        query from winning merely because its own augmented view is in support.
        """
        features, reference_features, labels, source_ids = [], [], [], []
        next_id = 0
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                images, y = self.parse_batch(batch)
                raw = batch.get("source_ids", batch.get("source_id", batch.get("index"))) if isinstance(batch, dict) else None
                sid = torch.arange(next_id, next_id + y.numel()) if raw is None else torch.as_tensor(raw).reshape(-1)
                if sid.numel() != y.numel():
                    raise ValueError("support source ids must align with labels.")
                next_id += y.numel()
                adapted, reference = self.clip_model.encode_image(
                    images, return_reference=True
                )
                features.append(F.normalize(adapted, dim=-1).detach().cpu())
                reference_features.append(F.normalize(reference, dim=-1).detach().cpu())
                labels.append(y.detach().cpu())
                source_ids.append(sid.detach().cpu())
        finally:
            self.train(was_training)
        labels, source_ids = torch.cat(labels).long(), torch.cat(source_ids)
        adapted_cache = build_loo_cache_from_embeddings(
            torch.cat(features), labels, source_ids, class_index
        )
        reference_cache = build_loo_cache_from_embeddings(
            torch.cat(reference_features), labels, source_ids, class_index
        )
        if not torch.equal(adapted_cache["classes"], reference_cache["classes"]) or not torch.equal(
            adapted_cache["source_ids"], reference_cache["source_ids"]
        ):
            raise RuntimeError("Adapted/reference LOO support caches lost sample alignment.")
        adapted_cache["reference_features"] = torch.cat(reference_features)
        adapted_cache["loo_reference_prototypes"] = reference_cache["loo_visual_prototypes"]
        adapted_cache["reference_visual_prototypes"] = reference_cache["visual_prototypes"]
        return adapted_cache

    @staticmethod
    def _state_class_ids(state, device=None):
        if not state:
            return torch.empty(0, dtype=torch.long, device=device)
        ids = state.get("class_ids", state.get("class_index", []))
        return torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)

    def expandable_blocks(self):
        return [index for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
                if getattr(block, "moe_adapter", None) is not None]

    @torch.no_grad()
    def plan_layer_expansion(self, block_idx, features, class_index, threshold=None):
        """Expand a layer iff at least one current class is uncovered by every AE."""
        block_idx = int(block_idx)
        threshold = float(getattr(
            self.cfg.TRAINER.BiMC.VISUAL_MOE, "EXPANSION_Z_THRESHOLD", 1.0
        ) if threshold is None else threshold)
        if block_idx not in features:
            raise KeyError(f"Block {block_idx} did not produce support features.")

        adapter = self.clip_model.visual.transformer.resblocks[block_idx].moe_adapter
        adapter.assert_descriptors_ready()
        values = features[block_idx]["cls_in"].to(self.device)
        labels = features[block_idx]["labels"].to(self.device)
        scores = adapter.descriptor_scores(values)

        expected = sorted(int(value) for value in class_index)
        observed = sorted(int(value) for value in torch.unique(labels, sorted=True).tolist())
        if observed != expected:
            raise RuntimeError(f"Block {block_idx} received classes {observed}, expected {expected}.")

        class_ids = torch.unique(labels, sorted=True)
        sample_best_z = scores.min(dim=1).values
        class_min_z = torch.stack([
            sample_best_z[labels.eq(class_id)].median() for class_id in class_ids
        ])
        class_scores = {
            int(class_id): float(class_min_z[row].item())
            for row, class_id in enumerate(class_ids.tolist())
        }
        uncovered = [
            class_id for class_id, score in class_scores.items() if score > threshold
        ]
        decision = LayerExpansionDecision(block_idx, bool(uncovered), class_scores, uncovered)
        print(
            f"=> [SEMA-AE][B{block_idx}] class_min_z={class_scores} "
            f"threshold={threshold:.3f} uncovered={uncovered} "
            f"decision={'EXPAND' if decision.expand else 'REUSE'}"
        )
        return decision

    def add_expert(self, block_idx, session_id):
        adapter = self.clip_model.visual.transformer.resblocks[int(block_idx)].moe_adapter
        reduction = int(getattr(self.cfg.TRAINER.BiMC.VISUAL_MOE, "INCREMENTAL_REDUCTION", 8))
        return adapter.add_expert(session_id=session_id, reduction=reduction)

    def export_moe_topology(self):
        return {str(index): block.moe_adapter.export_topology()
                for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
                if getattr(block, "moe_adapter", None) is not None}

    def rebuild_moe_topology(self, topology):
        topology = topology or {}
        for index, block in enumerate(self.clip_model.visual.transformer.resblocks):
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                adapter.rebuild_topology(topology.get(str(index), topology.get(index)))

    @torch.no_grad()
    def extract_teacher_features(self, images):
        # The reference stream is computed alongside the adapted stream.  Do
        # not temporarily detach adapters from live Transformer blocks.
        _, reference_features = self.clip_model.encode_image(
            images.to(self.device), return_reference=True
        )
        return reference_features

    def extract_img_feature_train(self, images, return_reference=False):
        output = self.clip_model.encode_image(
            images.to(self.device),
            return_moe_aux=True,
            return_reference=return_reference,
        )
        if return_reference:
            adapted_features, reference_features, moe_aux = output
            return adapted_features, moe_aux, reference_features
        return output

    def forward_train(self, images, task_stat, compute_teacher=False):
        if compute_teacher:
            img_feat, moe_aux, teacher_feat = self.extract_img_feature_train(
                images, return_reference=True
            )
        else:
            img_feat, moe_aux = self.extract_img_feature_train(images)
            teacher_feat = None
        img_feat_norm = F.normalize(img_feat, dim=-1)
        visual_proto = F.normalize(task_stat["image_proto"].to(self.device), dim=-1)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * img_feat_norm @ visual_proto.t()
        return logits, img_feat, teacher_feat, moe_aux

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        clip_weights, all_targets = [], []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k, device=self.device)
            all_targets.append(targets)
            k += 1
            classname = classname.replace("_", " ").replace("-", " ")
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(self.device)
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings.mean(dim=0) / class_embeddings.mean(dim=0).norm())
        return F.normalize(torch.stack(clip_weights, dim=0), dim=-1), torch.cat(all_targets, dim=0)

    @torch.no_grad()
    def inference_all_img_feature(self, loader):
        all_features, all_reference_features, all_labels = [], [], []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features, reference = self.clip_model.encode_image(
                images, return_reference=True
            )
            all_features.append(F.normalize(features, dim=-1))
            all_reference_features.append(F.normalize(reference, dim=-1))
            all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_reference_features = torch.cat(all_reference_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        prototypes = [all_features[all_labels == c].mean(dim=0) for c in torch.unique(all_labels)]
        reference_prototypes = [
            all_reference_features[all_labels == c].mean(dim=0)
            for c in torch.unique(all_labels)
        ]
        return (
            all_features, all_labels,
            F.normalize(torch.stack(prototypes, dim=0), dim=-1),
            F.normalize(torch.stack(reference_prototypes, dim=0), dim=-1),
        )

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings, mean_embeddings, all_targets = [], [], []
        with open(gpt_path, "r", encoding="utf-8") as file:
            gpt_prompt_dict = {k.lower().replace("_", " "): v for k, v in json.load(file).items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = gpt_prompt_dict[single_key.lower().replace("_", " ")]
            all_targets.append(torch.full((len(single_class_prompts),), k, device=self.device))
            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts]).to(self.device)
            text_features = self.clip_model.encode_text(x_tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0, keepdim=True))
            description_embeddings.append(text_features)
        return torch.cat(description_embeddings, dim=0), torch.cat(all_targets, dim=0), F.normalize(torch.cat(mean_embeddings, dim=0), dim=-1)

    def soft_calibration(self, base_protos, cur_protos):
        shift_weight, tau = self.cfg.TRAINER.BiMC.LAMBDA_I, self.cfg.TRAINER.BiMC.TAU
        base_protos, cur_protos = F.normalize(base_protos, p=2, dim=-1), F.normalize(cur_protos, p=2, dim=-1)
        norm_weights = torch.softmax(torch.mm(cur_protos, base_protos.T) * tau, dim=1)
        delta_protos = F.normalize(torch.matmul(norm_weights, base_protos), p=2, dim=-1)
        return F.normalize((1 - shift_weight) * cur_protos + shift_weight * delta_protos, dim=-1)

    def calibrate_incremental_visual(self, current, reference):
        """Calibrate in the base space, then transport the paired support shift."""
        if self.base_vision_prototype is None:
            raise RuntimeError("Base visual prototypes must be built before an incremental session.")
        shape = current.shape
        reference = reference.reshape(-1, shape[-1])
        current = current.reshape(-1, shape[-1])
        calibrated = self.soft_calibration(self.base_vision_prototype, reference)
        return F.normalize(calibrated + current - reference, dim=-1).reshape(shape)

    @torch.no_grad()
    def build_task_statistics(self, class_names, loader, class_index,
                              calibrate_novel_vision_proto=False):
        """Build aligned text and raw visual support prototypes for one session."""
        cls_begin_index = int(class_index[0])
        text_features, _ = self.inference_text_feature(
            class_names, self.template, cls_begin_index
        )
        _, _, description_proto = self.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, cls_begin_index
        )
        images_features, anchor_labels, images_proto, reference_images_proto = self.inference_all_img_feature(loader)

        class_ids = torch.as_tensor(
            class_index, device=images_proto.device, dtype=torch.long
        ).reshape(-1)
        observed_ids = torch.unique(anchor_labels, sorted=True).to(class_ids.device)
        matches = class_ids[:, None].eq(observed_ids[None, :])
        if not matches.sum(dim=1).eq(1).all() or class_ids.numel() != images_proto.size(0):
            raise ValueError("Visual support prototypes do not align one-to-one with session class ids.")
        images_proto = images_proto[matches.long().argmax(dim=1)]

        if cls_begin_index == 0:
            self.base_vision_prototype = images_proto.detach().clone()

        source_ids = getattr(getattr(loader, "dataset", None), "source_ids", None)
        source_ids = (
            torch.arange(len(anchor_labels))
            if source_ids is None else torch.as_tensor(source_ids, dtype=torch.long).cpu()
        )
        if len(source_ids) != len(anchor_labels):
            raise ValueError("Anchor source ids must align with support features.")

        return {
            "description_proto": description_proto,
            "text_features": text_features,
            "image_proto": images_proto,
            "reference_image_proto": reference_images_proto,
            "class_index": class_index,
            "class_ids": class_ids,
            "sample_cnt": len(images_features),
            "anchor_features": images_features.detach().to(device="cpu", dtype=torch.float16),
            "anchor_labels": anchor_labels.detach().cpu(),
            "anchor_source_ids": source_ids,
            "encoder_version": 0,
        }

    @torch.no_grad()
    def forward_ours(self, images, image_proto):
        """Compatibility inference entry point; classification is visual-only."""
        img_feat = F.normalize(self.extract_img_feature(images), dim=-1)
        visual_proto = F.normalize(torch.as_tensor(image_proto, device=self.device), dim=-1)
        return 100.0 * img_feat @ visual_proto.t()

    @torch.no_grad()
    def extract_img_feature(self, images):
        return self.clip_model.encode_image(images.to(self.device))

    def parse_batch(self, batch):
        # NumPy-backed FSCIL datasets can yield int32 labels on Windows.  Keep
        # targets in PyTorch's required class-index dtype for CE and indexing.
        return batch["image"].to(self.device), batch["label"].to(self.device, dtype=torch.long)
