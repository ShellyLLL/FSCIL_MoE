import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.clip.clip as clip


def build_loo_cache_from_embeddings(features, labels, source_ids, class_index=None):
    """Build query-exclusive class means from encoded support examples."""
    feature = F.normalize(torch.as_tensor(features).detach().float(), dim=-1)
    target = torch.as_tensor(labels, device=feature.device, dtype=torch.long).reshape(-1)
    source = torch.as_tensor(source_ids, device=feature.device).reshape(-1)
    if feature.ndim != 2 or feature.size(0) != target.numel() or source.numel() != target.numel():
        raise ValueError("features, labels and source_ids must align for visual LOO.")
    classes = (
        torch.unique(target, sorted=True)
        if class_index is None
        else torch.as_tensor(class_index, device=feature.device, dtype=torch.long).reshape(-1)
    )
    if classes.numel() == 0 or classes.unique().numel() != classes.numel():
        raise ValueError("LOO class ids must be non-empty and unique.")
    if not torch.isin(target, classes).all():
        raise ValueError("Support labels contain classes outside the requested LOO bank.")

    rows_per_query = []
    loo_valid = []
    for index, source_id in enumerate(source):
        rows = []
        valid = True
        for class_id in classes:
            mask = target.eq(class_id) & source.ne(source_id)
            if mask.any():
                rows.append(F.normalize(feature[mask].mean(dim=0), dim=-1))
            else:
                rows.append(feature.new_zeros(feature.size(1)))
                if target[index].eq(class_id):
                    valid = False
        rows_per_query.append(torch.stack(rows))
        loo_valid.append(valid)
    return {
        "features": feature,
        "labels": target,
        "source_ids": source,
        "classes": classes,
        "loo_visual_prototypes": torch.stack(rows_per_query),
        "loo_valid": torch.as_tensor(loo_valid, device=feature.device, dtype=torch.bool),
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
    return clip.build_model(state_dict or model.state_dict(), cfg=cfg)


class BiMC(nn.Module):
    """Frozen CLIP with one shared, incrementally expandable visual MoE encoder."""

    def __init__(self, cfg, template, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.template = template
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.BiMC.PREC in ("fp32", "amp"):
            clip_model.float()
        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.base_vision_prototype = None
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        moe_cfg = getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None)
        if moe_cfg is not None and bool(getattr(moe_cfg, "ENABLE", False)):
            for name, parameter in self.clip_model.visual.named_parameters():
                if "moe_adapter" in name:
                    parameter.requires_grad = True
        trainable = [
            name for name, parameter in self.clip_model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_numel = sum(
            parameter.numel() for parameter in self.clip_model.parameters()
            if parameter.requires_grad
        )
        total_numel = sum(parameter.numel() for parameter in self.clip_model.parameters())
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
            moe_cfg = getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None)
            if moe_cfg is not None and bool(getattr(moe_cfg, "ENABLE", False)):
                for block in self.clip_model.visual.transformer.resblocks:
                    adapter = getattr(block, "moe_adapter", None)
                    if adapter is not None:
                        adapter.train()
        return self

    @torch.no_grad()
    def collect_blockwise_cls_features(self, loader):
        """Collect each inserted adapter's actual input features and source labels."""
        block_features = {}
        fallback_source_id = 0
        moe_blocks = [
            index for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
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
                        raise RuntimeError(f"Block {block_idx} has invalid CLS input features.")
                    item = block_features.setdefault(block_idx, {
                        "cls_in": [], "labels": [], "source_ids": []
                    })
                    item["cls_in"].append(cls_in.detach().float().cpu())
                    item["labels"].append(labels.detach().cpu())
                    item["source_ids"].append(source_ids.detach().cpu())
        finally:
            self.train(was_training)
        for item in block_features.values():
            for key in ("cls_in", "labels", "source_ids"):
                item[key] = torch.cat(item[key], dim=0)
        return block_features

    @torch.no_grad()
    def build_layer_statistics(self, loader, class_index):
        """Store class-local layer-input means and diagonal variances, never old images."""
        features = self.collect_blockwise_cls_features(loader)
        class_ids = torch.as_tensor(class_index, dtype=torch.long).reshape(-1)
        statistics = {}
        for block_idx, item in features.items():
            values = item["cls_in"].float()
            labels = item["labels"].long()
            means, variances, counts = [], [], []
            for class_id in class_ids:
                selected = values[labels.eq(class_id)]
                if selected.numel() == 0:
                    raise ValueError(
                        f"Layer {block_idx} has no support features for class {int(class_id)}."
                    )
                means.append(selected.mean(dim=0))
                variances.append(selected.var(dim=0, unbiased=False).clamp_min(0.0))
                counts.append(selected.size(0))
            statistics[str(block_idx)] = {
                "class_ids": class_ids.clone(),
                "mean": torch.stack(means).cpu(),
                "variance": torch.stack(variances).cpu(),
                "count": torch.as_tensor(counts, dtype=torch.long),
            }
        return statistics

    @torch.no_grad()
    def build_leave_one_out_prototypes(self, loader, class_index=None):
        """Collect final visual features and support-source-exclusive class means."""
        features, labels, source_ids = [], [], []
        next_id = 0
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                images, target = self.parse_batch(batch)
                raw = (
                    batch.get("source_ids", batch.get("source_id", batch.get("index")))
                    if isinstance(batch, dict) else None
                )
                sid = (
                    torch.arange(next_id, next_id + target.numel())
                    if raw is None else torch.as_tensor(raw).reshape(-1)
                )
                if sid.numel() != target.numel():
                    raise ValueError("Support source ids must align with labels.")
                next_id += target.numel()
                encoded = F.normalize(self.clip_model.encode_image(images), dim=-1)
                features.append(encoded.detach().float().cpu())
                labels.append(target.detach().cpu())
                source_ids.append(sid.detach().cpu())
        finally:
            self.train(was_training)
        return build_loo_cache_from_embeddings(
            torch.cat(features), torch.cat(labels).long(), torch.cat(source_ids), class_index
        )

    @staticmethod
    def _state_class_ids(state, device=None):
        if not state:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(
            state.get("class_ids", state.get("class_index", [])),
            dtype=torch.long, device=device,
        ).reshape(-1)

    def fuse_prototypes(self, state):
        if state is None:
            return None
        if state.get("fused_proto") is not None:
            return F.normalize(torch.as_tensor(state["fused_proto"], device=self.device), dim=-1)
        image = torch.as_tensor(state["image_proto"], device=self.device)
        text = torch.as_tensor(state["text_features"], device=self.device)
        description = torch.as_tensor(state.get("description_proto", text), device=self.device)
        lambda_t = (
            self.cfg.TRAINER.BiMC.LAMBDA_T
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        )
        calibrated_text = F.normalize(
            (1.0 - lambda_t) * text + lambda_t * description, dim=-1
        )
        beta = float(self.cfg.DATASET.BETA)
        return F.normalize(beta * calibrated_text + (1.0 - beta) * image, dim=-1)

    def expandable_blocks(self):
        return [
            index for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
            if getattr(block, "moe_adapter", None) is not None
        ]

    def add_expert(self, block_idx, session_id):
        adapter = self.clip_model.visual.transformer.resblocks[int(block_idx)].moe_adapter
        reduction = int(getattr(
            self.cfg.TRAINER.BiMC.VISUAL_MOE, "INCREMENTAL_REDUCTION", 48
        ))
        return adapter.add_expert(session_id=session_id, reduction=reduction)

    def export_moe_topology(self):
        return {
            str(index): block.moe_adapter.export_topology()
            for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
            if getattr(block, "moe_adapter", None) is not None
        }

    def rebuild_moe_topology(self, topology):
        topology = topology or {}
        for index, block in enumerate(self.clip_model.visual.transformer.resblocks):
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                adapter.rebuild_topology(topology.get(str(index), topology.get(index)))

    @torch.no_grad()
    def extract_teacher_features(self, images):
        return self.clip_model.encode_image(
            images.to(self.device), disable_moe=True
        )

    def extract_img_feature_train(self, images):
        return self.clip_model.encode_image(
            images.to(self.device), return_moe_aux=True
        )

    def forward_train(self, images, task_stat, beta, compute_teacher=False):
        image_features, moe_aux = self.extract_img_feature_train(images)
        teacher_features = self.extract_teacher_features(images) if compute_teacher else None
        image_features_norm = F.normalize(image_features.float(), dim=-1)
        fused_proto = F.normalize(task_stat["fused_proto"].to(self.device).float(), dim=-1)
        logits = self.clip_model.logit_scale.exp() * image_features_norm @ fused_proto.t()
        return logits, image_features, teacher_features, moe_aux

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        clip_weights, all_targets = [], []
        class_id = int(cls_begin_index)
        for classname in class_names:
            all_targets.append(torch.full((len(template),), class_id, device=self.device))
            class_id += 1
            classname = classname.replace("_", " ").replace("-", " ")
            texts = [prompt.format(classname) for prompt in template]
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            clip_weights.append(F.normalize(embeddings.mean(dim=0), dim=-1))
        return F.normalize(torch.stack(clip_weights), dim=-1), torch.cat(all_targets)

    @torch.no_grad()
    def inference_all_img_feature(self, loader, class_index):
        all_features, all_labels = [], []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = F.normalize(self.clip_model.encode_image(images).float(), dim=-1)
            all_features.append(features)
            all_labels.append(labels)
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        class_ids = torch.as_tensor(class_index, device=labels.device, dtype=torch.long)
        prototypes = []
        for class_id in class_ids:
            selected = features[labels.eq(class_id)]
            if selected.numel() == 0:
                raise ValueError(f"No support images for class {int(class_id)}.")
            prototypes.append(F.normalize(selected.mean(dim=0), dim=-1))
        return features, labels, torch.stack(prototypes)

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        with open(gpt_path, "r", encoding="utf-8") as file:
            prompts_by_class = {
                key.lower().replace("_", " "): value for key, value in json.load(file).items()
            }
        description_embeddings, mean_embeddings, all_targets = [], [], []
        class_id = int(cls_begin_index)
        for class_name in class_names:
            prompts = prompts_by_class[class_name.lower().replace("_", " ")]
            all_targets.append(torch.full((len(prompts),), class_id, device=self.device))
            class_id += 1
            tokens = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.device)
            embeddings = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            description_embeddings.append(embeddings)
            mean_embeddings.append(F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1))
        return (
            torch.cat(description_embeddings, dim=0),
            torch.cat(all_targets, dim=0),
            torch.cat(mean_embeddings, dim=0),
        )

    def soft_calibration(self, base_prototypes, current_prototypes):
        """BiMC visual intra-modal calibration for any leading prototype dimensions."""
        shift_weight = float(self.cfg.TRAINER.BiMC.LAMBDA_I)
        temperature = float(self.cfg.TRAINER.BiMC.TAU)
        base = F.normalize(torch.as_tensor(base_prototypes, device=self.device).float(), dim=-1)
        current = F.normalize(torch.as_tensor(current_prototypes, device=self.device).float(), dim=-1)
        shape = current.shape
        flat = current.reshape(-1, current.size(-1))
        weights = torch.softmax(flat @ base.t() * temperature, dim=-1)
        inherited = F.normalize(weights @ base, dim=-1)
        calibrated = F.normalize(
            (1.0 - shift_weight) * flat + shift_weight * inherited, dim=-1
        )
        return calibrated.reshape(shape)

    @torch.no_grad()
    def build_task_statistics(self, class_names, loader, class_index,
                              calibrate_novel_vision_proto=False):
        class_ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long).reshape(-1)
        if class_ids.numel() != len(class_names):
            raise ValueError("Class names and global class ids must have the same length.")
        text_features, _ = self.inference_text_feature(
            class_names, self.template, int(class_ids[0].item())
        )
        _, _, description_proto = self.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, int(class_ids[0].item())
        )
        image_features, labels, raw_image_proto = self.inference_all_img_feature(
            loader, class_ids
        )
        if calibrate_novel_vision_proto:
            if self.base_vision_prototype is None:
                raise RuntimeError("Base visual prototypes must be built before visual calibration.")
            image_proto = self.soft_calibration(self.base_vision_prototype, raw_image_proto)
        else:
            image_proto = raw_image_proto
            if int(class_ids.min().item()) == 0:
                self.base_vision_prototype = raw_image_proto.detach().clone()

        lambda_t = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_T)
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        )
        calibrated_text = F.normalize(
            (1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1
        )
        beta = float(self.cfg.DATASET.BETA)
        fused_proto = F.normalize(
            beta * calibrated_text + (1.0 - beta) * image_proto, dim=-1
        )
        layer_stats = self.build_layer_statistics(loader, class_ids.detach().cpu())
        return {
            "class_ids": class_ids,
            "class_index": class_ids,
            "text_features": text_features.detach(),
            "description_proto": description_proto.detach(),
            "raw_image_proto": raw_image_proto.detach(),
            "image_proto": image_proto.detach(),
            "fused_proto": fused_proto.detach(),
            "layer_stats": layer_stats,
            "sample_count": int(image_features.size(0)),
        }

    @torch.no_grad()
    def forward_ours(self, images, image_proto, description_proto, text_features,
                     beta, fused_proto=None):
        image_features = F.normalize(self.extract_img_feature(images).float(), dim=-1)
        if fused_proto is None:
            lambda_t = (
                self.cfg.TRAINER.BiMC.LAMBDA_T
                if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
            )
            calibrated_text = F.normalize(
                (1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1
            )
            fused_proto = F.normalize(
                beta * calibrated_text + (1.0 - beta) * image_proto, dim=-1
            )
        else:
            fused_proto = F.normalize(fused_proto.to(self.device).float(), dim=-1)
        return 100.0 * image_features @ fused_proto.t()

    @torch.no_grad()
    def extract_img_feature(self, images):
        return self.clip_model.encode_image(images.to(self.device))

    def parse_batch(self, batch):
        return (
            batch["image"].to(self.device),
            batch["label"].to(self.device, dtype=torch.long),
        )
