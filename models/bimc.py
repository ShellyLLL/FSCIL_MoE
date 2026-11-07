"""Unified FSCIL model: one current visual encoder and one all-seen BiMC prototype classifier."""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.clip.clip as clip


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
        super().__init__()
        self.cfg, self.device, self.template = cfg, device, template
        bimc_cfg = cfg.TRAINER.BiMC
        if not bool(getattr(bimc_cfg, "FREEZE_TEXT_ENCODER", True)):
            raise ValueError("Unified FSCIL architecture requires FREEZE_TEXT_ENCODER=True.")
        if not bool(getattr(bimc_cfg, "FREEZE_VISUAL_BACKBONE", True)):
            raise ValueError("Unified FSCIL architecture requires FREEZE_VISUAL_BACKBONE=True.")
        self.clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.BiMC.PREC in ("fp32", "amp"):
            self.clip_model.float()
        self.clip_model.to(device).eval()
        self.base_vision_prototype = None
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for block in self.clip_model.visual.transformer.resblocks:
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                for p in adapter.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        for block in self.clip_model.visual.transformer.resblocks:
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                adapter.train(mode and any(p.requires_grad for p in adapter.parameters()))
        return self

    def parse_batch(self, batch):
        return batch["image"].to(self.device), batch["label"].to(self.device, dtype=torch.long)

    def adapters(self):
        return [(i, b.moe_adapter) for i, b in enumerate(
            self.clip_model.visual.transformer.resblocks
        ) if getattr(b, "moe_adapter", None) is not None]

    def extract_img_feature_train(self, images):
        return self.clip_model.encode_image(images.to(self.device), return_moe_aux=True)

    @torch.no_grad()
    def extract_img_feature(self, images):
        return self.clip_model.encode_image(images.to(self.device))

    @torch.no_grad()
    def collect_blockwise_cls_features(self, loader):
        collected = {}
        self.eval()
        for batch in loader:
            images, labels = self.parse_batch(batch)
            source = batch.get("source_ids", batch.get("source_id", batch.get("index")))
            if source is None:
                raise KeyError("APC requires stable source IDs.")
            source = torch.as_tensor(source).long().reshape(-1)
            _, aux_list = self.extract_img_feature_train(images)
            if len(aux_list) != len(self.adapters()):
                raise RuntimeError("MoE block auxiliary output count mismatch.")
            for (block, _), aux in zip(self.adapters(), aux_list):
                item = collected.setdefault(block, {
                    "raw_reference_cls": [], "adapter_input_cls": [],
                    "block_out": [], "labels": [], "source_ids": []
                })
                item["raw_reference_cls"].append(aux["raw_reference_cls"].detach().cpu())
                item["adapter_input_cls"].append(aux["adapter_input_cls"].detach().cpu())
                item["block_out"].append(aux["block_out"].detach().cpu())
                item["labels"].append(labels.detach().cpu())
                item["source_ids"].append(source.detach().cpu())
        for item in collected.values():
            for key in item:
                item[key] = torch.cat(item[key], dim=0)
        return collected

    @staticmethod
    def mean_normalized_features(features, dim=0):
        """Mean of per-sample L2-normalized features (BiMC Eq. 2/description center).

        Deliberately do not normalize the resultant mean: its magnitude carries
        information about within-class/description dispersion and participates
        in the subsequent linear modality calibration.
        """
        values = torch.as_tensor(features)
        if values.ndim < 2 or values.numel() == 0:
            raise ValueError("Prototype features must be a non-empty tensor with a feature dimension.")
        if not torch.isfinite(values).all():
            raise FloatingPointError("Prototype features contain non-finite values.")
        result = F.normalize(values.float(), dim=-1).mean(dim=dim)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Prototype mean contains non-finite values.")
        return result

    def calibrate_text(self, text_features, description_center):
        lam = float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        return (1.0 - lam) * text_features.float() + lam * description_center.float()

    def fuse_modalities(self, visual_prototype, text_features, description_center):
        """BiMC inter-modal calibration; cosine normalization belongs to scoring."""
        beta = float(self.cfg.DATASET.BETA)
        calibrated_text = self.calibrate_text(text_features, description_center)
        return beta * calibrated_text + (1.0 - beta) * visual_prototype.float()

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index=0):
        result = []
        for name in class_names:
            clean = str(name).replace("_", " ").replace("-", " ")
            tokens = clip.tokenize([t.format(clean) for t in template]).to(self.device)
            features = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            result.append(F.normalize(features.mean(0), dim=-1))
        return torch.stack(result)

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index=0):
        with open(gpt_path, "r", encoding="utf-8") as f:
            descriptions = {k.lower().replace("_", " "): v for k, v in json.load(f).items()}
        all_desc, centers = [], []
        for name in class_names:
            key = str(name).lower().replace("_", " ")
            if key not in descriptions:
                raise KeyError(f"Missing text descriptions for class {name!r}.")
            tokens = clip.tokenize(descriptions[key]).to(self.device)
            features = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            all_desc.append(features)
            centers.append(self.mean_normalized_features(features, dim=0))
        return all_desc, torch.stack(centers)

    def soft_calibration(self, base_protos, cur_protos):
        # The patent's visual prototype is already normalized before calibration.
        # Keep the weighted base-prototype resultant unnormalized: its magnitude
        # is part of the visual/text balance in the subsequent inter-modal fusion.
        base = torch.as_tensor(base_protos, device=self.device, dtype=torch.float32)
        current = torch.as_tensor(cur_protos, device=self.device, dtype=torch.float32)
        base_for_similarity = F.normalize(base, dim=-1)
        current_for_similarity = F.normalize(current, dim=-1)
        tau = float(self.cfg.TRAINER.BiMC.TAU)
        weights = torch.softmax(tau * current_for_similarity @ base_for_similarity.t(), dim=-1)
        inherited = weights @ base
        lam = float(self.cfg.TRAINER.BiMC.LAMBDA_I)
        return (1.0 - lam) * current + lam * inherited

    @torch.no_grad()
    def set_base_visual_prototype(self, prototypes):
        """Install the immutable base-session visual calibration anchor."""
        value = torch.as_tensor(
            prototypes, device=self.device, dtype=torch.float32
        ).detach()
        if value.ndim != 2 or value.shape[0] == 0:
            raise ValueError("Base visual prototypes must be a non-empty [classes, dim] tensor.")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Base visual prototypes contain non-finite values.")
        norms = value.norm(dim=-1)
        if torch.any(norms <= 1e-12):
            raise ValueError("Base visual prototypes contain zero-norm rows.")
        # Keep the mean-of-normalized-samples resultant unchanged. Its norm
        # must not be silently reset before visual/text inter-modal fusion.
        value = value.detach().clone()
        if self.base_vision_prototype is not None:
            existing = self.base_vision_prototype.to(device=value.device, dtype=value.dtype)
            if existing.shape != value.shape or not torch.allclose(
                existing, value, rtol=0.0, atol=1e-6
            ):
                raise RuntimeError(
                    "Base visual calibration anchor is immutable once initialized."
                )
        self.base_vision_prototype = value

    def calibrate_incremental_visual(self, current, reference=None):
        if self.base_vision_prototype is None:
            raise RuntimeError(
                "Base visual calibration anchor is missing. Initialize it from "
                "Session 0 image_proto or restore it from the checkpoint before "
                "incremental visual calibration."
            )
        current = torch.as_tensor(current, device=self.device, dtype=torch.float32)
        if current.ndim != 2 or current.shape[-1] != self.base_vision_prototype.shape[-1]:
            raise ValueError(
                "Incremental visual prototypes must be [novel_classes, base_embedding_dim]."
            )
        return self.soft_calibration(self.base_vision_prototype, current)

    @torch.no_grad()
    def build_task_statistics(self, class_names, loader, class_index,
                              calibrate_novel_vision_proto=False):
        self.eval()
        features, labels, source_ids = [], [], []
        for batch in loader:
            images, y = self.parse_batch(batch)
            sid = batch.get("source_ids", batch.get("source_id", batch.get("index")))
            if sid is None:
                raise KeyError("Support prototypes require stable source IDs.")
            features.append(F.normalize(self.extract_img_feature(images).float(), dim=-1))
            labels.append(y)
            source_ids.append(torch.as_tensor(sid, device=self.device).reshape(-1))
        features, labels, source_ids = torch.cat(features), torch.cat(labels), torch.cat(source_ids)
        ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        image_proto = []
        for cls in ids:
            selected = features[labels.eq(cls)]
            if selected.numel() == 0:
                raise ValueError(f"Missing support examples for global class {int(cls)}.")
            image_proto.append(self.mean_normalized_features(selected, dim=0))
        image_proto = torch.stack(image_proto)
        if int(ids.min().item()) > 0 and calibrate_novel_vision_proto:
            image_proto = self.calibrate_incremental_visual(image_proto)
        elif int(ids.min().item()) == 0:
            self.set_base_visual_prototype(image_proto)

        text = self.inference_text_feature(class_names, self.template)
        desc_list, desc_center = self.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH
        )
        fused = self.fuse_modalities(image_proto, text, desc_center)
        return {
            "class_ids": ids.detach().clone(),
            "class_index": ids.detach().clone(),
            "image_proto": image_proto.detach().clone(),
            "text_features": text.detach().clone(),
            "description_proto": desc_center.detach().clone(),
            "description_features": [v.detach().clone() for v in desc_list],
            "fused_proto": fused.detach().clone(),
            "support_features": features.detach().clone(),
            "support_labels": labels.detach().clone(),
            "support_source_ids": source_ids.detach().clone(),
        }

    def fuse_prototypes(self, state):
        return F.normalize(torch.as_tensor(state["fused_proto"], device=self.device).float(), dim=-1)

    def classification_scale(self):
        """Frozen CLIP logit scale used for stable prototype cross-entropy training."""
        return self.clip_model.logit_scale.detach().float().exp().clamp(max=100.0)

    def forward_train(self, images, prototypes):
        feature, aux = self.extract_img_feature_train(images)
        logits = self.classification_scale() * (
            F.normalize(feature.float(), dim=-1)
            @ F.normalize(prototypes.float(), dim=-1).t()
        )
        return logits, feature, aux

    def export_moe_topology(self):
        return {str(i): adapter.export_topology() for i, adapter in self.adapters()}

    def rebuild_moe_topology(self, topology):
        topology = topology or {}
        for i, adapter in self.adapters():
            adapter.rebuild_topology(topology.get(str(i), topology.get(i)))

    def add_expert(self, block_idx, session_id):
        return self.clip_model.visual.transformer.resblocks[int(block_idx)].moe_adapter.add_expert(session_id)

    def expandable_blocks(self):
        return [i for i, _ in self.adapters()]

