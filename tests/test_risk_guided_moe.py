"""Invariants for function-preserving expansion and versioned classifiers."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.clip.model import VisionTransformer, VisualMoEAdapter
from models.bimc import BiMC, build_loo_cache_from_embeddings
from engine.engine import Runner
from utils.evaluator import AccuracyEvaluator


class TinySupport(Dataset):
    def __init__(self, shots=5):
        self.labels = [2] * shots + [3] * shots
        self.source_ids = list(range(2 * shots))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = F.one_hot(torch.tensor(self.labels[index]), num_classes=4).float()
        return {"image": value, "label": self.labels[index],
                "source_id": self.source_ids[index]}


class StabilityRiskTests(unittest.TestCase):
    def test_zero_drift_for_same_version(self):
        runner = Runner.__new__(Runner)
        self.assertEqual(float(runner._representation_drift(torch.randn(2, 4), 0, 0)), 0.0)

    def test_drift_is_differentiable(self):
        class Encoder:
            def __init__(self):
                self.delta = torch.nn.Parameter(torch.tensor(0.2))

            def extract_img_feature(self, x, max_session=None):
                if max_session == 0:
                    return x.clone()
                return x + self.delta * torch.roll(x, 1, -1)

            def extract_img_feature_train(self, x, max_session=None):
                return self.extract_img_feature(x, max_session), []

        runner = Runner.__new__(Runner)
        runner.model_without_dp = Encoder()
        loss = runner._representation_drift(torch.eye(3, 4), 1, 0)
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(runner.model_without_dp.delta.grad))


class ExpansionInvariantTests(unittest.TestCase):
    def test_appending_an_expert_preserves_the_old_function(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.eval()
        inputs = torch.randn(5, 3, 8)
        before, _ = adapter(inputs)
        adapter.add_expert(session_id=1, reduction=2)
        at_creation, _ = adapter(inputs)
        torch.testing.assert_close(at_creation, before, rtol=0, atol=0)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.3)
        old, _ = adapter(inputs, max_session=0)
        current, _ = adapter(inputs, max_session=1)
        torch.testing.assert_close(old, before, rtol=0, atol=0)
        self.assertGreater((current - old).abs().max().item(), 0)
        adapter.remove_newest_expert()
        restored, _ = adapter(inputs)
        torch.testing.assert_close(restored, before, rtol=0, atol=0)

    def test_later_experts_cannot_change_an_old_visual_encoder(self):
        torch.manual_seed(11)
        visual = VisionTransformer(8, 4, 8, 2, 2, 8).eval()
        visual.transformer.resblocks[-1].moe_adapter = VisualMoEAdapter(
            8, num_experts=2, reduction=2)
        images = torch.randn(2, 3, 8, 8)
        before = visual(images, max_session=0)
        adapter = visual.transformer.resblocks[-1].moe_adapter
        adapter.add_expert(1, reduction=2)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.4)
        adapter.freeze_all()
        torch.testing.assert_close(visual(images, max_session=0), before, rtol=0, atol=0)
        self.assertGreater((visual(images, max_session=1) - before).abs().max().item(), 0)

    def test_topology_and_weights_restore_versioned_predictions(self):
        torch.manual_seed(13)
        source = VisualMoEAdapter(8, num_experts=2, reduction=2).eval()
        source.add_expert(1, reduction=4)
        with torch.no_grad():
            source.experts[-1].c_proj.bias.fill_(0.2)
        source.freeze_all()
        inputs = torch.randn(3, 2, 8)
        expected = [source(inputs, max_session=version)[0] for version in (0, 1)]
        restored = VisualMoEAdapter(8, num_experts=2, reduction=2)
        restored.rebuild_topology(source.export_topology())
        restored.load_state_dict(source.state_dict(), strict=True)
        for version, value in enumerate(expected):
            torch.testing.assert_close(restored(inputs, max_session=version)[0],
                                       value, rtol=0, atol=0)

    def test_visual_calibration_uses_current_bimc_prototype(self):
        model=BiMC.__new__(BiMC); torch.nn.Module.__init__(model)
        model.cfg=SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(LAMBDA_I=0.25,TAU=4)))
        model.base_vision_prototype=F.normalize(torch.eye(3)[:2],dim=-1)
        current=F.normalize(torch.tensor([[0.6,0.7,0.4]]),dim=-1)
        torch.testing.assert_close(model.calibrate_incremental_visual(current),model.soft_calibration(model.base_vision_prototype,current))

    def test_leave_one_out_excludes_the_query_source(self):
        features = F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                             [-1., 0.], [0., -1.]]), dim=-1)
        cache = build_loo_cache_from_embeddings(
            features, torch.tensor([2, 2, 3, 3]), torch.arange(4))
        torch.testing.assert_close(cache["loo_visual_prototypes"][0, 0], features[1])
        self.assertTrue(cache["loo_valid"].all())

    def test_visual_proto_is_not_recovered_from_normalized_fusion(self):
        model = BiMC.__new__(BiMC)
        torch.nn.Module.__init__(model)
        model.device = "cpu"
        model.template = ["a photo of a {}"]
        model.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
                TEXT_CALIBRATION=True, LAMBDA_T=0.5,
                FUSED_CONFLICT_STRENGTH=0.0, FUSED_CONFLICT_MARGIN=0.05)),
            DATASET=SimpleNamespace(GPT_PATH="unused", BETA=0.3))
        model.inference_text_feature = lambda *args: (
            F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1), None)
        model.inference_all_description_feature = lambda *args: (
            None, None, F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1))
        model.inference_all_img_feature = lambda *args, **kwargs: (
            torch.tensor([[1., 0.]]), torch.tensor([0]), torch.tensor([[1., 0.]]))
        state = model.build_task_statistics(["class"], None, [0], encoder_version=0)
        torch.testing.assert_close(state["image_proto"], torch.tensor([[1., 0.]]))
        self.assertEqual(state["encoder_version"], 0)

    def test_incremental_prototype_is_directly_calibrated_and_has_no_anchor_replay(self):
        model=BiMC.__new__(BiMC); torch.nn.Module.__init__(model)
        model.device,model.template="cpu",["a {}"]
        model.cfg=SimpleNamespace(DATASET=SimpleNamespace(GPT_PATH="unused",BETA=0.3),
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TEXT_CALIBRATION=False,LAMBDA_T=0.0,
                LAMBDA_I=0.25,TAU=4,FUSED_CONFLICT_STRENGTH=0.0,FUSED_CONFLICT_MARGIN=0.05)))
        model.base_vision_prototype=torch.tensor([[1.,0.]])
        model.inference_text_feature=lambda *args:(torch.tensor([[0.,1.]]),None)
        model.inference_all_description_feature=lambda *args:(None,None,torch.tensor([[0.,1.]]))
        current=torch.tensor([[0.6,0.8]])
        model.inference_all_img_feature=lambda *args,**kwargs:(current,torch.tensor([1]),F.normalize(current,dim=-1))
        state=model.build_task_statistics(["novel"],None,[1],calibrate_novel_vision_proto=True)
        torch.testing.assert_close(state["image_proto"],model.soft_calibration(model.base_vision_prototype,current))
        self.assertNotIn("anchor_features",state)

    def test_support_folds_exclude_every_held_out_source(self):
        runner = Runner.__new__(Runner)
        dataset = TinySupport()
        train = DataLoader(dataset, batch_size=4, shuffle=True)
        extract = DataLoader(dataset, batch_size=4, shuffle=False)
        folds = runner._support_folds(train, extract)
        self.assertEqual(len(folds), 5)
        for fit_train, fit_extract, validation in folds:
            fitting = set(fit_extract.dataset.indices)
            held_out = set(validation.dataset.indices)
            self.assertFalse(fitting & held_out)
            self.assertEqual(len(held_out), 2)
            self.assertEqual(len(fit_train.dataset), 8)

    def test_global_support_bank_includes_historical_negative_classes(self):
        class Model:
            def build_leave_one_out_prototypes(self,loader,class_index,max_session=None):
                fs,ys,ss=[],[],[]
                for batch in loader: fs.append(F.normalize(batch["image"],dim=-1));ys.append(batch["label"]);ss.append(batch["source_id"])
                return build_loo_cache_from_embeddings(torch.cat(fs),torch.cat(ys),torch.cat(ss),class_index)
            def calibrate_incremental_visual(self,current): return current
        runner=Runner.__new__(Runner);runner.device="cpu"
        runner.cfg=SimpleNamespace(DATASET=SimpleNamespace(BETA=0.3),TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TEXT_CALIBRATION=False,LAMBDA_T=0.0)))
        runner.model_without_dp=Model();runner._resume_state_dict_list=[{"class_ids":torch.tensor([0,1]),"fused_proto":torch.eye(4)[:2]}]
        loader=DataLoader(TinySupport(shots=3),batch_size=4,shuffle=False);text=F.normalize(torch.eye(4)[2:4],dim=-1)
        bank=runner._support_bank(loader,[2,3],text,text,version=1,calibrate=False)
        self.assertEqual(bank["classes"].tolist(),[0,1,2,3]);self.assertEqual(tuple(bank["fused_full"].shape),(4,4));self.assertEqual(tuple(bank["fused_loo"].shape),(6,4,4))

    def test_new_router_column_starts_at_calibrated_low_mass(self):
        torch.manual_seed(9)
        router = __import__("models.clip.model", fromlist=["ExpandableRouter"]).ExpandableRouter(4, 2)
        inputs = torch.randn(12, 4)
        router.add_column()
        mass = router.initialize_newest_column(inputs, target_mass=0.01)
        _, weights = router(inputs)
        self.assertAlmostEqual(mass, 0.01, delta=0.002)
        self.assertAlmostEqual(float(weights[:, -1].mean()), 0.01, delta=0.002)


if __name__ == "__main__":
    unittest.main()
