import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.bimc import BiMC, LayerExpansionDecision, build_loo_cache_from_embeddings
from engine.engine import Runner
from main import setup_cfg
from utils.metrics import session_metrics

# Load the mathematical CLIP module without importing the tokenizer/download path.
spec = importlib.util.spec_from_file_location('clip_math', Path(__file__).parents[1] / 'models/clip/model.py')
clip_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clip_math)
torch.set_num_threads(1)


def make_runner():
    torch.manual_seed(12)
    cfg = setup_cfg('configs/datasets/cifar100.yaml', 'configs/trainers/bimc.yaml', [
        'DEVICE.DEVICE_NAME', 'cpu', 'TRAINER.BiMC.VISUAL_MOE.NUM_BASE_EXPERTS', '2',
        'TRAINER.BiMC.INCREMENTAL.EPOCHS', '2',
        'TRAINER.BiMC.OPTIM.INCREMENTAL_DESCRIPTOR_EPOCHS', '1'])
    model = BiMC.__new__(BiMC)
    torch.nn.Module.__init__(model)
    model.cfg, model.device, model.template = cfg, 'cpu', ['{}']
    model.clip_model = clip_math.CLIP(8, 8, 2, 64, 4, 8, 32, 64, 1, 1)
    for block in model.clip_model.visual.transformer.resblocks:
        block.moe_adapter = clip_math.VisualMoEAdapter(64, 2, descriptor_dim=4)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.base_vision_prototype = F.normalize(torch.randn(2, 8), dim=-1)
    model._fused_history_prototypes = model._fused_history_class_ids = None
    model.eval()
    runner = Runner.__new__(Runner)
    runner.cfg, runner.device = cfg, 'cpu'
    runner.model = runner.model_without_dp = model
    runner.acc_list, runner.task_acc_list, runner.session_diagnostics = [], [], []
    return runner


def support():
    return [{'image': torch.randn(4, 3, 8, 8), 'label': torch.tensor([2, 2, 3, 3]),
             'source_id': torch.tensor([10, 11, 12, 13])}]


def state(runner, ids=(0, 1)):
    bank = F.normalize(torch.randn(len(ids), 8), dim=-1)
    return {'class_ids': torch.tensor(ids), 'image_proto': bank.clone(),
            'text_features': bank.clone(), 'description_proto': bank.clone(),
            'fused_proto': bank.clone(), 'feature_path': runner.model.feature_path()}


def test_old_scores_invariant_after_upstream_and_downstream_expansion():
    r = make_runner()
    old = r.merge_dicts([state(r)])
    images = torch.randn(3, 3, 8, 8)
    expected = r.model.classifier_scores(images, old).detach()
    for i, adapter in r._moe_adapters():
        adapter.add_expert(i + 1)
        # Strong perturbation exposes both softmax dilution and downstream input drift.
        with torch.no_grad():
            for p in adapter.newest_parameters():
                p.normal_(0, 2)
        adapter.freeze_all()
    actual = r.model.classifier_scores(images, old)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    all_state = r.merge_dicts([dict(old, feature_path={'0': 2, '1': 2}), state(r, (2,))])
    torch.testing.assert_close(r.model.classifier_scores(images, all_state)[:, :2], expected, atol=0, rtol=0)


def test_feature_context_restores_after_invalid_path_and_forward_failure():
    r = make_runner()
    with pytest.raises(ValueError):
        with r.model.use_feature_path({'0': 2, '1': 99}):
            pass
    assert all(a.active_expert_count is None for _, a in r._moe_adapters())
    with pytest.raises(RuntimeError):
        with r.model.use_feature_path(r.model.feature_path()):
            raise RuntimeError('forward failed')
    assert all(a.active_expert_count is None for _, a in r._moe_adapters())


def test_duplicate_source_views_excluded_and_one_shot_invalid():
    cache = build_loo_cache_from_embeddings(torch.tensor([[1., 0], [1., 0], [0., 1], [-1., 0]]),
                                           [2, 2, 2, 7], [10, 10, 11, 12])
    torch.testing.assert_close(cache['loo_visual_prototypes'][0, 0], torch.tensor([0., 1]))
    assert cache['loo_valid'].tolist() == [True, True, True, False]


def test_training_uses_calibrated_fusion_and_frozen_old_paths():
    r = make_runner()
    loader = support()
    old = r.merge_dicts([state(r)])
    text = F.normalize(torch.randn(2, 8), dim=-1)
    r.model.add_expert(0, 1)
    cache = r._current_support_cache(loader, [2, 3], text)
    batch = loader[0]
    logits, targets, _ = r._incremental_loo_logits(batch['image'], batch['label'], batch['source_id'], cache, old)
    feat = F.normalize(r.model.extract_img_feature(batch['image']), dim=-1)
    vis = r.model.calibrate_visual(cache['loo_visual_prototypes'])
    fused = F.normalize(r.cfg.DATASET.BETA * text[None] + (1-r.cfg.DATASET.BETA) * vis, dim=-1)
    expected = torch.cat([r.model.classifier_scores(batch['image'], old), torch.einsum('bd,bcd->bc', feat, fused)], 1)
    torch.testing.assert_close(logits, expected * r.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
    assert targets.tolist() == [2, 2, 3, 3]
    historical = r._historical_snapshot(r._adapter_by_block(1))
    r.train_expanded_module(1, 0, loader, loader, [2, 3], text, old)
    assert r._historical_state_unchanged(r._adapter_by_block(1), historical)[0]
    assert not any(p.requires_grad for n, p in r.model.named_parameters() if 'moe_adapter' not in n)


@pytest.mark.parametrize('outcome', ['reject', 'accept', 'exception'])
def test_candidate_transaction(outcome):
    r = make_runner()
    loader = support()
    old = state(r)
    text = F.normalize(torch.randn(2, 8), dim=-1)
    r.model.inference_text_feature = lambda *a: (text, None)
    r.model.inference_all_description_feature = lambda *a: (None, None, text)
    r.model.plan_layer_expansion = lambda i, *a, **k: LayerExpansionDecision(i, i == 0, {2: 1.}, [2])
    scores = iter([{'loss': 1., 'accuracy': .5, 'deficit': {10: 1., 11: 1., 12: 1., 13: 1.}},
                   {'loss': .5 if outcome == 'accept' else 2., 'accuracy': .5, 'deficit': {}}])
    r._support_objective = lambda *a: next(scores)
    before = copy.deepcopy(r.model.state_dict())
    path = r.model.feature_path()
    def train(*args):
        torch.rand(10)
        if outcome == 'exception':
            raise RuntimeError('training failure')
        return []
    r.train_expanded_module = train
    r.train_descriptor_for_expert = lambda *a: None
    rng = torch.get_rng_state().clone()
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='training failure'):
            r.train_incremental_task(1, loader, loader, [2, 3], ['a', 'b'], [old], True)
    else:
        result = r.train_incremental_task(1, loader, loader, [2, 3], ['a', 'b'], [old], True)
        assert result['trained'] == (outcome == 'accept')
    if outcome != 'accept':
        assert r.model.feature_path() == path
        for key, value in before.items():
            torch.testing.assert_close(r.model.state_dict()[key], value, rtol=0, atol=0)
        # Candidate creation/training randomness rolls back; baseline deterministic forwards consume none.
        assert torch.equal(rng, torch.get_rng_state())
    assert not any(p.requires_grad for p in r.model.parameters())


def test_visual_prototype_not_reconstructed_from_fusion():
    r = make_runner()
    model = r.model
    visual = F.normalize(torch.randn(2, 8), dim=-1)
    text = F.normalize(torch.randn(2, 8), dim=-1)
    model.inference_text_feature = lambda *a: (text, None)
    model.inference_all_description_feature = lambda *a: (None, None, text)
    model.inference_all_img_feature = lambda *a: (visual, torch.tensor([0, 1]), visual)
    result = model.build_task_statistics(['a', 'b'], [], [0, 1])
    torch.testing.assert_close(result['image_proto'], visual, rtol=0, atol=0)


def test_metrics_include_base_and_do_not_normalize_pd():
    assert session_metrics([80., 70., 60.]) == {'AA': 70., 'PD': 20., 'sessions': 3}
    assert session_metrics([80.])['PD'] == 0
    assert session_metrics([60., 80.])['PD'] == -20
    for values in ([], [float('nan')], [101.]):
        with pytest.raises(ValueError):
            session_metrics(values)


def test_checkpoint_roundtrip_preserves_paths_scores_and_metrics(tmp_path):
    r = make_runner()
    old = state(r)
    r.model.add_expert(0, 1)
    r._adapter_by_block(0).freeze_all()
    current = state(r, (2, 3))
    states = [old, current]
    images = torch.randn(3, 3, 8, 8)
    scores = r.model.classifier_scores(images, r.merge_dicts(states)).detach()
    cfg = r.cfg.clone()
    cfg.defrost()
    cfg.TRAINER.BiMC.CHECKPOINT.DIR = str(tmp_path)
    cfg.freeze()
    r.cfg = cfg
    r.acc_list = [81.1234567, 75.654321]
    r.data_manager = SimpleNamespace(get_support_manifest=lambda: {'support': 'fixed'})
    path = r.save_checkpoint(1, states)
    restored = make_runner()
    cfg = restored.cfg.clone()
    cfg.defrost()
    cfg.TRAINER.BiMC.CHECKPOINT.RESUME = str(path)
    cfg.freeze()
    restored.cfg = cfg
    restored.data_manager = SimpleNamespace(load_support_manifest=lambda value: None)
    restored._load_resume_checkpoint_if_requested()
    actual = restored.model.classifier_scores(images, restored.merge_dicts(restored._resume_state_dict_list))
    torch.testing.assert_close(actual, scores, atol=0, rtol=0)
    assert restored.acc_list == r.acc_list
    assert session_metrics(restored.acc_list) == session_metrics(r.acc_list)
    payload = torch.load(path, weights_only=False)
    payload['schema_version'] = 6
    torch.save(payload, path)
    with pytest.raises(ValueError, match='incompatible'):
        restored._load_resume_checkpoint_if_requested()


def test_descriptor_expansion_requires_classification_failure():
    r = make_runner()
    loader = support()
    for _, adapter in r._moe_adapters():
        for descriptor in adapter.descriptors:
            descriptor.fit_sample_count.fill_(2)
            descriptor.calibration_sample_count.fill_(2)
            descriptor.mean.fill_(-10.)
            descriptor.std.fill_(1.)
    no_error = {i: 0. for i in [10, 11, 12, 13]}
    assert not r.model.plan_layer_expansion(0, loader, [2, 3], deficits=no_error).expand
    error = dict(no_error)
    error[10] = 1.
    assert r.model.plan_layer_expansion(0, loader, [2, 3], deficits=error).expand


def test_auto_none_skips_candidate_training():
    r = make_runner()
    cfg = r.cfg.clone()
    cfg.defrost()
    cfg.TRAINER.BiMC.VISUAL_MOE.EXPANSION_MODE = 'none'
    cfg.freeze()
    r.cfg = cfg
    result = r.train_incremental_task(1, [], [], [2, 3], ['a', 'b'], [state(r)], True)
    assert result == {'trained': False, 'decisions': [], 'expanded': []}


def test_three_session_run_matches_resume(tmp_path):
    from utils.evaluator import AccuracyEvaluator
    generator = torch.Generator().manual_seed(101)
    batches = [{'image': torch.randn(4, 3, 8, 8, generator=generator),
                'label': torch.tensor([2*t, 2*t, 2*t+1, 2*t+1]),
                'source_id': torch.arange(4*t, 4*t+4)} for t in range(3)]
    text = F.normalize(torch.randn(6, 8, generator=generator), dim=-1)

    def build(directory, end=-1, resume=''):
        r = make_runner()
        cfg = r.cfg.clone()
        cfg.defrost()
        cfg.TRAINER.BiMC.OPTIM.BASE_EPOCHS = 1
        cfg.TRAINER.BiMC.OPTIM.BASE_DESCRIPTOR_EPOCHS = 1
        cfg.TRAINER.BiMC.CHECKPOINT.DIR = str(directory)
        cfg.TRAINER.BiMC.CHECKPOINT.RESUME = str(resume)
        cfg.TRAINER.BiMC.END_SESSION = end
        cfg.freeze()
        r.cfg = r.model.cfg = cfg
        r.model._apply_freeze_policy()
        r.model.inference_text_feature = lambda names, template, start: (text[start:start+len(names)], None)
        r.model.inference_all_description_feature = lambda names, path, start: (None, None, text[start:start+len(names)])
        def loader(task, source, mode, **kwargs):
            if source == 'test':
                return [{key: torch.cat([b[key] for b in batches[:task+1]]) for key in batches[0]}]
            return [batches[task]]
        r.data_manager = SimpleNamespace(num_tasks=3, class_index_in_task=[[0, 1], [2, 3], [4, 5]],
                                         class_names=list('abcdef'), get_dataloader=loader,
                                         get_support_manifest=lambda: {'synthetic': True},
                                         load_support_manifest=lambda manifest: None)
        r.evaluator = AccuracyEvaluator(r.data_manager.class_index_in_task)
        r._resume_state_dict_list, r._resume_completed_session = [], -1
        r._load_resume_checkpoint_if_requested()
        return r

    full = build(tmp_path / 'full')
    full.run()
    partial = build(tmp_path / 'split', end=0)
    partial.run()
    resumed = build(tmp_path / 'split', resume=tmp_path / 'split/session_00.pth')
    resumed.run()
    assert resumed.acc_list == full.acc_list
    assert resumed.model.feature_path() == full.model.feature_path()
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, atol=0, rtol=0)


def test_new_descriptor_keeps_its_input_after_later_upstream_expansion():
    r = make_runner()
    loader = support()
    for _, adapter in r._moe_adapters():
        for descriptor in adapter.descriptors:
            descriptor.fit_sample_count.fill_(2)
            descriptor.calibration_sample_count.fill_(2)
    r.model.add_expert(1, 1)
    r.train_descriptor_for_expert(1, loader)
    descriptor = r._adapter_by_block(1).descriptors[-1]
    path = r.model.feature_path()
    with r.model.use_feature_path(path):
        before = r.model.collect_blockwise_cls_features(loader)[1]['cls_in']
    r.model.add_expert(0, 2)
    with torch.no_grad():
        for parameter in r._adapter_by_block(0).newest_parameters():
            parameter.normal_(0, 3)
    with r.model.use_feature_path(path):
        after = r.model.collect_blockwise_cls_features(loader)[1]['cls_in']
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(descriptor.standardized_score(before), descriptor.standardized_score(after), rtol=0, atol=0)
    assert descriptor.is_ready()
