"""Global scoring, source exclusion, optimization, and full runner lifecycle."""
import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.global_moe import (FeatureExpert, FeatureRoute, GlobalFeatureMoE,
                               balanced_loss, fused_prototypes, unit)
from engine.feature_risk import (FeatureRiskTrainer, RiskConfig, combine,
                                 select_candidate, source_folds, subset)


torch.set_num_threads(1)


def cache(ids=(0, 1), dim=8, shots=5, start=0):
    gen = torch.Generator().manual_seed(10 + ids[0])
    labels = torch.tensor(ids).repeat_interleave(shots)
    text = torch.eye(dim)[list(ids)]
    features = unit(torch.eye(dim)[labels] + .12 * torch.randn(len(labels), dim, generator=gen))
    return dict(features=features, labels=labels, sources=torch.arange(start, start + len(labels)),
                class_ids=torch.tensor(ids), text=text)


def initialize(base):
    head = GlobalFeatureMoE(base['features'].shape[1], 10)
    route = FeatureRoute(head.dim)
    proto = fused_prototypes(route, [], base, base['text'], base, .3, 0, 16)
    head.bind(route, base['class_ids'], proto)
    return head


def test_old_scores_bitwise_stable_across_expansion_and_reuse():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    query = torch.randn(20, 8)
    old = head(query).clone()
    expert = FeatureExpert(8, 3)
    with torch.no_grad():
        expert.up.weight.normal_(std=.5)
    route = FeatureRoute(8, [0])
    p = fused_prototypes(route, [expert], current, current['text'], base, .3, .1, 16)
    head.bind(route, current['class_ids'], p, expert)
    middle = head(query).clone()
    later = cache((4, 5), start=20)
    head.bind(FeatureRoute(8, [0]), later['class_ids'], later['text'])
    assert torch.equal(old, head(query)[:, :2])
    assert torch.equal(middle, head(query)[:, :4])
    assert not any(p.requires_grad for p in head.parameters())
    restored = GlobalFeatureMoE.restore(head.export())
    assert torch.equal(head(query), restored(query))


def test_zero_initialized_expansion_preserves_identity():
    z = unit(torch.randn(10, 8))
    route, expert = FeatureRoute(8, [0]), FeatureExpert(8, 3)
    torch.testing.assert_close(route(z, [expert]), FeatureRoute(8)(z, []), rtol=1e-6, atol=1e-7)


def test_source_exclusion_removes_all_views_and_keeps_gradients():
    current = cache((2, 3))
    current['sources'][1] = current['sources'][0]
    expert = FeatureExpert(8, 3)
    with torch.no_grad():
        expert.up.weight.normal_(std=.1)
    route = FeatureRoute(8, [0])
    result = fused_prototypes(route, [expert], current, current['text'], cache(), .3, 0, 16,
                              current['labels'][:1], current['sources'][:1])
    mask = current['labels'].eq(2) & current['sources'].ne(current['sources'][0])
    expected = unit(.3 * current['text'][0] + .7 * unit(route(current['features'][mask], [expert]).mean(0)))
    torch.testing.assert_close(result[0, 0], expected)
    result[0, 0, 0].backward()
    assert expert.up.weight.grad.abs().sum() > 0
    changed = copy.deepcopy(current)
    changed['features'][:2] = torch.randn(2, 8) * 20
    again = fused_prototypes(route, [expert], changed, changed['text'], cache(), .3, 0, 16,
                             changed['labels'][:1], changed['sources'][:1])
    torch.testing.assert_close(result, again)


def test_calibration_transforms_base_samples_in_matching_space():
    base, current = cache(), cache((2, 3))
    route, expert = FeatureRoute(8, [0]), FeatureExpert(8, 2)
    with torch.no_grad():
        expert.up.weight.normal_()
    actual = fused_prototypes(route, [expert], current, current['text'], base, 0, .2, 4)
    v = torch.stack([unit(route(current['features'][current['labels'].eq(c)], [expert]).mean(0))
                     for c in current['class_ids']])
    b = torch.stack([unit(route(base['features'][base['labels'].eq(c)], [expert]).mean(0))
                     for c in base['class_ids']])
    expected = unit(.8 * v + .2 * unit((v @ b.T * 4).softmax(-1) @ b))
    torch.testing.assert_close(actual, expected)


def test_old_negative_samples_train_new_expert_without_changing_old_parameters():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=1))
    route, expert = trainer._make('expand', torch.device('cpu'), 4)
    logits, _ = trainer._logits(route, expert, base, current, base, head(base['features']))
    loss = balanced_loss(logits, base['labels'], torch.cat([base['class_ids'], current['class_ids']]))
    loss.backward()
    assert expert.up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in head.parameters())


def test_class_balance_is_invariant_to_repeating_a_class():
    logits = torch.tensor([[2., 0.], [3., 0.]])
    labels = torch.tensor([0, 1])
    ids = torch.tensor([0, 1])
    a = balanced_loss(logits, labels, ids)
    b = balanced_loss(logits[[0, 0, 0, 1]], labels[[0, 0, 0, 1]], ids)
    torch.testing.assert_close(a, b)


def test_fold_assignment_groups_sources_not_views():
    c = cache(shots=6)
    c['sources'][1] = c['sources'][0]
    assignment = source_folds(c, 5, 7)
    assert assignment[0] == assignment[1]
    assert torch.equal(assignment, source_folds(c, 5, 7))
    for fold in range(5):
        for cls in c['class_ids']:
            train = c['sources'][(assignment != fold) & (c['labels'] == cls)]
            val = c['sources'][(assignment == fold) & (c['labels'] == cls)]
            assert len(val) and not torch.isin(train, val).any()


def test_selection_filters_before_ranking_and_never_uses_standard_error():
    rows = [dict(name='a', gain=.03, old_regret=.02, risk=1., parameters=30),
            dict(name='b', gain=.01, old_regret=-.01, risk=1.02, parameters=20)]
    assert select_candidate(rows, 0, 0)['name'] == 'b'
    assert select_candidate(rows, .04, 0) is None


def test_actual_candidate_training_and_head_mutation_are_transactional():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=2, batch_size=4))
    before = head(base['features']).clone()
    # Exercise accepted refit with real gradients independently of stochastic
    # model-selection power on this deliberately trivial synthetic dataset.
    with patch('engine.feature_risk.select_candidate', side_effect=lambda rows, *_: rows[-1]):
        report = trainer.add_session(current, [base])
    assert report['selected'] == 'expand' and len(head.experts) == 1
    assert head.experts[0].up.weight.abs().sum() > 0
    assert torch.equal(before, head(base['features'])[:, :2])
    scores = head(base['features']).clone()
    with patch.object(trainer, '_fit', side_effect=RuntimeError('injected failure')):
        with pytest.raises(RuntimeError, match='injected'):
            trainer.add_session(cache((4, 5), start=20), [base, current])
    assert len(head.banks) == 2 and len(head.experts) == 1
    assert torch.equal(scores, head(base['features']))


def test_rejected_expansion_leaves_pool_empty_but_binds_new_classes():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=1, min_gain=1000.))
    report = trainer.add_session(current, [base])
    assert report['selected'] == 'identity'
    assert len(head.banks) == 2 and len(head.experts) == 0


def test_cross_validation_does_not_feed_validation_sources_to_candidates():
    base, current = cache(), cache((2, 3), start=10)
    trainer = FeatureRiskTrainer(initialize(base), RiskConfig(rank=2, epochs=1))
    original_fit, original_risk = trainer._fit, trainer._risk
    seen = {}
    def fit(name, cur, hist, ref, seed):
        seen['train'] = torch.cat([cur['sources'], hist['sources'], ref['sources']])
        return original_fit(name, cur, hist, ref, seed)
    def risk(route, extra, val, support, ref):
        assert not torch.isin(val['sources'], seen['train']).any()
        return original_risk(route, extra, val, support, ref)
    with patch.object(trainer, '_fit', side_effect=fit), patch.object(trainer, '_risk', side_effect=risk):
        trainer.add_session(current, [base])


def test_gradients_do_not_depend_on_chunk_size():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    a = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=1, batch_size=4))
    b = FeatureRiskTrainer(head, replace(a.cfg, batch_size=100))
    ra, ea = a._fit('expand', current, base, base, 42)
    rb, eb = b._fit('expand', current, base, base, 42)
    for pa, pb in zip(ea.parameters(), eb.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)


def test_candidate_initialization_does_not_consume_global_rng():
    trainer = FeatureRiskTrainer(initialize(cache()), RiskConfig(rank=2))
    before = torch.random.get_rng_state().clone()
    trainer._make('expand', torch.device('cpu'), 99)
    assert torch.equal(before, torch.random.get_rng_state())


def test_feature_expert_can_change_cross_cohort_winner():
    head = GlobalFeatureMoE(3, 10)
    head.bind(FeatureRoute(3), torch.tensor([0]), torch.tensor([[1., 0, 0]]))
    expert = FeatureExpert(3, 2)
    with torch.no_grad():
        expert.up.bias.copy_(torch.tensor([-2., 4., 0.]))
    head.bind(FeatureRoute(3, [0]), torch.tensor([1]), torch.tensor([[0., 1, 0]]), expert)
    z = unit(torch.tensor([[1., .1, 0.]]))
    # No task gate: a novel expert can win against an old class directly.
    assert head(z).argmax(1).item() == 1


class FeatureDataset(Dataset):
    def __init__(self, c):
        self.c = c
        self.source_ids = c['sources'].numpy()
        self.labels = c['labels'].numpy()
    def __len__(self):
        return len(self.c['labels'])
    def __getitem__(self, i):
        return dict(image=self.c['features'][i], label=self.c['labels'][i], source_id=self.c['sources'][i])


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('anchor', torch.tensor(1.))
    def extract_img_feature(self, x, max_session=None):
        assert max_session == 0
        return x * self.anchor
    def export_moe_topology(self):
        return {}
    def rebuild_moe_topology(self, topology):
        assert topology == {}


def tiny_runner(tmp_path, resume='', end=2):
    from main import setup_cfg
    from engine.global_runner import GlobalRiskRunner
    from utils.evaluator import AccuracyEvaluator
    cfg = setup_cfg('configs/datasets/cub200.yaml', 'configs/trainers/bimc.yaml', [
        'DEVICE.DEVICE_NAME', 'cpu', 'TRAINER.BiMC.GLOBAL_MOE.RANK', '2',
        'TRAINER.BiMC.GLOBAL_MOE.EPOCHS', '1', 'TRAINER.BiMC.GLOBAL_MOE.MIN_GAIN', '1000.0',
        'TRAINER.BiMC.CHECKPOINT.DIR', str(tmp_path), 'TRAINER.BiMC.CHECKPOINT.RESUME', resume,
        'TRAINER.BiMC.END_SESSION', str(end)])
    memories = [cache((2*i, 2*i+1), start=10*i) for i in range(3)]
    class Manager:
        class_names = list('abcdef')
        class_index_in_task = [[0, 1], [2, 3], [4, 5]]
        num_tasks = 3
        def get_support_manifest(self):
            return {'synthetic': True}
        def load_support_manifest(self, m):
            assert m == self.get_support_manifest()
        def get_dataloader(self, task_id, source, mode, accumulate_past=False):
            c = combine(memories[:task_id+1]) if source == 'test' else memories[task_id]
            return DataLoader(FeatureDataset(c), batch_size=6)
    runner = GlobalRiskRunner.__new__(GlobalRiskRunner)
    runner.cfg, runner.device = cfg, 'cpu'
    runner.model = runner.model_without_dp = TinyEncoder()
    runner.data_manager = Manager()
    runner.risk_config = runner.make_risk_config(cfg)
    runner.feature_head = None
    runner.feature_memories = []
    runner.evaluator = AccuracyEvaluator(Manager.class_index_in_task)
    runner._resume_completed_session = -1
    runner._resume_state_dict_list = []
    runner.acc_list, runner.task_acc_list, runner.session_diagnostics = [], [], []
    runner._text_prototypes = lambda names, ids: (torch.eye(8)[ids], torch.eye(8)[ids])
    if resume:
        runner._load_resume_checkpoint_if_requested()
    else:
        runner._initialize_base(memories[0])
        result = runner.evaluate_global(0)
        runner.acc_list = [result['mean_acc']]
        runner.task_acc_list = [result['task_acc']]
        runner.session_diagnostics = [{'session': 0, 'classifier': result}]
        runner._resume_completed_session = 0
    return runner


@patch('engine.feature_risk.select_candidate', side_effect=lambda rows, *_: rows[-1])
def test_full_three_session_runner_resume_matches_continuous(_selection, tmp_path):
    uninterrupted = tiny_runner(tmp_path / 'continuous')
    uninterrupted.run()
    first = tiny_runner(tmp_path / 'split', end=1)
    first.run()
    resumed = tiny_runner(tmp_path / 'split', str(tmp_path / 'split' / 'session_01.pth'))
    resumed.run()
    assert len(resumed.feature_head.experts) == 2
    assert uninterrupted.acc_list == resumed.acc_list
    assert uninterrupted.task_acc_list == resumed.task_acc_list
    for a, b in zip(uninterrupted.feature_head.state_dict().values(), resumed.feature_head.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(uninterrupted.feature_memories, resumed.feature_memories):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert resumed.session_diagnostics[-1]['AA'] == sum(resumed.acc_list) / 3


def test_resume_rejects_changed_protocol(tmp_path):
    runner = tiny_runner(tmp_path)
    runner.run()
    path = tmp_path / 'session_02.pth'
    payload = torch.load(path, weights_only=False)
    payload['protocol']['risk']['beta'] = .8
    torch.save(payload, path)
    with pytest.raises(ValueError, match='protocol'):
        tiny_runner(tmp_path, str(path))


def test_real_tiny_vit_stays_frozen_while_feature_expert_trains():
    from models.clip.model import VisionTransformer
    torch.manual_seed(18)
    encoder = VisionTransformer(8, 4, 8, 2, 2, 8).eval().requires_grad_(False)
    images = torch.randn(20, 3, 8, 8)
    weights = copy.deepcopy(encoder.state_dict())
    with torch.no_grad():
        z = unit(encoder(images))
    base, current = cache(), cache((2, 3), start=10)
    base['features'], current['features'] = z[:10], z[10:]
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=2))
    with patch('engine.feature_risk.select_candidate', side_effect=lambda rows, *_: rows[-1]):
        trainer.add_session(current, [base])
    assert head.experts[0].up.weight.abs().sum() > 0
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(v, weights[k]) for k, v in encoder.state_dict().items())


def test_fresh_base_session_runs_through_image_entrypoint(tmp_path):
    runner = tiny_runner(tmp_path, end=0)
    runner.cfg.defrost()
    runner.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE = False
    runner.cfg.freeze()
    runner.model_without_dp.build_task_statistics = lambda *a, **kw: {}
    runner._resume_completed_session = -1
    runner.feature_head, runner.feature_memories = None, []
    runner.acc_list, runner.task_acc_list, runner.session_diagnostics = [], [], []
    runner.run()
    assert len(runner.feature_head.banks) == 1
    assert runner._resume_completed_session == 0
    assert (tmp_path / 'session_00.pth').is_file()


def test_expansion_budget_still_allows_frozen_expert_reuse():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=1, max_experts=1))
    with patch('engine.feature_risk.select_candidate', side_effect=lambda rows, *_: rows[-1]):
        trainer.add_session(current, [base])
        result = trainer.add_session(cache((4, 5), start=20), [base, current])
    assert result['selected'] == 'reuse'
    assert len(head.experts) == 1 and len(head.banks) == 3


def test_bind_rejects_invalid_state_without_pool_mutation():
    head = initialize(cache())
    before = head.export()
    with pytest.raises(ValueError, match='existing'):
        head.bind(FeatureRoute(8, [0]), torch.tensor([0]), torch.eye(8)[:1], FeatureExpert(8, 2))
    assert len(head.experts) == 0 and len(head.banks) == 1
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in before['state'].items())


def test_unmocked_risk_selection_accepts_a_useful_expert():
    generator = torch.Generator().manual_seed(10)
    base, current = cache(), cache((2, 3), start=100)
    base['features'] = unit(torch.eye(8)[base['labels']] + .02 * torch.randn(10, 8, generator=generator))
    current['features'] = unit(.8 * torch.eye(8)[current['labels'] - 2]
                               + .2 * torch.eye(8)[current['labels']]
                               + .02 * torch.randn(10, 8, generator=generator))
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=4, epochs=30, lr=.03))
    report = trainer.add_session(current, [base])
    assert report['selected'] == 'expand'
    candidate = report['candidates'][-1]
    assert candidate['gain'] > 0 and candidate['old_regret'] <= 0


def test_schema9_base_migration_and_legacy_incremental_rejection(tmp_path):
    runner = tiny_runner(tmp_path / 'new')
    runner._moe_adapters = lambda: []
    memory = runner.feature_memories[0]
    old_dir = tmp_path / 'old'
    old_dir.mkdir()
    old_path = old_dir / 'base.pth'
    payload = dict(schema_version=9, completed_session=0,
                   model_state=runner.model_without_dp.state_dict(), moe_topology={},
                   task_states=[dict(class_ids=memory['class_ids'])],
                   support_manifest=runner.data_manager.get_support_manifest(),
                   task_gate_biases=[0.], acc_list=[1.23], task_acc_list=[[1.23]],
                   session_diagnostics=[])
    torch.save(payload, old_path)
    original_bytes = old_path.read_bytes()
    runner.cfg.defrost()
    runner.cfg.TRAINER.BiMC.CHECKPOINT.RESUME = str(old_path)
    runner.cfg.freeze()
    runner._load_resume_checkpoint_if_requested()
    assert runner.acc_list[0] != 1.23  # raw accuracy is re-evaluated
    assert len(runner.feature_memories) == 1
    assert old_path.read_bytes() == original_bytes
    assert torch.load(tmp_path / 'new' / 'session_00.pth', weights_only=False)['schema_version'] == 10
    payload['completed_session'] = 1
    torch.save(payload, old_path)
    with pytest.raises(ValueError, match='cannot be converted'):
        runner._load_resume_checkpoint_if_requested()


def test_failed_final_refit_does_not_commit_candidate():
    base, current = cache(), cache((2, 3), start=10)
    head = initialize(base)
    trainer = FeatureRiskTrainer(head, RiskConfig(rank=2, epochs=1))
    actual_fit = trainer._fit
    def fit(name, cur, *args):
        if len(cur['labels']) == 10:
            raise FloatingPointError('failed refit')
        return actual_fit(name, cur, *args)
    with patch.object(trainer, '_fit', side_effect=fit):
        with pytest.raises(FloatingPointError, match='refit'):
            trainer.add_session(current, [base])
    assert len(head.banks) == 1 and len(head.experts) == 0
