import unittest
import torch
from engine.apc import AdaptationAwarePrototypeController
from engine.apc_runner import APCRunner
from models.clip.model import VisualMoEAdapter


class APCTests(unittest.TestCase):
    def test_merge_block_input_anchors_normalizes_devices_and_sorts_classes(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states=[
            {
                "class_ids":torch.tensor([3],device=device),
                "block_input_counts":{"2":torch.tensor([2],device="cpu")},
                "block_input_anchors":{"2":torch.tensor([[30.],[31.]],device=device)},
            },
            {
                "class_ids":torch.tensor([1,2],device="cpu"),
                "block_input_counts":{"2":torch.tensor([1,2],device=device)},
                "block_input_anchors":{"2":torch.tensor([[10.],[20.],[21.]],device="cpu")},
            },
        ]
        ids,anchors,counts=APCRunner._merge_block_input_anchors(states,"2")
        self.assertEqual(ids.device.type,"cpu")
        self.assertEqual(anchors.device.type,"cpu")
        self.assertEqual(counts.device.type,"cpu")
        torch.testing.assert_close(ids,torch.tensor([1,2,3]))
        torch.testing.assert_close(counts,torch.tensor([1,2,2]))
        torch.testing.assert_close(anchors,torch.tensor([[10.],[20.],[21.],[30.],[31.]]))

    def test_historical_class_metadata_is_normalized_to_cpu(self):
        source_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_ids=torch.tensor([2,5,9],device=source_device)
        class_counts=torch.tensor([3,1,2],device="cpu")
        ids,counts,labels=APCRunner._cpu_class_metadata(class_ids,class_counts)
        self.assertEqual(ids.device.type,"cpu")
        self.assertEqual(counts.device.type,"cpu")
        self.assertEqual(labels.device.type,"cpu")
        torch.testing.assert_close(ids,torch.tensor([2,5,9]))
        torch.testing.assert_close(counts,torch.tensor([3,1,2]))
        torch.testing.assert_close(labels,torch.tensor([2,2,2,5,9,9]))

    def test_reproducible_calibration_and_family_size(self):
        g=torch.Generator().manual_seed(11); y=torch.arange(8).repeat_interleave(6)
        x=torch.eye(8)[y]+.02*torch.randn(48,8,generator=g)
        f={2:{"cls_in":x,"labels":y,"source_ids":torch.arange(48)}}
        c=AdaptationAwarePrototypeController(.05,32,7); a=c.calibrate(f,2,3); b=c.calibrate(f,2,3)
        torch.testing.assert_close(a["pair_scores"],b["pair_scores"],rtol=0,atol=0)
        self.assertGreaterEqual(c.threshold(a,2,20),c.threshold(a,2,3))

    def test_risk_uses_adapted_prototypes(self):
        y=torch.arange(6).repeat_interleave(4); x=torch.eye(6)[y]
        f={1:{"cls_in":x,"labels":y,"source_ids":torch.arange(24)}}
        c=AdaptationAwarePrototypeController(.05,16,3); s=c.calibrate(f,1,2)
        old=torch.eye(6)[:4]
        new={1:{"cls_in":torch.eye(6)[4:].repeat_interleave(4,0),
                "labels":torch.tensor([4]*4+[5]*4),"source_ids":torch.arange(8)}}
        d=c.decide(new,{"1":old},torch.arange(4),torch.tensor([4,5]),s)
        self.assertEqual(d[0]["block_idx"],1)

    def test_zero_up_expert_preserves_old_function_and_incremental_weights_bounded(self):
        a=VisualMoEAdapter(8,2,2).eval(); x=torch.randn(3,2,8)
        before,_=a(x); a.add_expert(1,2); after,aux=a(x)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertLess(float(aux["route_weights"][:,2:].sum(-1).max()),1.0)

    def test_old_prototype_protection_has_new_only_gradients(self):
        a=VisualMoEAdapter(8,2,2); a.freeze_all(); a.add_expert(1,2)
        gate,res=a.newest_contribution(torch.randn(4,8))
        (gate.square().mean()+res.square().mean()).backward()
        self.assertTrue(any(p.grad is not None for p in a.experts[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in a.experts[:-1].parameters()))


if __name__=="__main__": unittest.main()
