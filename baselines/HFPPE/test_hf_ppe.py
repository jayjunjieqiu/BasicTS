import os
import unittest

import torch

from arch import HFPPE, hf_ppe_loss


class TestHFPPE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shapes_history_and_future(self):
        B, Lh, Lf, N, C = 2, 36, 24, 5, 1
        model = HFPPE(enc_in=N, seq_len=Lh, pred_len=Lf, d_model=64, n_heads=4, e_layers=2, d_ff=128, m=3, ctx_len=12)
        history = torch.randn(B, Lh, N, C)
        future = torch.randn(B, Lf, N, C)
        out = model(history_data=history, future_data=future, batch_seen=0, epoch=1, train=True)
        self.assertIn('prediction', out)
        self.assertEqual(list(out['prediction'].shape), [B, Lf, N, 1])

    def test_arbitrary_horizon(self):
        B, Lh, N, C = 1, 48, 4, 1
        model = HFPPE(enc_in=N, seq_len=Lh, pred_len=24, d_model=32, n_heads=4, e_layers=1, d_ff=64, m=2, ctx_len=16)
        history = torch.randn(B, Lh, N, C)
        for Lf in [1, 7, 24, 37]:
            future = torch.zeros(B, Lf, N, C)
            out = model(history_data=history, future_data=future, batch_seen=0, epoch=1, train=False)
            self.assertEqual(list(out['prediction'].shape), [B, Lf, N, 1])

    def test_regularizers_present_and_finite(self):
        B, Lh, Lf, N, C = 2, 24, 16, 3, 1
        model = HFPPE(enc_in=N, seq_len=Lh, pred_len=Lf, d_model=32, n_heads=4, e_layers=1, d_ff=64, m=2, ctx_len=8)
        history = torch.randn(B, Lh, N, C)
        future = torch.randn(B, Lf, N, C)
        out = model(history_data=history, future_data=future, batch_seen=0, epoch=1, train=True)
        self.assertTrue(torch.isfinite(out['reg_unit']))
        self.assertTrue(torch.isfinite(out['reg_phase']))
        self.assertTrue(torch.isfinite(out['reg_cycle']))

    def test_loss_composition(self):
        B, Lh, Lf, N, C = 2, 24, 16, 3, 1
        model = HFPPE(enc_in=N, seq_len=Lh, pred_len=Lf, d_model=32, n_heads=4, e_layers=1, d_ff=64, m=2, ctx_len=8)
        history = torch.randn(B, Lh, N, C)
        future = torch.randn(B, Lf, N, C)
        out = model(history_data=history, future_data=future, batch_seen=0, epoch=1, train=True)
        # fake target
        target = torch.randn(B, Lf, N, 1)
        loss = hf_ppe_loss(
            prediction=out['prediction'],
            target=target,
            null_val=float('nan'),
            reg_unit=out['reg_unit'],
            reg_phase=out['reg_phase'],
            reg_cycle=out['reg_cycle'],
            lambda_unit=1e-3,
            lambda_phase=1e-3,
            lambda_cycle=1e-3,
        )
        self.assertTrue(loss.item() >= 0.0)


if __name__ == '__main__':
    unittest.main()

