"""
Module for custom loss functions.
"""

import torch
import torch.nn as nn


class MemAELoss(nn.Module):
    """MSE reconstruction + entropy regularization on attention weights."""

    def __init__(self, entropy_weight: float = 0.0002):
        super().__init__()
        self.entropy_weight = entropy_weight   # alpha
        self.mse = nn.MSELoss()

    def forward(self, recon, target, attn):
        """
        Args:
            recon:  (B, T, C, H, W) reconstruction
            target: (B, T, C, H, W) input
            attn:   (B, n_queries, N) attention weights
        Returns:
            total_loss, (recon_loss, entropy_loss)  # ayrı logla
        """
        eps = 1e-12

        # Reconstruction
        recon_loss = self.mse(recon, target)   # TODO: doğru mu, M1'le aynı

        # Entropy: per-query entropy, then mean
        # E = mean( -sum_i ( w_i * log(w_i + eps) ) )
        entropy = (-(attn * torch.log(attn + eps)).sum(dim=-1)).mean()

        # Sum
        total = recon_loss + self.entropy_weight * entropy

        return total, (recon_loss, entropy)
    

if __name__ == "__main__":
    # Smoke test
    loss_fn = MemAELoss()
    recon = torch.randn(2, 16, 1, 128, 128)
    target = torch.randn(2, 16, 1, 128, 128)
    attn = torch.softmax(torch.randn(2, 1024, 2000), dim=-1)   # valid distribution
    total, (rl, ent) = loss_fn(recon, target, attn)
    print(f"total: {total.item():.4f}, recon: {rl.item():.4f}, entropy: {ent.item():.4f}")