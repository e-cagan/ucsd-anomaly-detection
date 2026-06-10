"""
Memory-Augmented Autoencoder (MemAE) for video anomaly detection.

Encoder/decoder backbone = M1 AutoEncoder, unchanged. A memory module is
inserted between them: the decoder can only reconstruct from stored normal
prototypes, so anomalies (absent from memory) reconstruct poorly.

Ref: Gong et al. 2019, "Memorizing Normality to Detect Anomaly" (ICCV).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    """
    Memory bank with sparse attention-based addressing.

    Forward: bottleneck feature -> queries -> address memory -> reconstructed
    feature (+ attention weights for entropy loss / sparsity inspection).

    Args:
        n_slots: N, number of memory items (paper value)
        feat_dim: C, dimension of each memory item = bottleneck channel dim (16)
        shrink_thres: lambda, sparse addressing threshold (paper value)
    """

    def __init__(self, n_slots: int, feat_dim: int = 16, shrink_thres: float = None):
        super().__init__()
        self.n_slots = n_slots
        self.feat_dim = feat_dim

        # Edge case check
        if shrink_thres is None:
            shrink_thres = 1.0 / n_slots   # lambda, dependant on N
        self.shrink_thres = shrink_thres

        # Memory bank: learnable (N, C) matrix, trained end-to-end via backprop
        self.memory = nn.Parameter(torch.randn(size=(self.n_slots, self.feat_dim)))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: bottleneck feature, shape (B, C, T, H, W) = (B, 16, 4, 16, 16)
        Returns:
            z_hat: reconstructed feature, same shape as z
            attn:  attention weights, shape (B, n_queries, N) -- for loss/viz
        """
        B, C, T, H, W = z.shape
        n_queries = T * H * W   # 4*16*16 = 1024

        # (B,C,T,H,W) -> queries (B, n_queries, C)
        z = z.permute(dims=(0, 2, 3, 4, 1))
        query = z.reshape(shape=(B, n_queries, C))   # shape (B, n_queries, C)

        # Cosine similarity: query vs her memory slot.
        query_n  = F.normalize(query, dim=-1)        # (B, n_queries, C)
        memory_n = F.normalize(self.memory, dim=-1)  # (N, C)
        sim = query_n @ memory_n.t()                 # (B, n_queries, N)

        # Softmax over N
        attn = F.softmax(sim, dim=-1)   # -1 dim to autocalculate dimensions, shape (B, n_queries, N)

        # Sparse addressing: hard shrinkage + renormalize
        eps = 1e-12
        # hard shrinkage
        attn = F.relu(attn - self.shrink_thres) * attn / (torch.abs(attn - self.shrink_thres) + eps)
        # renormalize
        attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)

        # Weighted sum: with plain (unnormalized) memory
        z_hat_flat = attn @ self.memory   # (B,n_queries,N) @ (N,C) = (B,n_queries,C)

        # queries -> (B,C,T,H,W) backwards
        # Backwards of the first step: reshape -> (B,T,H,W,C), after permute -> (B,C,T,H,W)
        z_hat = z_hat_flat.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)      # (B,C,T,H,W)

        return z_hat, attn


class MemoryAE(nn.Module):
    """
    M1 encoder + MemoryModule + M1 decoder.
    Encoder/decoder backbone unchanged from M1 (clean ablation).
    """

    def __init__(self, n_slots: int, shrink_thres: float = None):
        super().__init__()

        # Edge case check
        if shrink_thres is None:
            shrink_thres = 1.0 / n_slots   # lambda, dependant on N

        # Encoder layers
        self.encoder = nn.Sequential(
            # enc1
            nn.Conv3d(1, 16, (3,3,3), stride=(1,2,2), padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(),
            # enc2
            nn.Conv3d(16, 32, (3,3,3), stride=(2,2,2), padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            # enc3
            nn.Conv3d(32, 64, (3,3,3), stride=(2,2,2), padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            # bottleneck — encoder's last piece
            nn.Conv3d(64, 16, (3,3,3), stride=(1,1,1), padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(),
        )
        # Memory layer
        self.memory = MemoryModule(n_slots=n_slots, feat_dim=16, shrink_thres=shrink_thres)
        # Decoder layers
        self.decoder = nn.Sequential(
            # dec1
            nn.ConvTranspose3d(16, 32, (3,3,3), stride=(2,2,2), padding=1, output_padding=(1,1,1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            # dec2
            nn.ConvTranspose3d(32, 16, (3,3,3), stride=(2,2,2), padding=1, output_padding=(1,1,1)),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(),
            # dec3
            nn.ConvTranspose3d(16, 1, (3,3,3), stride=(1,2,2), padding=1, output_padding=(0,1,1)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W) -- loader format (same with M1)
        Returns:
            recon: (B, T, C, H, W)
            attn:  (B, n_queries, N)
        """
        # M1 permute logic: loader (B,T,C,H,W) -> conv (B,C,T,H,W)
        x = x.permute(0, 2, 1, 3, 4)   # (B,C,T,H,W)

        # encoder -> bottleneck
        z = self.encoder(x)                    # (B, 16, 4, 16, 16)

        # memory addressing
        z_hat, attn = self.memory(z)           # (B, 16, 4, 16, 16), (B, 1024, N)

        # decoder
        recon = self.decoder(z_hat)            # (B, C, T, H, W)

        # permute backwards to loader format
        recon = recon.permute(0, 2, 1, 3, 4)   # (B,T,C,H,W)

        return recon, attn
    

if __name__ == "__main__":
    # Smoke test
    model = MemoryAE(n_slots=2000)   # paper N
    x = torch.randn(2, 16, 1, 128, 128)

    # Control piece by piece
    xp = x.permute(0,2,1,3,4)                           # (2,16,1,128,128) -> (2,1,16,128,128)
    z = model.encoder(xp)
    print("bottleneck:", z.shape)                       # should be (2, 16, 4, 16, 16)

    z_hat, attn = model.memory(z)
    print("z_hat:", z_hat.shape, "attn:", attn.shape)   # (2,16,4,16,16), (2,1024,2000)

    recon, attn = model(x)
    active_frac = (attn > 0).float().mean()
    active_per_query = (attn > 0).float().sum(dim=-1).mean()
    print(f"shrink_thres (lambda): {model.memory.shrink_thres}")
    print(f"active slot fraction: {active_frac:.4f}")
    print(f"avg active slots/query: {active_per_query:.1f} / {model.memory.n_slots}")