"""
Module for autoencoder model.
"""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Auto encoder model class.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Encoder layers
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(),

            # Bottleneck
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.LeakyReLU(),

            # Decoder layers
            nn.ConvTranspose3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.LeakyReLU(),

            # Output layer
            nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        # Permute to match the shape that dataloader gives
        x = x.permute(0, 2, 1, 3, 4)   # (B,T,C,H,W) -> (B,C,T,H,W)
        x = self.network(x)
        x = x.permute(0, 2, 1, 3, 4)   # backwards: (B,C,T,H,W) -> (B,T,C,H,W)
        return x
    

if __name__ == "__main__":
    # Smoke test to assert that shapes are correctly matches
    model = AutoEncoder()
    x = torch.randn(2, 16, 1, 128, 128)

    # debug: seperate variable to permute manually
    xd = x.permute(0,2,1,3,4)
    for layer in model.network:
        xd = layer(xd)
        if isinstance(xd, torch.Tensor):
            print(type(layer).__name__, tuple(xd.shape))

    # real forward prop
    out = model(x)
    print("out:", tuple(out.shape))
    assert out.shape == x.shape