"""
Module for UNet based predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPredictor(nn.Module):
    """
    U net based predictor model class.
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.LeakyReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.LeakyReLU()
        )
        
        # Decoder blocks
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.LeakyReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU()
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU()
        )
        
        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x: torch.Tensor):
        # x: (B, 15, 1, H, W) -> squeeze/reshape -> (B, 15, H, W)
        x = x.squeeze(2)                    # # (B,15,1,H,W) -> (B,15,H,W)
        s1 = self.enc1(x)                   # (B,32,128,128) <- skip1
        s2 = self.enc2(self.pool(s1))       # (B,64,64,64)   <- skip2
        s3 = self.enc3(self.pool(s2))       # (B,128,32,32)  <- skip3
        b  = self.bottleneck(self.pool(s3)) # (B,256,16,16)

        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))   # cat→384 -> 128, (B,128,32,32)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))   # cat→192 -> 64,  (B,64,64,64)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))   # cat→96  -> 32,  (B,32,128,128)

        return self.out(d1)     # (B,1,128,128)
    

if __name__ == "__main__":
    model = UNetPredictor()
    x = torch.randn(2, 15, 1, 128, 128)
    out = model(x)
    print(out.shape)   # expected: (2, 1, 128, 128)