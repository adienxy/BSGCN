import math

import torch.fft
import torch
import torch.nn as nn


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=11, w=6): # keep h, w equals x.shape dim 1, 2
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) # [11, 6, 768]
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape # [16, 121, 768] -> [batch_size, max_sequence_length, bert_dim]
        # print(x.shape)
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C) # [16, 11, 11, 768]
        # print(x.shape)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') # [16, 11, 6, 768]
        # print(x.shape)
        weight = torch.view_as_complex(self.complex_weight) # [11, 6 ,768]
        # print(weight.shape)
        x = x * weight # [16, 11, 6, 768] * [11, 6 ,768]
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x
