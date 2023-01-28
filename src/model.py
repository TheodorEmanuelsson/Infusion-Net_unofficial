import torch
import torch.nn as nn
import tools.dct as dct

class DCT2d(nn.Module):
    def __init__(self, norm='infusion', tau=0.2):
        super().__init__()
        self.norm = norm
        self.tau = tau

    def mask_image(self, tensor):
        x, y = torch.meshgrid(torch.arange(tensor.shape[1]), torch.arange(tensor.shape[2]))
        mask = (y >= -x + 2*self.tau*tensor.shape[2]).float()
        return tensor * mask
    
    def forward(self, x):
        dct_transform = dct.dct_2d(x, norm=self.norm)
        return self.mask_image(dct_transform)

class IDCT2d(nn.Module):
    def __init__(self, norm='infusion'):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        return dct.idct_2d(x, norm=self.norm)

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.module(x)

class HFExtraction(nn.Module):
    def __init__(self, num_features, reduction):
        super(HFExtraction, self).__init__()

        self.module = nn.Sequential(
            DCT2d(),
            IDCT2d(),
            RCAB(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)

class HFAssistant(nn.Module):
    def __init__(self, num_features, reduction):
        super(HFAssistant, self).__init__()

        self.rgb_stream = HFExtraction(num_features, reduction)
        self.ir_stream = HFExtraction(num_features, reduction)

    def forward(self, rgb, ir):
        rgb_out = self.rgb_stream(rgb)
        ir_out = self.ir_stream(ir)

        return rgb + ir_out, ir + rgb_out

class InputPhaseBlock(nn.Module):
    def __init__(self, num_features, reduction):
        super(InputPhaseBlock, self).__init__()

        self.four_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.base_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features*4, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.four_conv(x)

        x1 = self.base_conv(x)

        x2 = self.base_conv(x1)
        x2 = self.base_conv(x2)

        x3 = self.base_conv(x2)
        x3 = self.base_conv(x3)

        merged = torch.cat([x, x1, x2, x3], dim=1)

        return self.final_conv(merged)
        


class InnerPhaseBlock(nn.Module):
    def __init__(self, num_features, reduction):
        super(InnerPhaseBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.base_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features*4, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x1 = self.max_pool(x)
        x1 = self.base_conv(x1)
        x2 = self.double_conv(x)

        x = torch.cat([x1, x2], dim=1)

        x1 = self.base_conv(x)

        x2 = self.base_conv(x)

        x3 = self.double_conv(x2)

        x4 = self.double_conv(x3)

        merged = torch.cat([x1, x2, x3, x4], dim=1)

        return self.final_conv(merged)
        



