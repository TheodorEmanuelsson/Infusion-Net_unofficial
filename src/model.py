import torch
import torch.nn as nn
import tools.dct as dct

class DCT2d(nn.Module):
    def __init__(self, norm='infusion', tau=0.2):
        super().__init__()
        self.norm = norm
        self.tau = tau

    def mask_image(self, tensor):
        print(tensor.shape)
        x, y = torch.meshgrid(torch.arange(tensor.shape[2]), torch.arange(tensor.shape[3]))
        mask = (y >= -x + 2*self.tau*tensor.shape[3]).float()
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
    def __init__(self, num_features, reduction, tau=0.2):
        super(HFExtraction, self).__init__()

        self.tau = tau

        self.module = nn.Sequential(
            DCT2d(tau=self.tau),
            IDCT2d(),
            RCAB(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)

class HFAssistant(nn.Module):
    def __init__(self, num_features, reduction, tau=0.2):
        super(HFAssistant, self).__init__()

        self.tau = tau
        self.rgb_stream = HFExtraction(num_features, reduction, tau=self.tau)
        self.ir_stream = HFExtraction(num_features, reduction, tau=self.tau)

    def forward(self, rgb, ir):
        rgb_steam_out = self.rgb_stream(rgb)
        ir_stream_out = self.ir_stream(ir)

        return rgb_steam_out, ir_stream_out

class InputPhaseBlock(nn.Module):
    def __init__(self, num_features):
        super(InputPhaseBlock, self).__init__()

        self.four_conv = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
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
        
class WeightFusion(nn.Module):
    def __init__(self):
        super(WeightFusion, self).__init__()

        self.weight_fusion = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return self.weight_fusion * x


class InfusionNet(nn.Module):
    def __init__(self, num_features, reduction, tau):
        super(InfusionNet, self).__init__()

        self.tau = tau

        # Initialize Phase 0
        self.rgb_phase_0 = InputPhaseBlock(num_features)
        self.ir_phase_0 = InputPhaseBlock(num_features)

        self.HFA_0 = HFAssistant(num_features, reduction, tau=self.tau)
        # Weightning parameters Phase
        self.rgb_alpha_0 = WeightFusion()
        self.ir_alpha_0 = WeightFusion()
        self.rgb_beta_0 = WeightFusion()
        self.ir_beta_0 = WeightFusion()

        # Initialize Phase 1
        self.rgb_phase_1 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_1 = InnerPhaseBlock(num_features, reduction)

        self.HFA_1 = HFAssistant(num_features, reduction, tau=self.tau)
        # Weightning parameters Phase 1
        self.rgb_alpha_1 = WeightFusion()
        self.ir_alpha_1 = WeightFusion()
        self.rgb_beta_1 = WeightFusion()
        self.ir_beta_1 = WeightFusion()
        # Output Weightning parameters Phase 1
        self.rgb_weight_1 = WeightFusion()
        self.ir_weight_1 = WeightFusion()

        # Initialize Phase 2
        self.rgb_phase_2 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_2 = InnerPhaseBlock(num_features, reduction)

        self.HFA_2 = HFAssistant(num_features, reduction, tau=self.tau)
        # Weightning parameters Phase 2
        self.rgb_alpha_2 = WeightFusion()
        self.ir_alpha_2 = WeightFusion()
        self.rgb_beta_2 = WeightFusion()
        self.ir_beta_2 = WeightFusion()
        # Output Weightning parameters Phase 2
        self.rgb_weight_2 = WeightFusion()
        self.ir_weight_2 = WeightFusion()

        # Initialize Phase 3
        self.rgb_phase_3 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_3 = InnerPhaseBlock(num_features, reduction)

        self.HFA_3 = HFAssistant(num_features, reduction)
        # Weightning parameters Phase 3
        self.rgb_alpha_3 = WeightFusion()
        self.ir_alpha_3 = WeightFusion()
        self.rgb_beta_3 = WeightFusion()
        self.ir_beta_3 = WeightFusion()
        # Output Weightning parameters Phase 3
        self.rgb_weight_3 = WeightFusion()
        self.ir_weight_3 = WeightFusion()

    def forward(self, x):
        rgb_input = x[:, :3, :, :]
        ir_input = x[:, 3:, :, :]

        ##### Phase 0
        rgb_phase_0 = self.rgb_phase_0(rgb_input)
        ir_phase_0 = self.ir_phase_0(ir_input)

        # Pass through HFA
        rgb_HFA_0, ir_HFA_0 = self.HFA_0(rgb_phase_0, ir_phase_0)

        # Residual connections
        out_rgb_0 = rgb_phase_0 + self.rgb_alpha_0(rgb_HFA_0) + self.rgb_beta_0(ir_HFA_0)
        out_ir_0 = ir_phase_0 + self.ir_alpha_0(ir_HFA_0) + self.ir_beta_0(rgb_HFA_0)

        ##### Phase 1
        rgb_phase_1 = self.rgb_phase_1(out_rgb_0)
        ir_phase_1 = self.ir_phase_1(out_ir_0)

        # Pass through HFA
        rgb_HFA_1, ir_HFA_1 = self.HFA_1(rgb_phase_1, ir_phase_1)

        # Residual connections
        out_rgb_1 = rgb_phase_1 + self.rgb_alpha_1(rgb_HFA_1) + self.rgb_beta_1(ir_HFA_1)
        out_ir_1 = ir_phase_1 + self.ir_alpha_1(ir_HFA_1) + self.ir_beta_1(rgb_HFA_1)

        # Output Weightning
        detection_map_1 = self.rgb_weight_1(out_rgb_1) + self.ir_weight_1(out_ir_1)

        ##### Phase 2
        rgb_phase_2 = self.rgb_phase_2(out_rgb_1)
        ir_phase_2 = self.ir_phase_2(out_ir_1)

        # Pass through HFA
        rgb_HFA_2, ir_HFA_2 = self.HFA_2(rgb_phase_2, ir_phase_2)

        # Residual connections
        out_rgb_2 = rgb_phase_2 + self.rgb_alpha_2(rgb_HFA_2) + self.rgb_beta_2(ir_HFA_2)
        out_ir_2 = ir_phase_2 + self.ir_alpha_2(ir_HFA_2) + self.ir_beta_2(rgb_HFA_2)

        # Output Weightning
        detection_map_2 = self.rgb_weight_2(out_rgb_2) + self.ir_weight_2(out_ir_2)

        ##### Phase 3
        rgb_phase_3 = self.rgb_phase_3(out_rgb_2)
        ir_phase_3 = self.ir_phase_3(out_ir_2)

        # Pass through HFA
        rgb_HFA_3, ir_HFA_3 = self.HFA_3(rgb_phase_3, ir_phase_3)

        # Residual connections
        out_rgb_3 = rgb_phase_3 + self.rgb_alpha_3(rgb_HFA_3) + self.rgb_beta_3(ir_HFA_3)
        out_ir_3 = ir_phase_3 + self.ir_alpha_3(ir_HFA_3) + self.ir_beta_3(rgb_HFA_3)

        # Output Weightning
        detection_map_3 = self.rgb_weight_3(out_rgb_3) + self.ir_weight_3(out_ir_3)

        ##### To Detection Map
        detection_map = torch.cat((detection_map_1, detection_map_2, detection_map_3), dim=1)

        return detection_map



if __name__ == '__main__':
    print('Testing the model')



        



