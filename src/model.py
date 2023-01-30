import torch
import torch.nn as nn
import tools.dct as dct_tools

class DCT2d(nn.Module):
    """ Discrete Cosine Transform 2D Layer with masking lower frequencies
    
    Arguments:
    ----------
        norm: str
            Normalization type for DCT
        tau: float
            Threshold for masking lower frequencies
    """
    def __init__(self, norm='infusion', tau=0.2, mask_freq:bool=True):
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.mask_freq = mask_freq

    def mask_image(self, tensor):
        """ Mask lower frequencies in DCT domain following Equation 5"""
        print(f'Input frequency domain shape: {tensor.shape}')
        x, y = torch.meshgrid(torch.arange(tensor.shape[-2]), torch.arange(tensor.shape[-1]))
        mask = (y >= -x + 2*self.tau*tensor.shape[-1]).float()
        return tensor * mask
    
    def forward(self, x):
        print(f'Input image shape: {x.shape}')
        dct_transform = nn.Parameter(dct_tools.dct_2d(x, norm=self.norm), requires_grad=False)

        if self.mask_freq:
            dct_transform = self.mask_image(dct_transform)

        return dct_transform

class IDCT2d(nn.Module):
    """ Inverse Discrete Cosine Transform 2D Layer

    Arguments:
    ----------
        norm: str
            Normalization type for DCT
    """
    def __init__(self, norm='infusion'):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        inv_dct_transform = dct_tools.idct_2d(x, norm=self.norm)
        return inv_dct_transform

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.channel_attention(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.rcab = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.rcab(x)

class HEBlock(nn.Module):
    def __init__(self, num_features, reduction, tau:float=0.2, norm:str='infusion'):
        super(HEBlock, self).__init__()

        self.tau = tau

        self.dct = DCT2d(norm=norm, tau=self.tau)
        self.idct = IDCT2d(norm=norm)

    def forward(self, x):
        x = self.dct(x)
        return self.idct(x)

class HFExtraction(nn.Module):
    def __init__(self, num_features:int, reduction:int, tau:float=0.2, norm:str='infusion'):
        super(HFExtraction, self).__init__()

        self.tau = tau

        self.hfe = HEBlock(num_features, reduction, tau=self.tau, norm=norm)
        self.rcab = RCAB(num_features, reduction)

    def forward(self, x):
        he = self.hfe(x)
        out = self.rcab(he)
        return x + out, he

class HFAssistant(nn.Module):
    def __init__(self, num_features, reduction, tau=0.2):
        super(HFAssistant, self).__init__()

        self.tau = tau
        self.rgb_stream = HFExtraction(num_features, reduction, tau=self.tau)
        self.ir_stream = HFExtraction(num_features, reduction, tau=self.tau)

    def forward(self, rgb, ir):
        rgb_residual_rcab, rbg_he = self.rgb_stream(rgb)
        ir_residual_rcab, ir_he = self.ir_stream(ir)

        return rgb_residual_rcab, rbg_he, ir_residual_rcab, ir_he

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
        
class WeightFusionParam(nn.Module):
    def __init__(self):
        super(WeightFusionParam, self).__init__()

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
        # Weightning parameters Phase 0
        self.rgb_alpha_0 = WeightFusionParam()
        self.ir_alpha_0 = WeightFusionParam()
        self.rgb_beta_0 = WeightFusionParam()
        self.ir_beta_0 = WeightFusionParam()

        # Initialize Phase 1
        self.rgb_phase_1 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_1 = InnerPhaseBlock(num_features, reduction)

        self.HFA_1 = HFAssistant(num_features, reduction, tau=self.tau)
        # Weightning parameters Phase 1
        self.rgb_alpha_1 = WeightFusionParam()
        self.ir_alpha_1 = WeightFusionParam()
        self.rgb_beta_1 = WeightFusionParam()
        self.ir_beta_1 = WeightFusionParam()
        # Output Weightning parameters Phase 1
        self.rgb_weight_1 = WeightFusionParam()
        self.ir_weight_1 = WeightFusionParam()

        # Initialize Phase 2
        self.rgb_phase_2 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_2 = InnerPhaseBlock(num_features, reduction)

        self.HFA_2 = HFAssistant(num_features, reduction, tau=self.tau)
        # Weightning parameters Phase 2
        self.rgb_alpha_2 = WeightFusionParam()
        self.ir_alpha_2 = WeightFusionParam()
        self.rgb_beta_2 = WeightFusionParam()
        self.ir_beta_2 = WeightFusionParam()
        # Output Weightning parameters Phase 2
        self.rgb_weight_2 = WeightFusionParam()
        self.ir_weight_2 = WeightFusionParam()

        # Initialize Phase 3
        self.rgb_phase_3 = InnerPhaseBlock(num_features, reduction)
        self.ir_phase_3 = InnerPhaseBlock(num_features, reduction)

        self.HFA_3 = HFAssistant(num_features, reduction)
        # Weightning parameters Phase 3
        self.rgb_alpha_3 = WeightFusionParam()
        self.ir_alpha_3 = WeightFusionParam()
        self.rgb_beta_3 = WeightFusionParam()
        self.ir_beta_3 = WeightFusionParam()
        # Output Weightning parameters Phase 3
        self.rgb_weight_3 = WeightFusionParam()
        self.ir_weight_3 = WeightFusionParam()

    def forward(self, x):
        if not len(x.shape) == 4:
            x = x.unsqueeze(0)

        rgb_input = x[:, :3, :, :]
        ir_input = x[:, 3:, :, :]

        print("RGB input shape: ", rgb_input.shape)
        print("IR input shape: ", ir_input.shape)

        ##### Phase 0
        rgb_phase_0 = self.rgb_phase_0(rgb_input)
        ir_phase_0 = self.ir_phase_0(ir_input)

        print("RGB phase 0 shape: ", rgb_phase_0.shape)
        print("IR phase 0 shape: ", ir_phase_0.shape)

        # Pass through HFA
        rgb_residual_rcab_0, rbg_he_0, ir_residual_rcab_0, ir_he_0 = self.HFA_0(rgb_phase_0, ir_phase_0)
        print("RGB residual RCAB 0 shape: ", rgb_residual_rcab_0.shape)
        print("RGB HE 0 shape: ", rbg_he_0.shape)
        print("IR residual RCAB 0 shape: ", ir_residual_rcab_0.shape)
        print("IR HE 0 shape: ", ir_he_0.shape)

        # Residual connections
        out_rgb_0 = rgb_phase_0 + self.rgb_alpha_0(rbg_he_0) + self.rgb_beta_0(ir_residual_rcab_0)
        out_ir_0 = ir_phase_0 + self.ir_alpha_0(ir_he_0) + self.ir_beta_0(rgb_residual_rcab_0)
        print("RGB out 0 shape: ", out_rgb_0.shape)
        print("IR out 0 shape: ", out_ir_0.shape)

        ##### Phase 1
        rgb_phase_1 = self.rgb_phase_1(out_rgb_0)
        ir_phase_1 = self.ir_phase_1(out_ir_0)
        print("RGB phase 1 shape: ", rgb_phase_1.shape)
        print("IR phase 1 shape: ", ir_phase_1.shape)

        # Pass through HFA
        rgb_residual_rcab_1, rbg_he_1, ir_residual_rcab_1, ir_he_1 = self.HFA_1(rgb_phase_1, ir_phase_1)
        print("RGB residual RCAB 1 shape: ", rgb_residual_rcab_1.shape)
        print("RGB HE 1 shape: ", rbg_he_1.shape)
        print("IR residual RCAB 1 shape: ", ir_residual_rcab_1.shape)
        print("IR HE 1 shape: ", ir_he_1.shape)

        # Residual connections
        out_rgb_1 = rgb_phase_1 + self.rgb_alpha_1(rbg_he_1) + self.rgb_beta_1(ir_residual_rcab_1)
        out_ir_1 = ir_phase_1 + self.ir_alpha_1(ir_he_1) + self.ir_beta_1(rgb_residual_rcab_1)
        print("RGB out 1 shape: ", out_rgb_1.shape)
        print("IR out 1 shape: ", out_ir_1.shape)

        # Output Weightning
        detection_map_1 = self.rgb_weight_1(out_rgb_1) + self.ir_weight_1(out_ir_1)

        ##### Phase 2
        rgb_phase_2 = self.rgb_phase_2(out_rgb_1)
        ir_phase_2 = self.ir_phase_2(out_ir_1)

        print("RGB phase 2 shape: ", rgb_phase_2.shape)
        print("IR phase 2 shape: ", ir_phase_2.shape)


        # Pass through HFA
        rgb_residual_rcab_2, rbg_he_2, ir_residual_rcab_2, ir_he_2 = self.HFA_2(rgb_phase_2, ir_phase_2)

        print("RGB residual RCAB 2 shape: ", rgb_residual_rcab_2.shape)
        print("RGB HE 2 shape: ", rbg_he_2.shape)
        print("IR residual RCAB 2 shape: ", ir_residual_rcab_2.shape)
        print("IR HE 2 shape: ", ir_he_2.shape)


        # Residual connections
        out_rgb_2 = rgb_phase_2 + self.rgb_alpha_2(rbg_he_2) + self.rgb_beta_2(ir_residual_rcab_2)
        out_ir_2 = ir_phase_2 + self.ir_alpha_2(ir_he_2) + self.ir_beta_2(rgb_residual_rcab_2)

        print("RGB out 2 shape: ", out_rgb_2.shape)
        print("IR out 2 shape: ", out_ir_2.shape)


        # Output Weightning
        detection_map_2 = self.rgb_weight_2(out_rgb_2) + self.ir_weight_2(out_ir_2)

        ##### Phase 3
        rgb_phase_3 = self.rgb_phase_3(out_rgb_2)
        ir_phase_3 = self.ir_phase_3(out_ir_2)

        print("RGB phase 3 shape: ", rgb_phase_3.shape)
        print("IR phase 3 shape: ", ir_phase_3.shape)


        # Pass through HFA
        rgb_residual_rcab_3, rbg_he_3, ir_residual_rcab_3, ir_he_3  = self.HFA_3(rgb_phase_3, ir_phase_3)

        print("RGB residual RCAB 3 shape: ", rgb_residual_rcab_3.shape)
        print("RGB HE 3 shape: ", rbg_he_3.shape)
        print("IR residual RCAB 3 shape: ", ir_residual_rcab_3.shape)
        print("IR HE 3 shape: ", ir_he_3.shape)


        # Residual connections
        out_rgb_3 = rgb_phase_3 + self.rgb_alpha_3(rbg_he_3) + self.rgb_beta_3(ir_residual_rcab_3)
        out_ir_3 = ir_phase_3 + self.ir_alpha_3(ir_he_3) + self.ir_beta_3(rgb_residual_rcab_3)

        print("RGB out 3 shape: ", out_rgb_3.shape)
        print("IR out 3 shape: ", out_ir_3.shape)

        # Output Weightning
        detection_map_3 = self.rgb_weight_3(out_rgb_3) + self.ir_weight_3(out_ir_3)

        print("Detection map 1 shape: ", detection_map_1.shape)
        print("Detection map 2 shape: ", detection_map_2.shape)
        print("Detection map 3 shape: ", detection_map_3.shape)

        ##### To Detection Map
        detection_map = torch.cat((detection_map_1, detection_map_2, detection_map_3), dim=1)

        print("Detection map output shape: ", detection_map.shape)

        return detection_map

if __name__ == '__main__':
    print('Testing the model')



        



