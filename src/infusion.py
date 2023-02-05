import yaml
import torch
import torch.nn as nn
import tools.dct as dct_tools
from yaml.loader import SafeLoader

class DCT2d(nn.Module):
    """ Discrete Cosine Transform 2D Layer with masking lower frequencies
    
    Arguments:
    ----------
        norm: str
            Normalization type for DCT
        tau: float
            Threshold for masking lower frequencies
    """
    def __init__(self, norm='ortho', tau=0.2, mask_freq:bool=True):
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.mask_freq = mask_freq

    def mask_image(self, tensor):
        """ Mask lower frequencies in DCT domain following Equation 5"""
        x_range = torch.arange(tensor.shape[-1], device=tensor.device)
        y_range = torch.arange(tensor.shape[-2], device=tensor.device)
        x, y = torch.meshgrid(x_range, y_range)
        mask = (y >= -x + 2*self.tau*tensor.shape[-1]).float()
        return tensor * torch.transpose(mask, 0, 1)
    
    def forward(self, x):
        dct_transform = dct_tools.dct_2d(x, norm=self.norm)

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
    def __init__(self, norm='ortho'):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        inv_dct_transform = dct_tools.idct_2d(x, norm=self.norm)
        return inv_dct_transform

class ChannelAttention(nn.Module):
    """ Channel Attention Module taken from https://github.com/yjn870/RCAN-pytorch 
    
    Arguments:
    ----------
        num_features: int
            Number of features in the input tensor
        reduction: int
            Reduction factor for the channel attention module
    """


    def __init__(self, num_features=1, reduction=1):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features , kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.channel_attention(x)
        return x + attention


class RCAB(nn.Module):
    """ Residual Channel Attention Block taken from https://github.com/yjn870/RCAN-pytorch 
    
    Arguments:
    ----------
        num_features: int
            Number of features in the input tensor
        reduction: int
            Reduction factor for the channel attention module
    """
    def __init__(self, num_features=1, reduction=1):
        super(RCAB, self).__init__()
        self.rcab = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.rcab(x)

class HEBlock(nn.Module):
    """ HE Block for extracting high frequency features 
    
    Arguments:
    ----------
        tau: float
            Threshold for masking lower frequencies
        norm: str
            Normalization type for DCT
        mask_freq: bool
            Whether to mask lower frequencies
    """
    def __init__(self, tau:float=0.2, norm:str='ortho', mask_freq:bool=True):
        super(HEBlock, self).__init__()

        self.tau = tau
        self.dct = DCT2d(norm=norm, tau=self.tau, mask_freq=mask_freq)
        self.idct = IDCT2d(norm=norm)

    def forward(self, x):
        x = self.dct(x)
        return self.idct(x)

class HFExtraction(nn.Module):
    """ HF Extraction Module for extracting high frequency features with RCAB block 
    
    Arguments:
    ----------
        num_features: int
            Number of features in the input tensor
        reduction: int
            Reduction factor for the channel attention module
        tau: float
            Threshold for masking lower frequencies
        norm: str
            Normalization type for DCT
    """
    def __init__(self, num_features:int=1, reduction:int=1, tau:float=0.2, norm:str='ortho'):
        super(HFExtraction, self).__init__()

        self.tau = tau

        self.hfe = HEBlock(tau=self.tau, norm=norm)
        self.rcab = RCAB(num_features, reduction)

    def forward(self, x):
        he = self.hfe(x)
        out = self.rcab(he)
        return x + out, he

class HFAssistant(nn.Module):
    """ HF Assistant Module for extracting high frequency features with RCAB block for both RGB and IR streams 
    
    Arguments:
    ----------
        num_features: int
            Number of features in the input tensor
        reduction: int
            Reduction factor for the channel attention module
        tau: float
            Threshold for masking lower frequencies
        norm: str
            Normalization type for DCT
    """
    def __init__(self, num_features=1, reduction=1, tau=0.2, norm='ortho'):
        super(HFAssistant, self).__init__()

        self.tau = tau
        self.norm = norm
        self.rgb_stream = HFExtraction(num_features, reduction, tau=self.tau)
        self.ir_stream = HFExtraction(num_features, reduction, tau=self.tau)

    def forward(self, rgb, ir):
        rgb_residual_rcab, rbg_he = self.rgb_stream(rgb)
        ir_residual_rcab, ir_he = self.ir_stream(ir)

        return rgb_residual_rcab, rbg_he, ir_residual_rcab, ir_he

class InputPhaseBlock(nn.Module):
    """Phase 0 block for the input of the network

    Arguments:
    ----------
        num_features: int
            Number of features after the first convolution
    """
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
            nn.Conv2d(num_features*4, 1, kernel_size=3, padding=1),
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
    """Phase 1, 2, 3 blocks.
    
    Arguments:
    ----------
        num_features: int
            Number of input features
    """
    def __init__(self, num_features):
        super(InnerPhaseBlock, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.input_double_conv = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.base_double_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features*4, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x1 = self.max_pool(x)
        x1 = self.input_conv(x1)
        x2 = self.input_double_conv(x)
        
        x = torch.cat([x1, x2], dim=1)

        x1 = self.reduce_conv(x)

        x2 = self.reduce_conv(x)

        x3 = self.base_double_conv(x2)

        x4 = self.base_double_conv(x3)

        x_out = torch.cat([x1, x2, x3, x4], dim=1)

        return self.final_conv(x_out)
        
class WeightFusionParam(nn.Module):
    """ Weighting parameters for the fusion of the two inputs """
    def __init__(self):
        super(WeightFusionParam, self).__init__()

        self.weight_fusion = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return self.weight_fusion * x


class InfusionNet(nn.Module):
    """ InfusionNet model """
    def __init__(self, num_features, reduction, tau):
        super(InfusionNet, self).__init__()

        self.tau = tau

        # Initialize Phase 0
        self.rgb_phase_0 = InputPhaseBlock(num_features)
        self.ir_phase_0 = InputPhaseBlock(num_features)

        self.HFA_0 = HFAssistant(tau=self.tau)
        # Weightning parameters Phase 0
        self.rgb_alpha_0 = WeightFusionParam()
        self.ir_alpha_0 = WeightFusionParam()
        self.rgb_beta_0 = WeightFusionParam()
        self.ir_beta_0 = WeightFusionParam()

        # Initialize Phase 1
        self.rgb_phase_1 = InnerPhaseBlock(num_features)
        self.ir_phase_1 = InnerPhaseBlock(num_features)

        self.HFA_1 = HFAssistant(tau=self.tau)
        # Weightning parameters Phase 1
        self.rgb_alpha_1 = WeightFusionParam()
        self.ir_alpha_1 = WeightFusionParam()
        self.rgb_beta_1 = WeightFusionParam()
        self.ir_beta_1 = WeightFusionParam()
        # Output Weightning parameters Phase 1
        self.rgb_weight_1 = WeightFusionParam()
        self.ir_weight_1 = WeightFusionParam()

        # Initialize Phase 2
        self.rgb_phase_2 = InnerPhaseBlock(num_features)
        self.ir_phase_2 = InnerPhaseBlock(num_features)

        self.HFA_2 = HFAssistant(tau=self.tau)
        # Weightning parameters Phase 2
        self.rgb_alpha_2 = WeightFusionParam()
        self.ir_alpha_2 = WeightFusionParam()
        self.rgb_beta_2 = WeightFusionParam()
        self.ir_beta_2 = WeightFusionParam()
        # Output Weightning parameters Phase 2
        self.rgb_weight_2 = WeightFusionParam()
        self.ir_weight_2 = WeightFusionParam()

        # Initialize Phase 3
        self.rgb_phase_3 = InnerPhaseBlock(num_features)
        self.ir_phase_3 = InnerPhaseBlock(num_features)

        self.HFA_3 = HFAssistant()
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

        ##### Phase 0
        rgb_phase_0 = self.rgb_phase_0(rgb_input)
        ir_phase_0 = self.ir_phase_0(ir_input)

        # Pass through HFA
        rgb_residual_rcab_0, rbg_he_0, ir_residual_rcab_0, ir_he_0 = self.HFA_0(rgb_phase_0, ir_phase_0)

        # Residual connections
        out_rgb_0 = rgb_phase_0 + self.rgb_alpha_0(rbg_he_0) + self.rgb_beta_0(ir_residual_rcab_0)
        out_ir_0 = ir_phase_0 + self.ir_alpha_0(ir_he_0) + self.ir_beta_0(rgb_residual_rcab_0)

        ##### Phase 1
        rgb_phase_1 = self.rgb_phase_1(out_rgb_0)
        ir_phase_1 = self.ir_phase_1(out_ir_0)

        # Pass through HFA
        rgb_residual_rcab_1, rbg_he_1, ir_residual_rcab_1, ir_he_1 = self.HFA_1(rgb_phase_1, ir_phase_1)

        # Residual connections
        out_rgb_1 = rgb_phase_1 + self.rgb_alpha_1(rbg_he_1) + self.rgb_beta_1(ir_residual_rcab_1)
        out_ir_1 = ir_phase_1 + self.ir_alpha_1(ir_he_1) + self.ir_beta_1(rgb_residual_rcab_1)

        # Output Weightning
        detection_map_1 = self.rgb_weight_1(out_rgb_1) + self.ir_weight_1(out_ir_1)

        ##### Phase 2
        rgb_phase_2 = self.rgb_phase_2(out_rgb_1)
        ir_phase_2 = self.ir_phase_2(out_ir_1)

        # Pass through HFA
        rgb_residual_rcab_2, rbg_he_2, ir_residual_rcab_2, ir_he_2 = self.HFA_2(rgb_phase_2, ir_phase_2)

        # Residual connections
        out_rgb_2 = rgb_phase_2 + self.rgb_alpha_2(rbg_he_2) + self.rgb_beta_2(ir_residual_rcab_2)
        out_ir_2 = ir_phase_2 + self.ir_alpha_2(ir_he_2) + self.ir_beta_2(rgb_residual_rcab_2)

        # Output Weightning
        detection_map_2 = self.rgb_weight_2(out_rgb_2) + self.ir_weight_2(out_ir_2)

        ##### Phase 3
        rgb_phase_3 = self.rgb_phase_3(out_rgb_2)
        ir_phase_3 = self.ir_phase_3(out_ir_2)
        # Pass through HFA
        rgb_residual_rcab_3, rbg_he_3, ir_residual_rcab_3, ir_he_3  = self.HFA_3(rgb_phase_3, ir_phase_3)

        # Residual connections
        out_rgb_3 = rgb_phase_3 + self.rgb_alpha_3(rbg_he_3) + self.rgb_beta_3(ir_residual_rcab_3)
        out_ir_3 = ir_phase_3 + self.ir_alpha_3(ir_he_3) + self.ir_beta_3(rgb_residual_rcab_3)

        # Output Weightning
        detection_map_3 = self.rgb_weight_3(out_rgb_3) + self.ir_weight_3(out_ir_3)

        ##### To Detection Map
        detection_map = torch.cat((detection_map_1, detection_map_2, detection_map_3), dim=1)

        return detection_map

class InfusionDetection(nn.Module):
    def __init__(self, infusion_model, detection_model):
        super().__init__()

        self.infusion_model = infusion_model
        self.model = detection_model

        #print(detection_model.hyp)
        self.hyp = detection_model.hyp

    def forward(self, x):
        x = self.infusion_model(x)
        x = self.model(x)

        return x



if __name__ == '__main__':
    print('Testing the model')



        



