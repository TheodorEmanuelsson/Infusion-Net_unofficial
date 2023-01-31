import sys
import torch
from torch import nn

sys.path.append("..")
from yolov7_builder.core.detection_heads import Yolov7DetectionHead
from yolov7_builder.core.layers import (
    MP,
    SPPCSPC,
    Concat,
    Conv,
    RepConv,
)


def get_yolov7_config(num_classes=80, anchor_sizes_per_layer=None, num_channels=3):
    if anchor_sizes_per_layer is None:
        anchor_sizes_per_layer = torch.tensor(
            [
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]],
            ]
        )
    # `strides` are defined by architecture, explicit for clarity
    strides = torch.tensor([8.0, 16.0, 32.0])
    detection_head = Yolov7DetectionHead

    return {
        "state_dict_path": "https://github.com/Chris-hughes10/Yolov7-training/releases/download/0.1.0/yolov7_training_state_dict.pt",
        "num_classes": num_classes,
        "aux_detection": False,
        "num_channels": num_channels,
        "image_size": (640, 640),
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "anchor_sizes_per_layer": anchor_sizes_per_layer,
        "backbone": [
            [-1, 1, Conv, [32, 3, 1]],
            [-1, 1, Conv, [64, 3, 2]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [-1, 1, Conv, [64, 1, 1]],
            [-2, 1, Conv, [64, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [512, 1, 1]],
            [-3, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],
        ],
        "head": [
            [-1, 1, SPPCSPC, [512]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [37, 1, Conv, [256, 1, 1]],
            [[-1, -2], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [24, 1, Conv, [128, 1, 1]],
            [[-1, -2], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3, 63], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3, 51], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [-2, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [75, 1, RepConv, [256, 3, 1]],
            [88, 1, RepConv, [512, 3, 1]],
            [101, 1, RepConv, [1024, 3, 1]],
            [
                [102, 103, 104],
                1,
                detection_head,
                [num_classes, anchor_sizes_per_layer, strides],
            ],
        ],
    }