import torch
import torch.nn as nn
from tools.utils import load_config
from yolov7_utils import YOLOv7Loss, ImplicitHead, YOLOv7NECK, EELAN

class YOLOv7(nn.Module):
    def __init__(self, num_classes:int, num_features:int, cfg_path:str):
        super().__init__()

        self.num_classes = num_classes
        self.input_features = num_features

        self.cfg = load_config(cfg_path)

        self.backbone = EELAN(self.cfg['backbone']['depths'], self.cfg['backbone']['channels'], self.input_features, self.cfg['backbone']['outputs'], self.cfg['backbone']['norm'], self.cfg['backbone']['act'])
        self.neck = YOLOv7NECK(self.cfg['neck']['depths'], self.cfg['neck']['channels'], self.cfg['neck']['norm'], self.cfg['neck']['act'])
        self.head = ImplicitHead(self.cfg['head']['num_class'], self.cfg['head']['num_anchor'], self.cfg['head']['channels'])
        self.loss = YOLOv7Loss(self.cfg['loss']['num_class'], self.cfg['loss']['stride'], self.cfg['loss']['anchors'])

    def forward(self, x, targets=None):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        if targets is not None:
            return self.loss(x, targets)

        return x


        