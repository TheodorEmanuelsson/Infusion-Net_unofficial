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

        cbackbone = self.cfg['backbone']
        cneck = self.cfg['neck']
        chead = self.cfg['head']
        closs = self.cfg['loss']
        cbackbone['input_channels'] = self.input_features

        self.backbone = eval(cbackbone['name'])(cbackbone)
        self.neck = eval(cneck['name'])(cneck)
        self.head = eval(chead['name'])(chead)
        self.loss = eval(closs['name'])(closs)

    def forward(self, x, targets=None):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        if targets is not None:
            return self.loss(x, targets)

        return x
def eelan(cfg):
    backbone = EELAN(cfg['depths'], cfg['channels'], cfg['input_channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone

def yolov7neck(cfg):
    neck = YOLOv7NECK(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck

def implicit_head(cfg):
    head = ImplicitHead(cfg['num_class'], cfg['num_anchor'], cfg['channels'])

def yolov7(cfg):
    head = YOLOv7Loss(cfg['num_class'], cfg['stride'], cfg['anchors'])
    return head



        