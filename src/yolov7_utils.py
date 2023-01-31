# Defining the YOLOv7 loss function
# Adapted from https://github.com/Iywie/pl_YOLO/blob/master/models/losses/yolov7/yolov7_loss.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

class YOLOv7Loss(nn.Module):
    def __init__(self,
                 num_classes,
                 strides,
                 anchors,
                 label_smoothing=0,
                 focal_g=0.0,):
        super(YOLOv7Loss, self).__init__()

        self.anchors = torch.tensor(anchors)
        self.num_classes = num_classes
        self.strides = strides

        self.nl = len(strides)
        self.na = len(anchors[0])
        self.ch = 5 + self.num_classes

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.threshold = 4.0

        self.grids = [torch.zeros(1)] * len(strides)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, -1, 1, 1, 2)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)
        self.BCEcls, self.BCEobj, self.gr = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def __call__(self, inputs, targets):
        # input of inputs: [batch, ch * anchor, h, w]

        batch_size = targets.shape[0]
        # input: [batch, anchor, h, w, ch]
        for i in range(self.nl):
            prediction = inputs[i].view(
                inputs[i].size(0), self.na, self.ch, inputs[i].size(2), inputs[i].size(3)
            ).permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction

        # inference
        if not self.training:
            preds = []
            for i in range(self.nl):
                pred = inputs[i].sigmoid()
                h, w = pred.shape[2:4]
                # Three steps to localize predictions: grid, shifts of x and y, grid with stride
                if self.grids[i].shape[2:4] != pred.shape[2:4]:
                    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
                    grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                    self.grids[i] = grid
                else:
                    grid = self.grids[i]

                pred[..., :2] = (pred[..., :2] * 2. - 0.5 + grid) * self.strides[i]
                pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * self.anchor_grid[i].type_as(pred)
                pred = pred.reshape(batch_size, -1, self.ch)
                preds.append(pred)

            # preds: [batch_size, all predictions, n_ch]
            predictions = torch.cat(preds, 1)
            # from (cx,cy,w,h) to (x1,y1,x2,y2)
            box_corner = predictions.new(predictions.shape)
            box_corner = box_corner[:, :, 0:4]
            box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
            box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
            box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
            box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
            predictions[:, :, :4] = box_corner[:, :, :4]
            return predictions

        # Compute loss
        # Processing ground truth to tensor (img_idx, class, cx, cy, w, h)
        n_gt = (targets.sum(dim=2) > 0).sum(dim=1)

        gts_list = []
        for img_idx in range(batch_size):
            nt = n_gt[img_idx]
            gt_boxes = targets[img_idx, :nt, 1:5]
            gt_classes = targets[img_idx, :nt, 0].unsqueeze(-1)
            gt_img_ids = torch.ones_like(gt_classes).type_as(gt_classes) * img_idx
            gt = torch.cat((gt_img_ids, gt_classes, gt_boxes), 1)
            gts_list.append(gt)
        targets = torch.cat(gts_list, 0)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(inputs, targets)

        cls_loss = torch.zeros(1).type_as(inputs[0])
        box_loss = torch.zeros(1).type_as(inputs[0])
        obj_loss = torch.zeros(1).type_as(inputs[0])

        for i, prediction in enumerate(inputs):
            #   image, anchor, gridy, gridx
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = torch.zeros_like(prediction[..., 0]).type_as(prediction)  # target obj

            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                grid = torch.stack([gi, gj], dim=1)

                #   进行解码，获得预测结果
                xy = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box = torch.cat((xy, wh), 1)

                #   对真实框进行处理，映射到特征层上
                selected_tbox = targets[i][:, 2:6] / self.strides[i]
                selected_tbox[:, :2] = selected_tbox[:, :2] - grid.type_as(prediction)

                #   计算预测框和真实框的回归损失
                iou = bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss += (1.0 - iou).mean()
                # -------------------------------------------#
                #   根据预测结果的iou获得置信度损失的gt
                # -------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # -------------------------------------------#
                #   计算匹配上的正样本的分类损失
                # -------------------------------------------#
                selected_tcls = targets[i][:, 1].long()
                t = torch.full_like(prediction_pos[:, 5:], self.cn).type_as(prediction)  # targets
                t[range(n), selected_tcls] = self.cp
                cls_loss += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

            # -------------------------------------------#
            #   计算目标是否存在的置信度损失
            #   并且乘上每个特征层的比例
            # -------------------------------------------#
            obj_loss += self.BCEobj(prediction[..., 4], tobj) * self.balance[i]  # obj loss

        # -------------------------------------------#
        #   将各个部分的损失乘上比例
        #   全加起来后，乘上batch_size
        # -------------------------------------------#
        box_loss *= self.box_ratio
        obj_loss *= self.obj_ratio
        cls_loss *= self.cls_ratio

        loss = box_loss + obj_loss + cls_loss

        losses = {"loss": loss}
        return losses

    def build_targets(self, predictions, targets):

        # indice: [img_idx, anchor_idx, grid_x, grid_y]
        indices, anch = self.find_3_positive(predictions, targets)

        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]

        # label assignment for each image
        for batch_idx in range(predictions[0].shape[0]):

            # targets of this image
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, map in enumerate(predictions):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = map[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.strides[i]  # / 8.
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.strides[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            # Cost matrix
            pair_wise_iou = bboxes_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            # Dynamic k
            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(self.nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(self.nl):
            if matching_targets[i]:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([]).type_as(targets)
                matching_as[i] = torch.tensor([]).type_as(targets)
                matching_gjs[i] = torch.tensor([]).type_as(targets)
                matching_gis[i] = torch.tensor([]).type_as(targets)
                matching_targets[i] = torch.tensor([]).type_as(targets)
                matching_anchs[i] = torch.tensor([]).type_as(targets)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        """
        Args:
            predictions(tensor): [nb, na, w, h, ch]
            targets(tensor): [image_idx, class, x, y, w, h]
        Return:
            indice: [img_idx, anchor_idx, grid_x, grid_y]
            anchor: [anchor_w, anchor_h]
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7).type_as(targets).long()  # normalized to gridspace gain
        ai = torch.arange(na).type_as(targets).view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):

            # put anchor and target to feature map
            anchors = (self.anchors[i] / self.strides[i]).type_as(predictions[i])
            gain[2:6] = self.strides[i]
            target = targets / gain
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # w and h

            # Match targets to anchors
            if nt:
                # target and anchor wh ratio in threshold
                r = target[:, :, 4:6] / anchors[:, None]  # wh ratio
                wh_mask = torch.max(r, 1. / r).max(2)[0] < self.threshold  # compare
                t = target[wh_mask]

                # Positive adjacent grid
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse grid xy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image_idx, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class ImplicitHead(nn.Module):
    def __init__(
            self,
            num_classes,
            num_anchors,
            in_channels,
    ):
        super().__init__()
        self.n_anchors = num_anchors
        self.num_classes = num_classes
        ch = self.n_anchors * (5 + num_classes)

        self.conv = nn.ModuleList()
        self.ia = nn.ModuleList()
        self.im = nn.ModuleList()
        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.ia.append(ImplicitA(in_channels[i]))
            self.conv.append(
                nn.Conv2d(in_channels[i], ch, 1)
            )
            self.im.append(ImplicitM(ch))

    def forward(self, inputs):
        outputs = []
        for k, (ia, head_conv, im, x) in enumerate(zip(self.ia, self.conv, self.im, inputs)):
            # x: [batch_size, n_ch, h, w]
            x = ia(x)
            x = head_conv(x)
            x = im(x)
            outputs.append(x)
        return outputs

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class YOLOv7NECK(nn.Module):
    """
    Only proceed 3 layer input. Like stage2, stage3, stage4.
    """

    def __init__(
            self,
            depths=(1, 1, 1, 1),
            in_channels=(512, 1024, 1024),
            norm='bn',
            act="silu",
    ):
        super().__init__()

        # top-down conv
        self.spp = SPPCSPC(in_channels[2], in_channels[2] // 2, k=(5, 9, 13))
        self.conv_for_P5 = BaseConv(in_channels[2] // 2, in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_C4 = BaseConv(in_channels[1], in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.p5_p4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.conv_for_P4 = BaseConv(in_channels[2] // 4, in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.conv_for_C3 = BaseConv(in_channels[0], in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.p4_p3 = CSPLayer(
            in_channels[2] // 4,
            in_channels[2] // 8,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        # bottom-up conv
        self.downsample_conv1 = Transition(in_channels[2] // 8, in_channels[2] // 4, mpk=2, norm=norm, act=act)
        self.n3_n4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.downsample_conv2 = Transition(in_channels[2] // 4, in_channels[2] // 2, mpk=2, norm=norm, act=act)
        self.n4_n5 = CSPLayer(
            in_channels[2],
            in_channels[2] // 2,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.n3 = BaseConv(in_channels[2] // 8, in_channels[2] // 4, 3, 1, norm=norm, act=act)
        self.n4 = BaseConv(in_channels[2] // 4, in_channels[2] // 2, 3, 1, norm=norm, act=act)
        self.n5 = BaseConv(in_channels[2] // 2, in_channels[2], 3, 1, norm=norm, act=act)

    def forward(self, inputs):
        #  backbone
        [c3, c4, c5] = inputs
        # top-down
        p5 = self.spp(c5)
        p5_shrink = self.conv_for_P5(p5)
        p5_upsample = self.upsample(p5_shrink)
        p4 = torch.cat([p5_upsample, self.conv_for_C4(c4)], 1)
        p4 = self.p5_p4(p4)

        p4_shrink = self.conv_for_P4(p4)
        p4_upsample = self.upsample(p4_shrink)
        p3 = torch.cat([p4_upsample, self.conv_for_C3(c3)], 1)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)
        n5 = torch.cat([n4_downsample, p5], 1)
        n5 = self.n4_n5(n5)

        n3 = self.n3(n3)
        n4 = self.n4(n4)
        n5 = self.n5(n5)

        outputs = (n3, n4, n5)
        return outputs

def get_normalization(name, out_channels):
    if name is None:
        return None
    if name == "bn":
        module = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
    elif name == "ln":
        module = nn.LayerNorm(out_channels)
    else:
        raise AttributeError("Unsupported normalization function type: {}".format(name))
    return module

def get_activation(name="silu", inplace=True):
    if name is None:
        return None
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hswish':
        module = HSwish()
    elif name == "gelu":
        module = nn.GELU()
    else:
        raise AttributeError("Unsupported activation function type: {}".format(name))
    return module

class HSwish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class BaseConv(nn.Module):
    """A Convolution2d -> Normalization -> Activation"""
    def __init__(
        self, in_channels, out_channels, ksize, stride, padding=None, groups=1, bias=False, norm="bn", act="silu"
    ):
        super().__init__()
        # same padding
        if padding is None:
            pad = (ksize - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.norm = get_normalization(norm, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if self.norm is None and self.act is None:
            return self.conv(x)
        elif self.act is None:
            return self.norm(self.conv(x))
        elif self.norm is None:
            return self.act(self.conv(x))
        return self.act(self.norm(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        self.cv1 = BaseConv(c1, c2, 1, 1)
        self.cv2 = BaseConv(c1, c2, 1, 1)
        self.cv3 = BaseConv(c2, c2, 3, 1)
        self.cv4 = BaseConv(c2, c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = BaseConv(4 * c2, c2, 1, 1)
        self.cv6 = BaseConv(c2, c2, 3, 1)
        self.cv7 = BaseConv(2 * c2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottle=1,
        shortcut=True,
        expansion=0.5,
        norm='bn',
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            num_bottle (int): number of Bottlenecks. Default value: 1.
            shortcut (bool): residual operation.
            expansion (int): the number that hidden channels compared with output channels.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, norm=norm, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, norm=norm, act=act)
            for _ in range(num_bottle)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class Bottleneck(nn.Module):
    # Standard bottleneck from ResNet
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        norm='bn',
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.bn = get_normalization(norm, out_channels)
        self.act = get_activation(act, inplace=True)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, norm=norm, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class Transition(nn.Module):
    def __init__(self, in_channel, out_channel, mpk=2, norm='bn', act="silu"):
        super(Transition, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=mpk, stride=mpk)
        self.conv1 = BaseConv(in_channel, out_channel//2, 1, 1)
        self.conv2 = BaseConv(in_channel, out_channel//2, 1, 1)
        self.conv3 = BaseConv(out_channel//2, out_channel//2, 3, 2, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.conv1(x_1)

        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)

        return torch.cat([x_2, x_1], 1)

class EELAN(nn.Module):
    """
    Extended efficient layer aggregation networks (EELAN)
    """
    def __init__(
        self,
        depths=(4, 4, 4, 4),
        channels=(64, 128, 256, 512, 1024),
        out_features=("stage2", "stage3", "stage4"),
        norm='bn',
        act="silu",
    ):
        super().__init__()

        # parameters of the network
        assert out_features, "please provide output features of EELAN!"
        self.out_features = out_features
        print('Initializing EELAN backbone...')
        # stem
        print('Initializing stem...')
        self.stem = nn.Sequential(
            BaseConv(3, 32, 3, 1, norm=norm, act=act),
            BaseConv(32, channels[0], 3, 2, norm=norm, act=act),
            BaseConv(channels[0], channels[0], 3, 1, norm=norm, act=act),
        )
        print('Initializing stage1...')
        # stage1
        self.stage1 = nn.Sequential(
            BaseConv(channels[0], channels[1], 3, 2, norm=norm, act=act),
            CSPLayer(channels[1], channels[2], expansion=0.5, num_bottle=depths[0], norm=norm, act=act),
        )
        print('Initializing stage2...')
        # stage2
        self.stage2 = nn.Sequential(
            Transition(channels[2], mpk=2, norm=norm, act=act),
            CSPLayer(channels[2], channels[3], expansion=0.5, num_bottle=depths[1], norm=norm, act=act),
        )
        print('Initializing stage3...')
        # stage3
        self.stage3 = nn.Sequential(
            Transition(channels[3], mpk=2, norm=norm, act=act),
            CSPLayer(channels[3], channels[4], expansion=0.5, num_bottle=depths[2], norm=norm, act=act),
        )
        print('Initializing stage4...')
        # stage4
        self.stage4 = nn.Sequential(
            Transition(channels[4], mpk=2, norm=norm, act=act),
            SPPBottleneck(channels[4], channels[4], norm=norm, act=act),
            CSPLayer(channels[4], channels[4], expansion=0.5, num_bottle=depths[3], norm=norm, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.stage1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        if len(self.out_features) <= 1:
            return x
        return [v for k, v in outputs.items() if k in self.out_features]

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), norm='bn', act="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x