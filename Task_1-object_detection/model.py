import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_ch = num_anchors * (5 + num_classes)
        self.conv1 = ConvBlock(in_ch, 256)
        self.conv2 = nn.Conv2d(256, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SimpleDetector(nn.Module):
    def __init__(self, num_classes=5, img_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = img_size // 16
        self.num_anchors = 3

        self.anchors = torch.tensor([
            [10, 10],
            [25, 25],
            [50, 50]
        ], dtype=torch.float32)

        self.backbone = Backbone()
        self.head = DetectionHead(256, num_classes, self.num_anchors)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)

        batch_size = x.size(0)
        grid_h, grid_w = output.size(2), output.size(3)

        output = output.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        return output

    def decode_predictions(self, output, conf_thresh=0.3, nms_thresh=0.4):
        batch_size = output.size(0)
        grid_h, grid_w = output.size(2), output.size(3)

        device = output.device
        anchors = self.anchors.to(device)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        all_boxes = []
        all_scores = []
        all_labels = []

        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []

            for a in range(self.num_anchors):
                pred = output[b, a]

                tx = torch.sigmoid(pred[:, :, 0])
                ty = torch.sigmoid(pred[:, :, 1])
                tw = pred[:, :, 2]
                th = pred[:, :, 3]
                conf = torch.sigmoid(pred[:, :, 4])
                cls_probs = torch.softmax(pred[:, :, 5:], dim=-1)

                cx = (grid_x + tx) * (self.img_size / grid_w)
                cy = (grid_y + ty) * (self.img_size / grid_h)
                w = torch.exp(tw) * anchors[a, 0]
                h = torch.exp(th) * anchors[a, 1]

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                cls_scores, cls_idx = cls_probs.max(dim=-1)
                final_scores = conf * cls_scores

                mask = final_scores > conf_thresh

                if mask.sum() > 0:
                    b_x1 = x1[mask]
                    b_y1 = y1[mask]
                    b_x2 = x2[mask]
                    b_y2 = y2[mask]
                    b_scores = final_scores[mask]
                    b_labels = cls_idx[mask]

                    box_tensor = torch.stack([b_x1, b_y1, b_x2, b_y2], dim=1)
                    boxes.append(box_tensor)
                    scores.append(b_scores)
                    labels.append(b_labels)

            if len(boxes) > 0:
                boxes = torch.cat(boxes, dim=0)
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)

                keep = self.nms(boxes, scores, nms_thresh)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def nms(self, boxes, scores, thresh):
        if boxes.size(0) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            mask = iou <= thresh
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=5, img_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_anchors = 3
        self.anchors = torch.tensor([
            [10, 10],
            [25, 25],
            [50, 50]
        ], dtype=torch.float32)
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        device = predictions.device
        batch_size = predictions.size(0)
        grid_h, grid_w = predictions.size(2), predictions.size(3)

        anchors = self.anchors.to(device)

        obj_mask = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        noobj_mask = torch.ones(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        tx = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        ty = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        tw = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        th = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, device=device)
        tcls = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, dtype=torch.long, device=device)

        for b in range(batch_size):
            boxes = targets[b]['boxes']
            labels = targets[b]['labels']

            if boxes.size(0) == 0:
                continue

            for i in range(boxes.size(0)):
                x1, y1, x2, y2 = boxes[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                gi = int(cx / self.img_size * grid_w)
                gj = int(cy / self.img_size * grid_h)
                gi = min(gi, grid_w - 1)
                gj = min(gj, grid_h - 1)

                ious = []
                for a in range(self.num_anchors):
                    anchor_w, anchor_h = anchors[a]
                    inter_w = min(w, anchor_w)
                    inter_h = min(h, anchor_h)
                    inter = inter_w * inter_h
                    union = w * h + anchor_w * anchor_h - inter
                    ious.append(inter / union)

                best_a = int(torch.tensor(ious).argmax())

                obj_mask[b, best_a, gj, gi] = 1
                noobj_mask[b, best_a, gj, gi] = 0

                tx[b, best_a, gj, gi] = cx / self.img_size * grid_w - gi
                ty[b, best_a, gj, gi] = cy / self.img_size * grid_h - gj
                tw[b, best_a, gj, gi] = torch.log(w / anchors[best_a, 0] + 1e-16)
                th[b, best_a, gj, gi] = torch.log(h / anchors[best_a, 1] + 1e-16)
                tcls[b, best_a, gj, gi] = labels[i]

        pred_tx = torch.sigmoid(predictions[:, :, :, :, 0])
        pred_ty = torch.sigmoid(predictions[:, :, :, :, 1])
        pred_tw = predictions[:, :, :, :, 2]
        pred_th = predictions[:, :, :, :, 3]
        pred_conf = predictions[:, :, :, :, 4]
        pred_cls = predictions[:, :, :, :, 5:]

        loss_x = self.lambda_coord * self.mse(pred_tx * obj_mask, tx * obj_mask)
        loss_y = self.lambda_coord * self.mse(pred_ty * obj_mask, ty * obj_mask)
        loss_w = self.lambda_coord * self.mse(pred_tw * obj_mask, tw * obj_mask)
        loss_h = self.lambda_coord * self.mse(pred_th * obj_mask, th * obj_mask)

        loss_conf_obj = self.bce(pred_conf * obj_mask, obj_mask)
        loss_conf_noobj = self.lambda_noobj * self.bce(pred_conf * noobj_mask, torch.zeros_like(pred_conf) * noobj_mask)

        obj_indices = obj_mask.bool()
        if obj_indices.sum() > 0:
            pred_cls_flat = pred_cls[obj_indices]
            tcls_flat = tcls[obj_indices]
            loss_cls = self.ce(pred_cls_flat, tcls_flat)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj + loss_cls

        num_obj = obj_mask.sum() + 1e-16
        total_loss = total_loss / num_obj

        return total_loss
