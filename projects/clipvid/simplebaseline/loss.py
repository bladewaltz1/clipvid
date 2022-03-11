# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from scipy.optimize import linear_sum_assignment

from detectron2.utils.comm import get_world_size

from .box_ops import generalized_box_iou


class SetCriterion:
    def __init__(self, cfg):
        self.matcher = HungarianMatcher(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.focal_loss_alpha = cfg.MODEL.SimpleBaseline.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SimpleBaseline.GAMMA
        self.focal_weight = cfg.MODEL.SimpleBaseline.FOCAL_WEIGHT
        self.l1_weight = cfg.MODEL.SimpleBaseline.L1_WEIGHT
        self.giou_weight = cfg.MODEL.SimpleBaseline.GIOU_WEIGHT

    def loss_labels(self, preds, targets, indices, num_boxes):
        assert "pred_logits" in preds
        logits = preds["pred_logits"]
        num_queries = logits.shape[1]
        logits = logits.flatten(0, 1)
        labels = torch.zeros_like(logits)

        batch_idx, query_idx = self._get_pred_permutation_idx(indices)
        flattened_idx = batch_idx * num_queries + query_idx
        permutated_label_idx = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )

        labels[flattened_idx, permutated_label_idx] = 1

        loss_focal = sigmoid_focal_loss_jit(
            logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        )
        loss_focal = loss_focal / num_boxes * self.focal_weight
        losses = {"loss_focal": loss_focal}

        return losses

    def loss_boxes(self, preds, targets, indices, num_boxes):
        assert "pred_boxes" in preds
        idx = self._get_pred_permutation_idx(indices)
        pred_boxes = preds["pred_boxes"][idx]
        tgt_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)]
        )

        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou(pred_boxes, tgt_boxes))
        losses["loss_giou"] = loss_giou.sum() / num_boxes * self.giou_weight

        normalizer = torch.cat(
            [v["image_size"].repeat(len(v["boxes"]), 1) for v in targets]
        )
        pred_boxes = pred_boxes / normalizer
        tgt_boxes = tgt_boxes / normalizer
        loss_l1 = F.l1_loss(pred_boxes, tgt_boxes, reduction="none")
        losses["loss_l1"] = loss_l1.sum() / num_boxes * self.l1_weight

        return losses

    def _get_pred_permutation_idx(self, indices):
        query_idx = torch.cat([idx for (idx, _) in indices])
        batch_idx = torch.cat(
            [torch.full_like(idx, i) for i, (idx, _) in enumerate(indices)]
        )
        return batch_idx, query_idx

    def __call__(self, preds, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([float(num_boxes)], device=self.device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for i, pred in enumerate(preds[::-1]):
            indices = self.matcher(pred, targets)
            loss = self.loss_labels(pred, targets, indices, num_boxes)
            loss.update(self.loss_boxes(pred, targets, indices, num_boxes))
            loss = {f"{k}_{i}": v for k, v in loss.items()}
            losses.update(loss)
        return losses


class HungarianMatcher:
    def __init__(self, cfg):
        self.focal_weight = cfg.MODEL.SimpleBaseline.FOCAL_WEIGHT
        self.l1_weight = cfg.MODEL.SimpleBaseline.L1_WEIGHT
        self.giou_weight = cfg.MODEL.SimpleBaseline.GIOU_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.SimpleBaseline.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SimpleBaseline.GAMMA

    @torch.no_grad()
    def __call__(self, preds, targets):
        """
        Args:
            preds:
              - pred_logits: [batch_size, num_queries, num_classes]
              - pred_boxes: [batch_size, num_queries, 4]
            targets: a list of (length = batch_size)
              - labels: [num_target_boxes]
              - boxes: [num_target_boxes, 4]
              - image_size: [1, 4]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
              - index_i: the indices of the selected predictions (in order)
              - index_j: the indices of the corresponding targets (in order)
        """
        bs, num_queries = preds["pred_logits"].shape[:2]

        pred_prob = preds["pred_logits"].flatten(0, 1).sigmoid()
        pred_boxes = preds["pred_boxes"].flatten(0, 1)

        tgt_labels = torch.cat([v["labels"] for v in targets])
        tgt_boxes = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. 
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        neg_cost = (1 - alpha) * (pred_prob ** gamma) * \
            (-(1 - pred_prob + 1e-8).log())
        pos_cost = alpha * ((1 - pred_prob) ** gamma) * \
            (-(pred_prob + 1e-8).log())
        cost_focal = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(pred_boxes, tgt_boxes)

        # Compute the L1 cost between boxes
        normalizer = torch.cat([v["image_size"] for v in targets])
        normalizer = normalizer.unsqueeze(1).repeat(1, num_queries, 1)
        pred_boxes = pred_boxes / normalizer.flatten(0, 1)
        normalizer = torch.cat(
            [v["image_size"].repeat(len(v["boxes"]), 1) for v in targets]
        )
        tgt_boxes = tgt_boxes / normalizer
        cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # Final cost matrix
        C = self.l1_weight * cost_l1 + self.focal_weight * cost_focal \
            + self.giou_weight * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
