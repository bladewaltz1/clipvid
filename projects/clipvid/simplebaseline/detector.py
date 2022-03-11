# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    detector_postprocess
)
from detectron2.structures import Boxes, ImageList, Instances

from .decoder import Decoder
from .loss import SetCriterion
from .memory import Memory


@META_ARCH_REGISTRY.register()
class SimpleBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        num_queries = cfg.MODEL.SimpleBaseline.NUM_QUERIES
        hidden_dim = cfg.MODEL.SimpleBaseline.HIDDEN_DIM

        self.backbone = build_backbone(cfg)
        self.queries = nn.Embedding(num_queries, hidden_dim)
        self.decoder = Decoder(cfg, input_shape=self.backbone.output_shape())

        self.criterion = SetCriterion(cfg=cfg)

        mu = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        sigma = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - mu) / sigma

        self.memory = Memory(
            cfg.MODEL.SimpleBaseline.MEMORY_SIZE,
            cfg.MODEL.SimpleBaseline.CHUNK_SIZE,
        )

    def forward(self, batched_inputs):
        assert len(batched_inputs) == 1
        batched_inputs = batched_inputs[0]
        imgs, img_box = self.preprocess_image(batched_inputs)

        src = self.backbone(imgs.tensor)
        features = []
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        if self.training:
            return self.forward_train(batched_inputs, features, img_box)
        else:
            return self.forward_inference(batched_inputs, features, img_box)

    def forward_train(self, batched_inputs, features, img_box):
        init_boxes = img_box[:, None, :]
        output = self.decoder(features, init_boxes, self.queries.weight)
        targets = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(targets)
        loss_dict = self.criterion(output, targets)
        return loss_dict

    def prepare_targets(self, targets):
        new_targets = []
        for target in targets:
            h, w = target.image_size
            image_size = torch.as_tensor([w, h, w, h], dtype=torch.float)
            new_targets.append({
                "image_size": image_size.unsqueeze(0).to(self.device),
                "labels": target.gt_classes.to(self.device),
                "boxes": target.gt_boxes.tensor.to(self.device)
            })
        return new_targets

    def forward_inference(self, batched_inputs, features, img_box):
        win_size = self.memory.window_size
        capacity = self.memory.capacity
        self.memory.insert(batched_inputs, features, img_box)

        if self.memory.has_all_frames():
            args = self.memory.pack_args()
            results = self.inference_window(*args)
        elif self.memory.is_full():
            args = self.memory.pack_args()
            if self.memory.has_first_frame():
                results = self.inference_window(*args, end=capacity - win_size)
            elif self.memory.has_last_frame():
                results = self.inference_window(*args, start=win_size)
            else:
                results = self.inference_window(*args, start=win_size,
                                                end=capacity - win_size)
        else:
            results = []

        if self.memory.has_last_frame():
            self.memory.reset()

        return results

    def inference_window(self, batched_inputs, features, img_box,
                         start=None, end=None):
        init_boxes = img_box[:, None, :]
        output = self.decoder(features, init_boxes, self.queries.weight)
        pred_logits = output[-1]["pred_logits"]
        pred_boxes = output[-1]["pred_boxes"]

        if start is not None and end is not None:
            pred_logits = pred_logits[start:end]
            pred_boxes = pred_boxes[start:end]
            batched_inputs = batched_inputs[start:end]
        elif start is not None:
            pred_logits = pred_logits[start:]
            pred_boxes = pred_boxes[start:]
            batched_inputs = batched_inputs[start:]
        elif end is not None:
            pred_logits = pred_logits[:end]
            pred_boxes = pred_boxes[:end]
            batched_inputs = batched_inputs[:end]

        image_size = (int(img_box[0, 3]), int(img_box[0, 2]))
        results = self.inference(pred_logits, pred_boxes, image_size)

        height = batched_inputs[0]["height"]
        width = batched_inputs[0]["width"]
        processed_results = []
        for results_per_image, input in zip(results, batched_inputs):
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(
                {"instances": r, "data_index": input["data_index"]}
            )

        return processed_results

    def inference(self, pred_logits, pred_boxes, image_size):
        results = []
        scores = torch.sigmoid(pred_logits)
        _, num_queries, num_classes = scores.shape
        labels = torch.arange(num_classes, device=self.device)
        labels = labels.unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        for score, box in zip(scores, pred_boxes):
            result = Instances(image_size)
            score, topk_indices = score.flatten(0, 1).topk(100)
            result.scores = score

            labels_per_image = labels[topk_indices]
            result.pred_classes = labels_per_image

            box = box.view(-1, 1, 4).repeat(1, num_classes, 1)
            box = box.view(-1, 4)[topk_indices]
            result.pred_boxes = Boxes(box)
            results.append(result)

        return results

    def preprocess_image(self, inputs):
        """Normalize, pad and batch the input images.
        """
        imgs = [self.normalizer(x["image"].to(self.device)) for x in inputs]
        imgs = ImageList.from_tensors(imgs, self.backbone.size_divisibility)

        img_box = []
        for item in inputs:
            h, w = item["image"].shape[-2:]
            img_box.append(torch.tensor([0, 0, w, h], dtype=torch.float32))
        img_box = torch.stack(img_box).to(self.device)

        return imgs, img_box
