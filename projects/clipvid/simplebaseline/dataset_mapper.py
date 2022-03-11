# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import random

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class SimpleBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SimpleBaseline.

    The callable currently does the following:

    1. Read the image from "filename"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    def __init__(self, cfg, is_train=True):
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.num_train_frames = cfg.MODEL.SimpleBaseline.NUM_TRAIN_FRAMES

    def __call__(self, records):
        """
        Args:
            records (list or dict): Metadata of one video.
        Returns:
            list of dicts
        """
        dataset_dicts = []
        if self.is_train:
            num_train_frames = random.choice(self.num_train_frames)
            if isinstance(records[0], list):
                chunks = random.sample(records, num_train_frames)
                for chunk in chunks:
                    sample_id = random.randint(0, len(chunk) - 1)
                    dataset_dicts.append(copy.deepcopy(chunk[sample_id]))
            elif isinstance(records[0], dict):
                for _ in range(num_train_frames):
                    dataset_dicts.append(copy.deepcopy(records[0]))
        else:
            dataset_dicts = copy.deepcopy(records)

        for data in dataset_dicts:
            image = utils.read_image(data["filename"], self.img_format)
            utils.check_image_size(data, image)

            image, tfms = T.apply_transform_gens(self.tfm_gens, image)
            img_shape = image.shape[:2]  # h, w

            data["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )

            if self.is_train:
                assert "annotations" in data
                for anno in data["annotations"]:
                    anno.pop("segmentation", None)
                    anno.pop("keypoints", None)

                annos = [
                    utils.transform_instance_annotations(obj, tfms, img_shape)
                    for obj in data.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, img_shape)
                data["instances"] = utils.filter_empty_instances(instances)

        return dataset_dicts
