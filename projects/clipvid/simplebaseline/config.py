# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_simplebaseline_config(cfg):
    cfg.MODEL.SimpleBaseline = CN()
    cfg.MODEL.SimpleBaseline.NUM_CLASSES = 30
    cfg.MODEL.SimpleBaseline.NUM_QUERIES = 72
    cfg.MODEL.SimpleBaseline.BOX_JIT = True
    cfg.MODEL.SimpleBaseline.BOX_JIT_RATIO = 0.05

    cfg.MODEL.SimpleBaseline.APPLY_SELSA_AT = (0,1,4,5)
    cfg.MODEL.SimpleBaseline.NUM_TRAIN_FRAMES = (3,)
    cfg.MODEL.SimpleBaseline.CHUNK_SIZE = 30
    cfg.MODEL.SimpleBaseline.MEMORY_SIZE = 1

    # Decoder
    cfg.MODEL.SimpleBaseline.NUM_HEADS = 8
    cfg.MODEL.SimpleBaseline.DROPOUT = 0.0
    cfg.MODEL.SimpleBaseline.FEEDFORWARD_DIM = 2048
    cfg.MODEL.SimpleBaseline.HIDDEN_DIM = 384
    cfg.MODEL.SimpleBaseline.NUM_CLS = 1
    cfg.MODEL.SimpleBaseline.NUM_REG = 3
    cfg.MODEL.SimpleBaseline.NUM_LAYERS = 6

    cfg.MODEL.SimpleBaseline.FOCAL_WEIGHT = 2.0
    cfg.MODEL.SimpleBaseline.ALPHA = 0.25
    cfg.MODEL.SimpleBaseline.GAMMA = 2.0
    cfg.MODEL.SimpleBaseline.PRIOR_PROB = 0.01
    cfg.MODEL.SimpleBaseline.GIOU_WEIGHT = 2.0
    cfg.MODEL.SimpleBaseline.L1_WEIGHT = 5.0

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
