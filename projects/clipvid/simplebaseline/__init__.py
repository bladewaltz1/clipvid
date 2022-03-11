# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_simplebaseline_config
from .detector import SimpleBaseline
from .dataset_mapper import SimpleBaselineDatasetMapper
from .dataset import register_vid_instances
from .evaluator import Saver
