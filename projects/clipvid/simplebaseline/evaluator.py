import os

import torch
from collections import OrderedDict

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager


class Saver(DatasetEvaluator):
    def __init__(self, output_dir, name):
        self._name = name
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._predictions = []

    def process(self, batched_inputs, outputs):
        """
        Args:
            outputs: the outputs of the E2EVID model. 
                It is a list of dicts with key "instances" that contains
                :class:`Instances`
        """
        for output in outputs:
            instances = output["instances"].to(self._cpu_device)
            self._predictions.append({
                "data_index": output["data_index"],
                "scores": instances.scores,
                "labels": instances.pred_classes,
                "bbox": instances.pred_boxes.tensor,
            })

    def evaluate(self):
        return self._save()

    def _save(self):
        predictions = self._predictions
        if len(predictions) == 0:
            return OrderedDict()
        predictions = sorted(predictions, key=lambda x: x["data_index"])

        PathManager.mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, f"{self._name}.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(predictions, f, _use_new_zipfile_serialization=False)

        return OrderedDict()
