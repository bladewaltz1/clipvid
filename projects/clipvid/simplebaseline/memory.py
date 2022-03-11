import itertools

import torch


class Memory:
    def __init__(self, memory_size, chunk_size):
        assert memory_size % 2 == 1
        self.memory_size = memory_size
        self.window_size = (memory_size - 1) // 2 * chunk_size
        self.capacity = memory_size * chunk_size
        self.reset()

    def insert(self, batched_inputs, features, img_box):
        """
        Args:
            batched_inputs: list of dicts, list length = N
            features: list of tensors, list length = 1, tensor shape (N,C,H,W)
            img_box: tensor of shape (N, 4)
        """
        if "first_frame" in batched_inputs[0].keys():
            self._has_first_frame = True
        if "last_frame" in batched_inputs[-1].keys():
            self._has_last_frame = True

        if self.is_full():
            self._memory["batched_inputs"].pop(0)
            self._memory["features"].pop(0)
            self._memory["img_box"].pop(0)
            self._has_first_frame = False

        self._memory["batched_inputs"].append(batched_inputs)
        self._memory["features"].append(features)
        self._memory["img_box"].append(img_box)

    def has_first_frame(self):
        return self._has_first_frame

    def has_last_frame(self):
        return self._has_last_frame

    def has_all_frames(self):
        return self._has_last_frame and self._has_first_frame

    def is_full(self):
        return len(self._memory["batched_inputs"]) == self.memory_size

    def pack_args(self):
        batched_inputs = list(itertools.chain(*self._memory["batched_inputs"]))
        features = [torch.cat([f[0] for f in self._memory["features"]])]
        img_box = torch.cat([boxes for boxes in self._memory["img_box"]])
        return batched_inputs, features, img_box

    def reset(self):
        self._has_first_frame = False
        self._has_last_frame = False
        self._memory = {}
        self._memory["batched_inputs"] = []
        self._memory["features"] = []
        self._memory["img_box"] = []
