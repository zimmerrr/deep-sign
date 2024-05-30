from typing import List
import torch


class AugmentationFn:
    def __call__(self, *args):
        raise NotImplementedError("You need to implement this method")


class AugmentationV1:
    def __init__(self, augmentation_list: List[AugmentationFn]):
        self.augmentation_list = augmentation_list

    def __call__(self, *args):
        for augmentation in self.augmentation_list:
            args = augmentation(*args)

        return args


class Transform(AugmentationFn):
    def __init__(
        self,
        jitter=(0.1, 0.1, 0.1),
        interleave=(4, 3, 3),
        p=0.5,
    ):
        self.jitter = torch.tensor(jitter)
        self.jitter_half = self.jitter / 2
        self.interleave = interleave
        self.p = p

    def __call__(self, *args, **kwargs):
        assert len(self.interleave) == len(args)
        if kwargs["force"] or torch.rand(1) < self.p:
            jitter = (torch.rand(3) * self.jitter) - self.jitter_half
            for arg, interleave in zip(args, self.interleave):
                for i in range(0, len(arg), interleave):
                    arg[i + 0] += jitter[0]
                    arg[i + 1] += jitter[1]
                    arg[i + 2] += jitter[2]
        return args


class Flip(AugmentationFn):
    def __init__(
        self,
        horizontal=0.5,
        vertical=0.5,
        interleave=(4, 3, 3),
        p=0.5,
    ):
        self.horizontal = horizontal
        self.vertical = vertical
        self.interleave = interleave
        self.p = p

    def __call__(self, *args, **kwargs):
        assert len(self.interleave) == len(args)
        if kwargs["force"] or torch.rand(1) < self.p:
            rand = torch.rand(2)
            flip = (
                rand[0] < self.horizontal,
                rand[1] < self.vertical,
            )
            for arg, interleave in zip(args, self.interleave):
                for i in range(0, len(arg), interleave):
                    if flip[0]:
                        arg[i + 0] = 1 - arg[i + 0]
                    if flip[1]:
                        arg[i + 1] = 1 - arg[i + 1]
        return args
