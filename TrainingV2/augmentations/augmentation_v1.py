from typing import List
import torch
from augmentations.augmentation_fn_v1 import AugmentationFn, Flip, Transform, Scale


class AugmentationFn:
    def __str__(self):
        dict_str = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])
        return f"{self.__class__.__name__}({dict_str})"

    def get_params(self):
        pass

    def generate_vars(self):
        pass

    def __call__(self, pose, face, lh, rh, force=False):
        raise NotImplementedError("You need to implement this method")


class AugmentationV1:
    def __init__(self, augmentation_list: List[AugmentationFn]):
        self.augmentation_list = augmentation_list

    def __str__(self):
        string = [self.__class__.__name__ + "("]
        string.extend([f"    {str(aug)}" for aug in self.augmentation_list])
        string.append(")")
        return "\n".join(string)

    def __call__(self, frames: torch.Tensor):
        for augmentation in self.augmentation_list:
            augmentation.generate_vars()

        for idx, keypoints in enumerate(frames):
            pose, face, lh, rh = torch.split(keypoints, [132, 1404, 63, 63], dim=0)

            for augmentation in self.augmentation_list:
                pose, face, lh, rh = augmentation(pose, face, lh, rh)

            frames[idx] = torch.concat([pose, face, lh, rh])
        return frames

if __name__ == "__main__":
    augmentation = AugmentationV1([Transform(), Flip(), Scale()])
    augmentation(torch.rand((10, 1662)))
    print(augmentation)
