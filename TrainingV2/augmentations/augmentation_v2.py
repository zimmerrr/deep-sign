from typing import List
import torch
from augmentations.augmentation_fn_v1 import (
    AugmentationFn,
    Flip,
    Transform,
    Scale,
    Rotate,
)


class AugmentationV2:
    def __init__(self, augmentation_list: List[AugmentationFn]):
        self.augmentation_list = augmentation_list

    def __str__(self):
        string = [self.__class__.__name__ + "("]
        string.extend([f"    {str(aug)}" for aug in self.augmentation_list])
        string.append(")")
        return "\n".join(string)

    def __call__(self, pose, face, lh, rh):
        for augmentation in self.augmentation_list:
            augmentation.generate_vars()

        for frame_idx in range(len(pose)):
            for augmentation in self.augmentation_list:
                pose[frame_idx], face[frame_idx], lh[frame_idx], rh[frame_idx] = (
                    augmentation(
                        pose[frame_idx],
                        face[frame_idx],
                        lh[frame_idx],
                        rh[frame_idx],
                    )
                )

        return pose, face, lh, rh


if __name__ == "__main__":
    augmentation = AugmentationV2([Rotate(), Flip(), Scale(), Transform()])
    augmentation(
        torch.rand((33 * 4)),
        torch.rand((468 * 3)),
        torch.rand((21 * 3)),
        torch.rand((21 * 3)),
    )
    print(augmentation)
