from typing import List, Optional, Tuple, Union

import torchvision.transforms as T
from PIL import Image as PILImage
from torch import Tensor


class BarlowTwinsTransform:
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, Optional[float], Optional[float]] = (0.485, 0.456, 0.406),
        std: Tuple[float, Optional[float], Optional[float]] = (0.229, 0.224, 0.225),
        **kwargs,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = T.Compose(
            [
                T.RandomResizedCrop(image_size, interpolation=PILImage.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [
                        T.GaussianBlur(
                            kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0)
                        )
                    ],
                    p=0.5,
                ),
                T.RandomSolarize(threshold=128, p=0.0),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Union[Tensor, PILImage.Image]) -> Tensor:
        return self.augment(image)


class BarlowTwinsPrimeTransform:
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, Optional[float], Optional[float]] = (0.485, 0.456, 0.406),
        std: Tuple[float, Optional[float], Optional[float]] = (0.229, 0.224, 0.225),
        **kwargs,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment_prime = T.Compose(
            [
                T.RandomResizedCrop(image_size, interpolation=PILImage.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [
                        T.GaussianBlur(
                            kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0)
                        )
                    ],
                    p=0.5,
                ),
                T.RandomSolarize(threshold=128, p=0.2),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Union[Tensor, PILImage.Image]) -> Tensor:
        return self.augment_prime(image)
