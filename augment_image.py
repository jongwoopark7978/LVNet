from __future__ import annotations

import math
import os
import random
from typing import Callable, List, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _validate_inputs(
    num_images: int,
    functions: Sequence[Callable[[Image.Image], Image.Image]],
    ratios: Sequence[float],
) -> None:
    if num_images <= 0:
        raise ValueError("num_images must be a positive integer")

    if not functions:
        raise ValueError("`functions` must contain at least one augmentation function")

    if len(functions) != len(ratios):
        raise ValueError("`functions` and `ratios` must have the same length")

    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("`ratios` must add up to 1.0")

    if any(r < 0 for r in ratios):
        raise ValueError("`ratios` cannot contain negative values")


def _allocate_counts(num_images: int, ratios: Sequence[float]) -> List[int]:
    exact = [r * num_images for r in ratios]
    counts = [math.floor(x) for x in exact]
    remainder = num_images - sum(counts)

    # distribute leftover images to largest remainders
    frac_indices = sorted(
        range(len(ratios)),
        key=lambda i: exact[i] - counts[i],
        reverse=True,
    )
    for i in range(remainder):
        counts[frac_indices[i]] += 1

    return counts


# ----------------------------------------------------------------------
# core API
# ----------------------------------------------------------------------
def augment_image(
    image: Image.Image,
    num_images: int,
    functions: List[Callable[[Image.Image], Image.Image]],
    ratios: List[float],
    shuffle: bool = True,
) -> List[Image.Image]:
    _validate_inputs(num_images, functions, ratios)
    counts = _allocate_counts(num_images, ratios)

    augmented: List[Image.Image] = []
    for func, count in zip(functions, counts):
        augmented.extend(func(image) for _ in range(count))

    if shuffle:
        random.shuffle(augmented)

    return augmented


# ----------------------------------------------------------------------
# augmentation ops
# ----------------------------------------------------------------------
def gaussian_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def gaussian_noise(
    img: Image.Image,
    mean: float = 0.0,
    std: float = 10.0,
    clip_min: int = 0,
    clip_max: int = 255,
) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape).astype(np.float32)
    noisy_arr = np.clip(arr + noise, clip_min, clip_max).astype(np.uint8)
    return Image.fromarray(noisy_arr, mode=img.mode)


def flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def rotate(img: Image.Image, degrees: float = 15.0) -> Image.Image:
    return img.rotate(degrees, expand=True)


def brighten(img: Image.Image, factor: float = 1.3) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

def identity(img: Image.Image) -> Image.Image:
    return img


# ----------------------------------------------------------------------
# example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img = Image.open("picture.png")

    augmented = augment_image(
        img,
        num_images=60,
        functions=[gaussian_blur, gaussian_noise, identity],
        ratios=[0.45, 0.45, 0.1],
    )

    os.makedirs("augmented", exist_ok=True)
    for i, im in enumerate(augmented):
        im = im.convert("RGB")
        im.save(f"augmented/{i}.jpg")

