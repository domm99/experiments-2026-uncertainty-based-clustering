import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as nnf
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


DATASET_ROOT = "./data"
FEATURE_SKEW_RMS = 0.15

__all__ = [
    "DATASET_ROOT",
    "FEATURE_SKEW_RMS",
    "FeatureSkewedSubset",
    "balanced_random_index_split",
    "download_dataset",
    "partition_dataset",
]


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("_", "").replace("-", "").upper()


def _ensure_float_tensor(image: Any) -> torch.Tensor:
    if torch.is_tensor(image):
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.is_floating_point():
            return image.to(dtype=torch.float32).clamp(0.0, 1.0)
        return F.convert_image_dtype(image, dtype=torch.float32).clamp(0.0, 1.0)

    return F.to_tensor(image).clamp(0.0, 1.0)


def _infer_dataset_image_shape(dataset: Dataset) -> tuple[int, int, int]:
    if len(dataset) == 0:
        raise ValueError("Cannot infer image shape from an empty dataset.")

    sample = dataset[0]
    if not isinstance(sample, tuple) or len(sample) < 1:
        raise TypeError("Expected dataset samples to contain at least an image.")

    image = _ensure_float_tensor(sample[0])
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(image.shape)}.")

    return tuple(int(value) for value in image.shape)


def balanced_random_index_split(
    dataset_size: int,
    n_splits: int,
    seed: int,
    allow_empty: bool = False,
) -> list[list[int]]:
    if n_splits <= 0:
        raise ValueError(f"n_splits must be > 0, got {n_splits}.")
    if dataset_size < 0:
        raise ValueError(f"dataset_size must be >= 0, got {dataset_size}.")
    if not allow_empty and n_splits > dataset_size:
        raise ValueError(
            f"n_splits={n_splits} would create empty subsets for dataset_size={dataset_size}."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()

    base_size = dataset_size // n_splits
    remainder = dataset_size % n_splits
    splits: list[list[int]] = []
    cursor = 0
    for split_id in range(n_splits):
        split_size = base_size + (1 if split_id < remainder else 0)
        splits.append(shuffled_indices[cursor : cursor + split_size])
        cursor += split_size

    return splits


def _regular_simplex_coordinates(n_groups: int) -> torch.Tensor:
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}.")
    if n_groups == 1:
        return torch.zeros((1, 0), dtype=torch.float32)

    coordinates = torch.zeros((n_groups, n_groups - 1), dtype=torch.float32)
    for column in range(n_groups - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        coordinates[: column + 1, column] = 1.0 / denominator
        coordinates[column + 1, column] = -(column + 1) / denominator

    return coordinates


def _create_equidistant_gaussian_patterns(
    n_groups: int,
    image_shape: Sequence[int],
    pattern_rms: float,
    seed: int,
    smooth_patterns: bool = True,
    smoothing_kernel_size: int = 7,
) -> list[torch.Tensor]:
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}.")
    if pattern_rms < 0:
        raise ValueError(f"pattern_rms must be >= 0, got {pattern_rms}.")
    if len(image_shape) != 3:
        raise ValueError(f"image_shape must be (C, H, W), got {tuple(image_shape)}.")

    channels, height, width = (int(value) for value in image_shape)
    image_dim = channels * height * width
    simplex_dim = n_groups - 1
    if simplex_dim > image_dim:
        raise ValueError(
            "Cannot create equidistant Gaussian patterns because "
            f"n_groups - 1 = {simplex_dim} exceeds image dimension {image_dim}."
        )

    if n_groups == 1 or pattern_rms == 0.0:
        zero = torch.zeros((channels, height, width), dtype=torch.float32)
        return [zero.clone() for _ in range(n_groups)]

    simplex = _regular_simplex_coordinates(n_groups)
    generator = torch.Generator()
    generator.manual_seed(seed)
    random_basis = torch.randn(
        image_dim,
        simplex_dim,
        generator=generator,
        dtype=torch.float32,
    )

    if smooth_patterns:
        max_kernel = min(smoothing_kernel_size, height, width)
        if max_kernel % 2 == 0:
            max_kernel -= 1

        if max_kernel >= 3:
            basis_as_images = random_basis.T.reshape(simplex_dim, channels, height, width)
            basis_as_images = basis_as_images.reshape(simplex_dim * channels, 1, height, width)
            basis_as_images = nnf.avg_pool2d(
                basis_as_images,
                kernel_size=max_kernel,
                stride=1,
                padding=max_kernel // 2,
            )
            basis_as_images = basis_as_images.reshape(simplex_dim, channels, height, width)
            random_basis = basis_as_images.reshape(simplex_dim, image_dim).T
            random_basis = random_basis - random_basis.mean(dim=0, keepdim=True)

    orthonormal_basis, _ = torch.linalg.qr(random_basis, mode="reduced")

    patterns = simplex @ orthonormal_basis.T
    row_norm = math.sqrt((n_groups - 1) / n_groups)
    target_norm = pattern_rms * math.sqrt(image_dim)
    patterns = patterns * (target_norm / row_norm)

    return [
        pattern.reshape(channels, height, width).contiguous()
        for pattern in patterns
    ]


class FeatureSkewedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        group_id: int,
        mean_shift_pattern: torch.Tensor,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.group_id = group_id
        self.mean_shift_pattern = mean_shift_pattern.detach().to(dtype=torch.float32).clone()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, Any]:
        dataset_index = self.indices[item]
        sample = self.dataset[dataset_index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise TypeError("Expected the wrapped dataset to return at least (image, label).")

        image, label = sample[0], sample[1]
        image = _ensure_float_tensor(image)
        if tuple(image.shape) != tuple(self.mean_shift_pattern.shape):
            raise ValueError(
                "Mean-shift pattern shape does not match image shape. "
                f"Expected {tuple(image.shape)}, got {tuple(self.mean_shift_pattern.shape)}."
            )

        pattern = self.mean_shift_pattern.to(device=image.device, dtype=image.dtype)
        return (image + pattern).clamp(0.0, 1.0).to(dtype=torch.float32), label


def download_dataset(dataset_name: str):
    to_tensor = transforms.ToTensor()
    dataset_key = _normalize_dataset_name(dataset_name)
    train_based_datasets = {
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "MNIST": datasets.MNIST,
        "FASHIONMNIST": datasets.FashionMNIST,
    }

    if dataset_key in train_based_datasets:
        dataset_factory = train_based_datasets[dataset_key]
        train_data = dataset_factory(
            root=DATASET_ROOT,
            train=True,
            transform=to_tensor,
            download=True,
        )
        test_data = dataset_factory(
            root=DATASET_ROOT,
            train=False,
            transform=to_tensor,
            download=True,
        )
        return train_data, test_data

    if dataset_key == "EMNIST":
        train_data = datasets.EMNIST(
            root=DATASET_ROOT,
            split="balanced",
            train=True,
            transform=to_tensor,
            download=True,
        )
        test_data = datasets.EMNIST(
            root=DATASET_ROOT,
            split="balanced",
            train=False,
            transform=to_tensor,
            download=True,
        )
        return train_data, test_data

    supported = ", ".join(["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "EMNIST"])
    raise ValueError(f"Unsupported dataset {dataset_name!r}. Supported datasets: {supported}.")


def partition_dataset(
    dataset: Dataset,
    number_of_areas: int,
    random_seed: int,
    feature_skew_rms: float = FEATURE_SKEW_RMS,
) -> list[Dataset]:
    group_indices = balanced_random_index_split(
        dataset_size=len(dataset),
        n_splits=number_of_areas,
        seed=random_seed,
    )
    image_shape = _infer_dataset_image_shape(dataset)
    mean_shift_patterns = _create_equidistant_gaussian_patterns(
        n_groups=number_of_areas,
        image_shape=image_shape,
        pattern_rms=feature_skew_rms,
        seed=random_seed,
    )

    return [
        FeatureSkewedSubset(
            dataset=dataset,
            indices=indices,
            group_id=group_id,
            mean_shift_pattern=mean_shift_patterns[group_id],
        )
        for group_id, indices in enumerate(group_indices)
    ]
