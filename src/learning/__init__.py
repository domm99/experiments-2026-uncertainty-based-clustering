import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


DATASET_ROOT = "./data"
FEATURE_SKEW_RMS = 0.15

__all__ = [
    "DATASET_ROOT",
    "FEATURE_SKEW_RMS",
    "FeatureSkewedSubset",
    "PredictorNetwork",
    "RNDModel",
    "TargetNetwork",
    "balanced_random_index_split",
    "create_target_rnd",
    "download_dataset",
    "evaluate_rnd_on_dataset",
    "partition_dataset",
    "set_seed",
    "train_rnd_on_dataset",
    "load_rnd_from_weights",
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


def _extract_image(sample: Any) -> torch.Tensor:
    if not isinstance(sample, tuple) or len(sample) < 1:
        raise TypeError("Dataset samples must return at least (image, label).")
    return _ensure_float_tensor(sample[0])


def _image_only_collate(batch: list[Any]) -> torch.Tensor:
    images = [_extract_image(sample) for sample in batch]
    if not images:
        raise ValueError("Cannot collate an empty batch.")

    first_shape = tuple(images[0].shape)
    for image in images:
        if tuple(image.shape) != first_shape:
            raise ValueError(
                "All images in a batch must have the same shape. "
                f"Expected {first_shape}, got {tuple(image.shape)}."
            )

    return torch.stack(images, dim=0)


def _batch_to_images(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        if batch.ndim != 4:
            raise ValueError(f"Expected image batch with shape BCHW, got {tuple(batch.shape)}.")
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        images = batch[0]
        if not torch.is_tensor(images):
            raise TypeError("Expected the first batch element to be an image tensor.")
        return images
    raise TypeError("Unsupported batch format.")


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


def _validate_rnd_dataset(dataset: Dataset) -> tuple[int, int, int]:
    if len(dataset) == 0:
        raise ValueError("Cannot train or evaluate RND on an empty dataset.")

    image_shape = _infer_dataset_image_shape(dataset)
    channels, height, width = image_shape
    if channels not in (1, 3):
        raise ValueError(
            "Only grayscale and RGB datasets are supported. "
            f"Got {channels} channels."
        )
    if height < 2 or width < 2:
        raise ValueError(
            "Images must have height and width >= 2 because the RND CNN uses MaxPool2d."
        )

    return image_shape


def _resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_uncertainty_reduction(uncertainty_reduction: str) -> str:
    uncertainty_reduction = uncertainty_reduction.lower()
    if uncertainty_reduction not in ("mean", "sum"):
        raise ValueError(
            "uncertainty_reduction must be either 'mean' or 'sum', "
            f"got {uncertainty_reduction!r}."
        )
    return uncertainty_reduction


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def init_rnd_weights(module: torch.nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TargetNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 64,
        channels: tuple[int, int] = (16, 32),
        spatial_pool_size: int = 4,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {input_channels}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if len(channels) != 2 or channels[0] <= 0 or channels[1] <= 0:
            raise ValueError(f"channels must contain two positive integers, got {channels}.")
        if spatial_pool_size <= 0:
            raise ValueError(f"spatial_pool_size must be > 0, got {spatial_pool_size}.")

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.spatial_pool_size = spatial_pool_size
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(spatial_pool_size, spatial_pool_size)),
            nn.Flatten(),
            nn.Linear(channels[1] * spatial_pool_size * spatial_pool_size, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 64,
        channels: tuple[int, int] = (4, 8),
        spatial_pool_size: int = 4,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {input_channels}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if len(channels) != 2 or channels[0] <= 0 or channels[1] <= 0:
            raise ValueError(f"channels must contain two positive integers, got {channels}.")
        if spatial_pool_size <= 0:
            raise ValueError(f"spatial_pool_size must be > 0, got {spatial_pool_size}.")

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.spatial_pool_size = spatial_pool_size
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(spatial_pool_size, spatial_pool_size)),
            nn.Flatten(),
            nn.Linear(channels[1] * spatial_pool_size * spatial_pool_size, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        target_network: TargetNetwork,
        embedding_dim: int = 64,
        predictor_channels: tuple[int, int] = (4, 8),
        spatial_pool_size: int = 4,
    ) -> None:
        super().__init__()
        if target_network.input_channels != input_channels:
            raise ValueError(
                "Target network input channels do not match the dataset. "
                f"Expected {input_channels}, got {target_network.input_channels}."
            )
        if target_network.embedding_dim != embedding_dim:
            raise ValueError(
                "Target network embedding dimension does not match. "
                f"Expected {embedding_dim}, got {target_network.embedding_dim}."
            )
        if target_network.spatial_pool_size != spatial_pool_size:
            raise ValueError(
                "Target network spatial pool size does not match. "
                f"Expected {spatial_pool_size}, got {target_network.spatial_pool_size}."
            )

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.predictor_channels = predictor_channels
        self.spatial_pool_size = spatial_pool_size
        self.target_network = target_network
        self.predictor_network = PredictorNetwork(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            channels=predictor_channels,
            spatial_pool_size=spatial_pool_size,
        )
        self.predictor_network.apply(init_rnd_weights)
        self.training_history: list[float] = []
        self._freeze_target()

    def _freeze_target(self) -> None:
        for parameter in self.target_network.parameters():
            parameter.requires_grad = False
        self.target_network.eval()

    def train(self, mode: bool = True) -> "RNDModel":
        super().train(mode)
        self.target_network.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_embedding = self.target_network(x)
        predictor_embedding = self.predictor_network(x)
        return target_embedding, predictor_embedding

    def uncertainty(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        reduction = _validate_uncertainty_reduction(reduction)
        self.eval()
        with torch.no_grad():
            target_embedding = self.target_network(x)
            predictor_embedding = self.predictor_network(x)
            squared_error = (predictor_embedding - target_embedding) ** 2
            if reduction == "mean":
                return torch.mean(squared_error, dim=1)
            return torch.sum(squared_error, dim=1)

    def warn_if_target_embedding_near_zero(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        try:
            first_batch = next(iter(loader))
        except StopIteration:
            return

        x = _batch_to_images(first_batch).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            target_abs_mean = self.target_network(x).abs().mean().item()

        if target_abs_mean < 1e-4:
            print(
                "WARNING: target_embedding.abs().mean() is very small "
                f"({target_abs_mean:.6g}); RND uncertainties may be weakly separated."
            )

    def train_on_loader(
        self,
        loader: DataLoader,
        epochs: int,
        lr: float,
        device: str | torch.device | None = None,
    ) -> list[float]:
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}.")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")

        torch_device = _resolve_torch_device(device)
        self.to(torch_device)
        self._freeze_target()
        self.warn_if_target_embedding_near_zero(loader, torch_device)

        optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=lr)
        epoch_losses: list[float] = []

        for _ in range(epochs):
            self.predictor_network.train()
            self.target_network.eval()
            total_loss = 0.0
            total_samples = 0

            for batch in loader:
                x = _batch_to_images(batch).to(device=torch_device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    target_embedding = self.target_network(x)
                predictor_embedding = self.predictor_network(x)
                loss = nnf.mse_loss(predictor_embedding, target_embedding)
                loss.backward()
                optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            if total_samples == 0:
                raise ValueError("Cannot train RND on an empty DataLoader.")

            epoch_losses.append(total_loss / total_samples)

        self.training_history.extend(epoch_losses)
        self.eval()
        return epoch_losses


def create_target_rnd(
    dataset: Dataset,
    seed: int,
    device: str | torch.device | None = None,
    embedding_dim: int = 64,
    target_channels: tuple[int, int] = (16, 32),
    spatial_pool_size: int = 4,
) -> TargetNetwork:
    image_shape = _validate_rnd_dataset(dataset)
    set_seed(seed)
    target_network = TargetNetwork(
        input_channels=image_shape[0],
        embedding_dim=embedding_dim,
        channels=target_channels,
        spatial_pool_size=spatial_pool_size,
    )
    target_network.apply(init_rnd_weights)
    for parameter in target_network.parameters():
        parameter.requires_grad = False

    target_network.eval()
    return target_network.to(_resolve_torch_device(device))

def load_rnd_from_weights(w, dataset, target_network, seed):
    image_shape = _validate_rnd_dataset(dataset)
    set_seed(seed)
    rnd_model = RNDModel(
        input_channels=image_shape[0],
        target_network=target_network,
        embedding_dim=64,
        predictor_channels=(4, 8),
        spatial_pool_size=4,
    )
    rnd_model.load_state_dict(w)
    return rnd_model

def train_rnd_on_dataset(
    dataset: Dataset,
    target_network: TargetNetwork,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | torch.device | None = None,
    embedding_dim: int = 64,
    predictor_channels: tuple[int, int] = (4, 8),
    spatial_pool_size: int = 4,
) -> RNDModel:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    image_shape = _validate_rnd_dataset(dataset)
    set_seed(seed)
    rnd_model = RNDModel(
        input_channels=image_shape[0],
        target_network=target_network,
        embedding_dim=embedding_dim,
        predictor_channels=predictor_channels,
        spatial_pool_size=spatial_pool_size,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 100_000)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=loader_generator,
        collate_fn=_image_only_collate,
    )
    rnd_model.train_on_loader(train_loader, epochs=epochs, lr=lr, device=device)
    return rnd_model


def evaluate_rnd_on_dataset(
    rnd_model: RNDModel,
    dataset: Dataset,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    uncertainty_reduction: str = "mean",
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    image_shape = _validate_rnd_dataset(dataset)
    if image_shape[0] != rnd_model.input_channels:
        raise ValueError(
            f"Dataset has {image_shape[0]} channels, but RND expects "
            f"{rnd_model.input_channels}."
        )

    uncertainty_reduction = _validate_uncertainty_reduction(uncertainty_reduction)
    torch_device = _resolve_torch_device(device)
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=_image_only_collate,
    )

    rnd_model.to(torch_device)
    rnd_model.eval()
    rnd_model._freeze_target()

    uncertainty_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in eval_loader:
            x = _batch_to_images(batch).to(device=torch_device, dtype=torch.float32)
            uncertainty = rnd_model.uncertainty(x, reduction=uncertainty_reduction)
            uncertainty_batches.append(uncertainty.detach().cpu())

    if not uncertainty_batches:
        raise ValueError("No samples found while evaluating RND.")

    uncertainties = torch.cat(uncertainty_batches, dim=0)
    return {
        "mean_uncertainty": float(uncertainties.mean().item()),
        "std_uncertainty": float(uncertainties.std(unbiased=False).item()),
        "min_uncertainty": float(uncertainties.min().item()),
        "max_uncertainty": float(uncertainties.max().item()),
        "num_samples": int(uncertainties.numel()),
        "uncertainty_reduction": uncertainty_reduction,
        "raw_uncertainties": [float(value) for value in uncertainties.tolist()],
    }
