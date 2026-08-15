"""
Multimodal model for real estate valuation prediction.

Combines a CNN branch that processes 128×128 satellite-image patches
(channels: NDVI, urban density, land-use mix) with an MLP branch that
processes tabular features.  The two branches are concatenated and
passed through a regression head.

Uses PyTorch.  If torch is not installed, calling :func:`train` or
:func:`predict` raises an informative ``RuntimeError``.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    logger.warning("PyTorch not installed — multimodal model will not be available")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "annualized_valuation"
EXCLUDE_COLS: Tuple[str, ...] = ("cell_id", "lat", "lon", TARGET_COL)
IMG_SIZE = 128
IMG_CHANNELS = 3  # NDVI, urban_density, land_use_mix


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


if _HAS_TORCH:

    class ValuationDataset(Dataset):
        """PyTorch dataset for multimodal valuation training.

        Parameters
        ----------
        df:
            Feature DataFrame.
        image_dir:
            Directory containing ``{cell_id}.npy`` satellite patches.
        tabular_features:
            Ordered list of tabular feature column names.
        """

        def __init__(
            self,
            df: pd.DataFrame,
            image_dir: Optional[Union[str, Path]] = None,
            tabular_features: Optional[List[str]] = None,
        ) -> None:
            self.df = df.reset_index(drop=True)
            self.image_dir = Path(image_dir) if image_dir else None
            if tabular_features is None:
                drop = [c for c in EXCLUDE_COLS if c in df.columns]
                tabular_features = [c for c in df.columns if c not in drop]
            self.tabular_features = tabular_features

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            row = self.df.iloc[idx]

            # Tabular features
            tab = torch.tensor(
                row[self.tabular_features].values.astype(np.float32),
                dtype=torch.float32,
            )

            # Image patch
            img = torch.zeros(IMG_CHANNELS, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            if self.image_dir is not None:
                cell_id = row.get("cell_id", str(idx))
                img_path = self.image_dir / f"{cell_id}.npy"
                if img_path.exists():
                    arr = np.load(img_path)
                    if arr.shape == (IMG_CHANNELS, IMG_SIZE, IMG_SIZE):
                        img = torch.tensor(arr, dtype=torch.float32)
                    elif arr.shape == (IMG_SIZE, IMG_SIZE, IMG_CHANNELS):
                        img = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)

            # Target
            target = torch.tensor(
                float(row.get(TARGET_COL, 0.0)),
                dtype=torch.float32,
            )
            return img, tab, target


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


if _HAS_TORCH:

    class CNNBranch(nn.Module):
        """CNN that processes 128×128×3 satellite patches."""

        def __init__(self, out_dim: int = 64) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64×64
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32×32
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16×16
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),  # 1×1
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.features(x))

    class MLPBranch(nn.Module):
        """MLP that processes tabular features."""

        def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), out_dim: int = 64) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev = in_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(0.2)])
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            layers.append(nn.ReLU(inplace=True))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class MultimodalValuationModel(nn.Module):
        """Full multimodal model: CNN + MLP → regression head."""

        def __init__(self, tabular_dim: int, cnn_out: int = 64, mlp_out: int = 64) -> None:
            super().__init__()
            self.cnn = CNNBranch(out_dim=cnn_out)
            self.mlp = MLPBranch(in_dim=tabular_dim, out_dim=mlp_out)
            self.head = nn.Sequential(
                nn.Linear(cnn_out + mlp_out, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
            cnn_feat = self.cnn(img)
            mlp_feat = self.mlp(tab)
            combined = torch.cat([cnn_feat, mlp_feat], dim=1)
            return self.head(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed.  Install it with `pip install torch` "
            "to use the multimodal model."
        )


def train(
    df: pd.DataFrame,
    image_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
) -> Dict:
    """Train the multimodal valuation model.

    Parameters
    ----------
    df:
        Feature DataFrame.
    image_dir:
        Directory with ``{cell_id}.npy`` image patches.  If *None* or
        missing, zero-tensors are used as placeholders.
    save_dir:
        Where to save model weights.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Optimiser learning rate.
    val_split:
        Fraction of data held out for validation.

    Returns
    -------
    dict with keys ``model``, ``feature_names``, ``metrics``.
    """
    _require_torch()

    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop]
    tabular_dim = len(feature_names)

    logger.info("Multimodal training | rows=%d | tabular_dim=%d", len(df), tabular_dim)

    # Shuffle and split
    idx = np.random.RandomState(42).permutation(len(df))
    n_val = max(1, int(len(df) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = ValuationDataset(train_df, image_dir, feature_names)
    val_ds = ValuationDataset(val_df, image_dir, feature_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalValuationModel(tabular_dim=tabular_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses: List[float] = []
        for imgs, tabs, targets in train_loader:
            imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs, tabs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for imgs, tabs, targets in val_loader:
                imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
                preds = model(imgs, tabs)
                val_losses.append(criterion(preds, targets).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                epoch + 1, epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"],
            )

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"best_val_loss": best_val_loss, "epochs": epochs}

    # Save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_path = save_dir / "multimodal.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": feature_names,
                "tabular_dim": tabular_dim,
                "metrics": metrics,
            },
            weights_path,
        )
        logger.info("Saved multimodal model → %s", weights_path)

    return {"model": model, "feature_names": feature_names, "metrics": metrics}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    df: pd.DataFrame,
    model=None,
    image_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Generate predictions from the multimodal model.

    Parameters
    ----------
    df:
        Feature DataFrame.
    model:
        A trained ``MultimodalValuationModel``.  If *None*, loaded from
        *model_path*.
    image_dir:
        Directory with image patches.
    model_path:
        Path to a saved ``.pt`` checkpoint.
    batch_size:
        Inference batch size.

    Returns
    -------
    pd.DataFrame with column ``pred_mm``.
    """
    _require_torch()

    if model is None:
        if model_path is None:
            raise ValueError("Provide either *model* or *model_path*")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        feature_names = checkpoint["feature_names"]
        tabular_dim = checkpoint["tabular_dim"]
        model = MultimodalValuationModel(tabular_dim=tabular_dim)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        drop = [c for c in EXCLUDE_COLS if c in df.columns]
        feature_names = [c for c in df.columns if c not in drop]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = ValuationDataset(df, image_dir, feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: List[float] = []
    with torch.no_grad():
        for imgs, tabs, _ in loader:
            imgs, tabs = imgs.to(device), tabs.to(device)
            preds = model(imgs, tabs)
            all_preds.extend(preds.cpu().numpy().tolist())

    return pd.DataFrame({"pred_mm": all_preds}, index=df.index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train multimodal valuation model")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    result = train(df, image_dir=args.image_dir, save_dir=args.save_dir, epochs=args.epochs)
    print(result["metrics"])
