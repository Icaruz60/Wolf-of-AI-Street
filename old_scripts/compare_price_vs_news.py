# compare_price_vs_news.py

import os
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Device & seeds
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class Config:
    csv_path: str = "data/eurusd_merged.csv"
    time_column: str = "time"
    close_column: str = "close"

    # news columns that actually exist in your file
    news_columns: tuple = ("mean_score", "mean_tone", "n_articles")

    sequence_length: int = 24   # hours in input window
    horizon: int = 4            # predict 4 hours ahead
    max_return: float = 0.002   # normalization scale for returns

    train_fraction: float = 0.7
    validation_fraction: float = 0.15

    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2

    # "strong signal" threshold: |pred| >= this -> trade
    significance_threshold: float = 0.01


cfg = Config()


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class PriceDataset(Dataset):
    """
    Features:
        - always normalized close price
        - optionally normalized news features
    Target:
        - normalized future return in [-1, 1]
    """

    def __init__(self, df: pd.DataFrame, cfg: Config, use_news: bool):
        self.cfg = cfg

        close = df[cfg.close_column].astype(float).values
        self.close = close

        # normalize close
        close_norm = (close - close.mean()) / (close.std() + 1e-8)

        # build news features if requested
        if use_news:
            available_news_cols = [c for c in cfg.news_columns if c in df.columns]

            if len(available_news_cols) == 0:
                # no news cols -> single zero feature
                news_features = np.zeros((len(close), 1), dtype=np.float32)
            else:
                news_list = []
                for col in available_news_cols:
                    arr = df[col].astype(float).values
                    std = arr.std()
                    if std > 0:
                        arr_norm = (arr - arr.mean()) / (std + 1e-8)
                    else:
                        arr_norm = np.zeros_like(arr)
                    news_list.append(arr_norm)
                news_features = np.stack(news_list, axis=1)  # (N, k)

            features = np.concatenate(
                [close_norm.reshape(-1, 1), news_features], axis=1
            )
        else:
            # price-only model
            features = close_norm.reshape(-1, 1)

        self.features = features.astype(np.float32)

        self.sequence_length = cfg.sequence_length
        self.horizon = cfg.horizon
        self.max_return = cfg.max_return

        # last usable index for prediction
        self.max_t = len(close) - self.horizon - 1
        if self.max_t <= self.sequence_length:
            raise ValueError("Not enough data to build sequences.")

    def __len__(self):
        return self.max_t - self.sequence_length + 1

    def __getitem__(self, index: int):
        # last index in the input window
        t = index + self.sequence_length - 1
        start = t - self.sequence_length + 1
        end = t + 1

        x_seq = self.features[start:end]  # (seq_len, n_features)

        p_t = self.close[t]
        p_future = self.close[t + self.horizon]
        raw_return = (p_future - p_t) / p_t

        # normalize return into [-1, 1]
        norm_return = np.clip(raw_return / self.max_return, -1.0, 1.0)

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(norm_return, dtype=torch.float32)
        return x_tensor, y_tensor


# -------------------------------------------------
# Model
# -------------------------------------------------
class PriceLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.out_activation = nn.Tanh()  # output in [-1, 1]

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]           # (batch, hidden_size)
        out = self.fc(h_last)      # (batch, 1)
        out = self.out_activation(out)
        return out.squeeze(-1)     # (batch,)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_csv(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    if cfg.time_column not in df.columns:
        raise ValueError(
            f"Missing time column '{cfg.time_column}'. Found: {list(df.columns)}"
        )
    if cfg.close_column not in df.columns:
        raise ValueError(
            f"Missing close column '{cfg.close_column}'. Found: {list(df.columns)}"
        )

    # sort by time
    df = df.sort_values(cfg.time_column).reset_index(drop=True)
    return df


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, significance_threshold: float):
    model.eval()
    total_loss = 0.0
    n = 0

    correct_sign = 0
    total_sign = 0

    sig_correct_sign = 0
    sig_total = 0
    realized_returns = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        # overall sign accuracy
        sign_pred = torch.sign(preds)
        sign_true = torch.sign(y)
        correct_sign += torch.sum(sign_pred == sign_true).item()
        total_sign += y.numel()

        # significant signals: |pred| >= threshold
        sig_mask = preds.abs() >= significance_threshold

        if sig_mask.any():
            sig_correct_sign += torch.sum(
                sign_pred[sig_mask] == sign_true[sig_mask]
            ).item()
            sig_total += sig_mask.sum().item()

            realized_returns.extend(
                (y[sig_mask] * sign_pred[sig_mask])
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    avg_loss = total_loss / n
    sign_acc = correct_sign / total_sign if total_sign > 0 else 0.0
    sig_acc = sig_correct_sign / sig_total if sig_total > 0 else 0.0
    coverage = sig_total / total_sign if total_sign > 0 else 0.0
    avg_realized = float(np.mean(realized_returns)) if realized_returns else 0.0

    return avg_loss, sign_acc, sig_acc, coverage, avg_realized


def run_experiment(df: pd.DataFrame, cfg: Config, use_news: bool, tag: str):
    print(f"\n============================")
    print(f"Experiment: {tag}")
    print(f"use_news = {use_news}")
    print(f"============================")

    dataset = PriceDataset(df, cfg, use_news=use_news)
    print("Dataset length:", len(dataset))

    # time-based split
    N = len(dataset)
    n_train = int(N * cfg.train_fraction)
    n_val = int(N * cfg.validation_fraction)
    n_test = N - n_train - n_val

    train_indices = range(0, n_train)
    val_indices = range(n_train, n_train + n_val)
    test_indices = range(n_train + n_val, N)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    print(
        f"Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    input_dim = dataset.features.shape[1]
    model = PriceLSTMRegressor(
        input_dim=input_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_sign_acc, val_sig_acc, val_cov, val_sig_ret = eval_epoch(
            model,
            val_loader,
            criterion,
            significance_threshold=cfg.significance_threshold,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_sign_acc={val_sign_acc:.3f} | "
            f"val_sig_acc={val_sig_acc:.3f} | "
            f"val_sig_cov={val_cov:.3f} | "
            f"val_sig_ret={val_sig_ret:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_sign_acc, test_sig_acc, test_cov, test_sig_ret = eval_epoch(
        model,
        test_loader,
        criterion,
        significance_threshold=cfg.significance_threshold,
    )

    print(
        f"TEST [{tag}] | "
        f"loss={test_loss:.6f} | "
        f"sign_acc={test_sign_acc:.3f} | "
        f"sig_acc={test_sig_acc:.3f} | "
        f"sig_cov={test_cov:.3f} | "
        f"sig_ret={test_sig_ret:.4f}"
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print(f"Using device: {device}")
    df = load_csv(cfg)

    # price-only model
    run_experiment(df, cfg, use_news=False, tag="PRICE_ONLY")

    # price + news model
    run_experiment(df, cfg, use_news=True, tag="PRICE_PLUS_NEWS")


if __name__ == "__main__":
    main()
