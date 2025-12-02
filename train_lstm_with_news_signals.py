"""
LSTM on EUR/USD hourly data with news features (news-era only).

- Assumes a CSV with columns:
    timestamp_utc, open, high, low, close,
    news_count, news_score_sum, news_score_mean, news_score_max,
    news_tone_avg, news_tone_pos, news_tone_neg, news_tone_polarity

- Expected file: eurusd_merged_news_only.csv
  (trimmed so that news_* features are actually meaningful).

- Builds sliding windows of length seq_len and predicts next-hour return (normalized).
- Uses robust normalization to avoid NaNs.
- Prints detailed training / validation metrics, with special focus on
  "significant trades" (sig_* metrics) and a threshold chosen on the
  validation set to maximize expected normalized return.
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class Config:
    # Use the trimmed file produced by trim_eurusd_news_era.py
    csv_path: str = "data/final_dataset.csv"

    seq_len: int = 72          # hours of context (3 days)
    batch_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 20

    # Temporal split fractions
    train_frac: float = 0.70
    val_frac: float = 0.15  # rest is test

    # Threshold grid in *normalized return space*
    thr_grid: List[float] = None

    # minimum significant coverage
    min_sig_cov: float = 0.02


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int):
        assert x.shape[0] == y.shape[0]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Last seq_len rows cannot start a full window
        return max(0, self.x.shape[0] - self.seq_len)

    def __getitem__(self, idx: int):
        i0 = idx
        i1 = idx + self.seq_len
        x_seq = self.x[i0:i1]          # (seq_len, n_features)
        y_next = self.y[i1]            # predict return at next step after window
        return torch.from_numpy(x_seq), torch.tensor([y_next], dtype=torch.float32)


# -------------------------------------------------
# Model
# -------------------------------------------------
class PriceNewsLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]       # (batch, hidden_size)
        pred = self.head(last)     # (batch, 1), normalized return
        return pred


# -------------------------------------------------
# Data loading & preprocessing
# -------------------------------------------------
def robust_standardize(train_arr: np.ndarray, full_arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standard z-score, but if std ~0 we avoid division and just return zeros
    (feature is essentially constant).
    """
    mean = float(train_arr.mean())
    std = float(train_arr.std())
    if std < 1e-8:
        return np.zeros_like(full_arr, dtype=np.float32), mean, 0.0
    out = (full_arr - mean) / std
    return out.astype(np.float32), mean, std


def build_features_and_target(df: pd.DataFrame):
    # Sort by time to be safe
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Price columns
    price_cols = ["open", "high", "low", "close"]

    # News columns
    count_col = "news_count"
    news_sum_col = "news_score_sum"
    news_mean_col = "news_score_mean"
    news_max_col = "news_score_max"
    tone_cols = ["news_tone_avg", "news_tone_pos", "news_tone_neg", "news_tone_polarity"]

    required = ["timestamp_utc"] + price_cols + [count_col, news_sum_col, news_mean_col, news_max_col] + tone_cols
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in CSV. Found columns: {list(df.columns)}")

    # Build target: next-hour log return of close
    close = df["close"].astype(float).values
    eps = 1e-12
    ret = np.log((close[1:] + eps) / (close[:-1] + eps))  # length N-1

    # Align features with target
    df_feat = df.iloc[:-1].copy()
    assert len(df_feat) == len(ret)

    # Log-transform heavy-tailed non-negative news features
    df_feat[count_col] = np.log1p(df_feat[count_col].clip(lower=0))
    df_feat[news_sum_col] = np.log1p(df_feat[news_sum_col].clip(lower=0))
    df_feat[news_mean_col] = np.log1p(df_feat[news_mean_col].clip(lower=0))
    df_feat[news_max_col] = np.log1p(df_feat[news_max_col].clip(lower=0))

    feature_cols = price_cols + [count_col, news_sum_col, news_mean_col, news_max_col] + tone_cols
    feat_arr = df_feat[feature_cols].astype(float).values  # (N-1, D)

    return feat_arr, ret, feature_cols


def train_val_test_split_indices(n: int, train_frac: float, val_frac: float):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def load_and_prepare_data(cfg: Config):
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV file not found at {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"Missing 'timestamp_utc' column. Found: {list(df.columns)}")

    # Parse timestamps
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    feat_arr, ret, feature_cols = build_features_and_target(df)
    n = len(ret)
    print(f"Dataset length (after building target): {n}")

    # Train/val/test split on time order
    n_train, n_val, n_test = train_val_test_split_indices(n, cfg.train_frac, cfg.val_frac)

    idx_train_end = n_train
    idx_val_end = n_train + n_val

    feat_train = feat_arr[:idx_train_end]
    feat_val = feat_arr[idx_train_end:idx_val_end]
    feat_test = feat_arr[idx_val_end:]

    y_train = ret[:idx_train_end]
    y_val = ret[idx_train_end:idx_val_end]
    y_test = ret[idx_val_end:]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Normalize features using train statistics only
    feat_all_norm = np.empty_like(feat_arr, dtype=np.float32)
    stats = {}
    for j in range(feat_arr.shape[1]):
        col_train = feat_train[:, j]
        col_full = feat_arr[:, j]
        col_norm, mean, std = robust_standardize(col_train, col_full)
        feat_all_norm[:, j] = col_norm
        stats[feature_cols[j]] = (mean, std)

    # Normalize targets (standardized log returns)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-12:
        y_std = 1.0
    y_all_norm = ((ret - y_mean) / y_std).astype(np.float32)

    target_stats = (y_mean, y_std)

    # Re-split normalized arrays
    y_train_n = y_all_norm[:idx_train_end]
    y_val_n = y_all_norm[idx_train_end:idx_val_end]
    y_test_n = y_all_norm[idx_val_end:]

    feat_train_n = feat_all_norm[:idx_train_end]
    feat_val_n = feat_all_norm[idx_train_end:idx_val_end]
    feat_test_n = feat_all_norm[idx_val_end:]

    # Create datasets
    train_ds = SeqDataset(feat_train_n, y_train_n, cfg.seq_len)
    val_ds = SeqDataset(feat_val_n, y_val_n, cfg.seq_len)
    test_ds = SeqDataset(feat_test_n, y_test_n, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        stats,
        target_stats,
    )


# -------------------------------------------------
# Training & evaluation utils
# -------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)  # (batch, 1)

        optimizer.zero_grad()
        preds = model(x)        # (batch, 1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


def evaluate_loss_and_sign_acc(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    correct_sign = 0
    total_sign = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # (batch, 1)

            preds = model(x)  # (batch, 1)
            loss = criterion(preds, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            sign_pred = torch.sign(preds)
            sign_true = torch.sign(y)
            mask = (sign_true != 0)
            if mask.any():
                correct_sign += (sign_pred[mask] == sign_true[mask]).sum().item()
                total_sign += mask.sum().item()

    avg_loss = total_loss / max(1, total_count)
    sign_acc = correct_sign / total_sign if total_sign > 0 else 0.0
    return avg_loss, sign_acc


def collect_predictions(model, loader):
    model.eval()
    preds_list = []
    y_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_list.append(preds.squeeze(-1).cpu().numpy())
            y_list.append(y.squeeze(-1).cpu().numpy())

    if len(preds_list) == 0:
        return np.array([]), np.array([])

    preds_all = np.concatenate(preds_list)
    y_all = np.concatenate(y_list)
    return preds_all, y_all


def threshold_scan(preds, y_true, thr_values: List[float]):
    """
    Scan thresholds on |pred| in normalized space, compute:
        - sig_acc: accuracy of sign on subset
        - sig_cov: coverage (fraction of points where |pred| >= thr)
        - sig_ret: average signed normalized return: sign(pred) * true
    """
    results = []
    for thr in thr_values:
        mask = np.abs(preds) >= thr
        cov = mask.mean() if len(mask) > 0 else 0.0
        if mask.sum() == 0:
            results.append((thr, 0.0, cov, 0.0))
            continue

        pred_sub = preds[mask]
        true_sub = y_true[mask]

        sign_pred = np.sign(pred_sub)
        sign_true = np.sign(true_sub)

        correct = (sign_pred == sign_true).sum()
        sig_acc = correct / len(sign_true)

        sig_ret = float(np.mean(sign_pred * true_sub))
        results.append((thr, sig_acc, cov, sig_ret))

    return results


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = Config()
    if cfg.thr_grid is None:
        # Dense grid from 0.0 to 1.0 in steps of 0.0005
        #  -> 0.0000, 0.0005, 0.0010, ...
        cfg.thr_grid = np.round(
            np.arange(0.0, 1.0005, 0.0005), 4
        ).tolist()

    print(f"Using device: {device}")
    print(f"CSV path: {cfg.csv_path}")

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        feature_stats,
        target_stats,
    ) = load_and_prepare_data(cfg)

    input_size = len(feature_cols)

    model = PriceNewsLSTM(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_sign_acc = evaluate_loss_and_sign_acc(model, val_loader, criterion)

        # Collect validation predictions for sig_* metrics
        val_preds, val_true = collect_predictions(model, val_loader)
        if len(val_preds) == 0:
            val_sig_acc = 0.0
            val_sig_cov = 0.0
            val_sig_ret = 0.0
            best_thr_epoch = 0.0
        else:
            thr_results = threshold_scan(val_preds, val_true, cfg.thr_grid)

            best_thr_epoch = None
            best_sig_acc_epoch = -1.0
            best_sig_cov_epoch = 0.0
            best_sig_ret_epoch = 0.0

            for thr, sig_acc, sig_cov, sig_ret in thr_results:
                # enforce minimum coverage
                if sig_cov < cfg.min_sig_cov:
                    continue
                if sig_acc > best_sig_acc_epoch:
                    best_sig_acc_epoch = sig_acc
                    best_sig_cov_epoch = sig_cov
                    best_sig_ret_epoch = sig_ret
                    best_thr_epoch = thr

            # If nothing met coverage constraint, fall back to 0
            if best_thr_epoch is None:
                best_thr_epoch = 0.0
                best_sig_acc_epoch = 0.0
                best_sig_cov_epoch = 0.0
                best_sig_ret_epoch = 0.0

            val_sig_acc = best_sig_acc_epoch
            val_sig_cov = best_sig_cov_epoch
            val_sig_ret = best_sig_ret_epoch

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_sign_acc={val_sign_acc:.3f} | "
            f"val_sig_acc={val_sig_acc:.3f} | "
            f"val_sig_cov={val_sig_cov:.3f} | "
            f"val_sig_ret={val_sig_ret:.4f} | "
            f"val_best_thr={best_thr_epoch:.4f}"
        )

    # -----------------------------
    # Threshold search on validation set (final)
    # -----------------------------
    val_preds, val_true = collect_predictions(model, val_loader)
    if len(val_preds) == 0:
        print("No validation predictions (dataset too small with this seq_len). Skipping threshold scan.")
        best_thr = 0.0
        best_sig_acc = 0.0
        best_sig_cov = 0.0
        best_sig_ret = 0.0
    else:
        print("\nSearching best significance threshold on validation set...")
        thr_results = threshold_scan(val_preds, val_true, cfg.thr_grid)

        best_thr = None
        best_sig_acc = -1.0
        best_sig_cov = 0.0
        best_sig_ret = 0.0

        for thr, sig_acc, sig_cov, sig_ret in thr_results:

            # enforce minimum coverage
            if sig_cov < cfg.min_sig_cov:
                continue

            if sig_acc > best_sig_acc:
                best_sig_acc = sig_acc
                best_sig_cov = sig_cov
                best_sig_ret = sig_ret
                best_thr = thr

        if best_thr is None:
            best_thr = 0.0
            best_sig_acc = 0.0
            best_sig_cov = 0.0
            best_sig_ret = 0.0

        print(
            f"\nChosen significance_threshold: {best_thr:.4f} "
            f"(sig_acc={best_sig_acc:.3f}, sig_cov={best_sig_cov:.3f}, sig_ret={best_sig_ret:.4f})"
        )

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    test_loss, test_sign_acc = evaluate_loss_and_sign_acc(model, test_loader, criterion)
    test_preds, test_true = collect_predictions(model, test_loader)
    if len(test_preds) == 0:
        test_sig_acc = test_sig_cov = test_sig_ret = 0.0
        if best_thr is None:
            best_thr = 0.0
    else:
        mask = np.abs(test_preds) >= best_thr
        if mask.sum() == 0:
            test_sig_acc = 0.0
            test_sig_cov = 0.0
            test_sig_ret = 0.0
        else:
            sign_pred = np.sign(test_preds[mask])
            sign_true = np.sign(test_true[mask])
            test_sig_acc = (sign_pred == sign_true).mean()
            test_sig_cov = mask.mean()
            test_sig_ret = float(np.mean(sign_pred * test_true[mask]))

    print(
        f"TEST | loss={test_loss:.6f} | "
        f"sign_acc={test_sign_acc:.3f} | "
        f"sig_acc={test_sig_acc:.3f} | "
        f"sig_cov={test_sig_cov:.3f} | "
        f"sig_ret={test_sig_ret:.4f} | "
        f"thr={best_thr:.4f}"
    )

    torch.save(model.state_dict(), "price_lstm_with_news_signals.pt")
    print("Saved model to price_lstm_with_news_signals.pt")


if __name__ == "__main__":
    main()
