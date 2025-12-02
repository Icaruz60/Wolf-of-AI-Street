"""
Train an LSTM on EUR/USD hourly data with news features and
evaluate it as a trading strategy using position sizing directly
from the model's continuous [-1, 1]-like output.

Instead of tuning a hard significance threshold, we simulate an
equity curve where each hourly prediction controls the fraction
of capital allocated long/short.
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
    # Data
    csv_path: str = "data/final_dataset.csv"

    seq_len: int = 72          # hours of context (3 days)
    batch_size: int = 64

    # Model
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    # Optimizer / training
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 100

    # Temporal split fractions
    train_frac: float = 0.70
    val_frac: float = 0.15  # rest is test

    # Early stopping based on validation PnL
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0

    # Trading simulation parameters
    initial_capital: float = 10_000.0
    max_position: float = 0.5       # max fraction of capital long/short
    position_scale: float = 1.0     # scale before tanh, controls saturation
    pred_dead_zone: float = 0.0     # |pred| below this → no position


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

    # Timestamp aligned with the "next" bar (return of that hour)
    timestamps = df["timestamp_utc"].values[1:]

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

    return feat_arr, ret, feature_cols, timestamps


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

    feat_arr, ret, feature_cols, timestamps = build_features_and_target(df)
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

    ts_train = timestamps[:idx_train_end]
    ts_val = timestamps[idx_train_end:idx_val_end]
    ts_test = timestamps[idx_val_end:]

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
        ts_train,
        ts_val,
        ts_test,
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


# -------------------------------------------------
# Trading simulation
# -------------------------------------------------
def simulate_equity(
    preds_norm: np.ndarray,
    true_norm: np.ndarray,
    target_stats: Tuple[float, float],
    cfg: Config,
):
    """
    Simulate an equity curve where each hourly prediction controls
    the fraction of capital allocated long/short.

    - preds_norm, true_norm are in normalized space.
    - We de-normalize true_norm back to log-returns.
    - Position fraction f_t is derived by mapping preds_norm through
      a tanh to obtain a stable [-1, 1] signal.
    """
    if preds_norm.size == 0 or true_norm.size == 0:
        return np.array([cfg.initial_capital], dtype=np.float32), 0.0

    y_mean, y_std = target_stats
    # De-normalize log-returns
    true_logret = true_norm * y_std + y_mean  # shape (T,)

    capital = cfg.initial_capital
    equity = [capital]

    max_pos = cfg.max_position
    scale = cfg.position_scale
    dead_zone = cfg.pred_dead_zone

    for p_raw, r in zip(preds_norm, true_logret):
        # Map raw model output to [-1, 1] with tanh
        p = float(np.tanh(scale * float(p_raw)))

        # Dead zone around zero: no position if confidence is too low
        if abs(p) < dead_zone:
            f = 0.0
        else:
            f = max_pos * p

        # Convert log-return to simple price return
        price_ret = float(np.exp(r) - 1.0)
        # Capital update: capital_{t+1} = capital_t * (1 + f * price_ret)
        capital *= (1.0 + f * price_ret)
        equity.append(capital)

    equity_arr = np.array(equity, dtype=np.float32)
    total_log_ret = float(np.log(equity_arr[-1] / equity_arr[0]))
    return equity_arr, total_log_ret


# -------------------------------------------------
# CSV dumping for inspection
# -------------------------------------------------
def dump_predictions_to_csv(
    split_name: str,
    preds_norm: np.ndarray,
    true_norm: np.ndarray,
    split_timestamps: np.ndarray,
    target_stats: Tuple[float, float],
    cfg: Config,
    filename: str,
):
    """
    Dump per-row predictions and returns for inspection.

    For a given split (train/val/test) with target length L, the SeqDataset
    yields predictions only for indices [seq_len, L-1]. The i-th prediction
    from collect_predictions corresponds to target/return index:
        idx = seq_len + i
    in the split arrays.
    """
    if preds_norm.size == 0 or true_norm.size == 0:
        print(f"No predictions for split '{split_name}', skipping CSV dump.")
        return

    if preds_norm.shape != true_norm.shape:
        raise ValueError(f"Shape mismatch in dump_predictions_to_csv for '{split_name}'")

    y_mean, y_std = target_stats
    true_logret = true_norm * y_std + y_mean

    # Map normalized predictions to trading signal and position fraction
    signals = np.tanh(cfg.position_scale * preds_norm.astype(np.float64))
    pos_frac = np.where(
        np.abs(signals) < cfg.pred_dead_zone,
        0.0,
        cfg.max_position * signals,
    )

    # Compute simple returns for reference
    simple_ret = np.exp(true_logret) - 1.0

    # Map to timestamps: prediction i corresponds to target index seq_len + i
    idx_offset = cfg.seq_len
    max_idx = idx_offset + preds_norm.shape[0]
    if max_idx > len(split_timestamps):
        raise ValueError(
            f"Not enough timestamps for split '{split_name}': "
            f"needed up to index {max_idx}, have {len(split_timestamps)}"
        )

    ts_used = split_timestamps[idx_offset:max_idx]

    df_out = pd.DataFrame(
        {
            "timestamp_utc": ts_used,
            "pred_norm": preds_norm.astype(float),
            "signal": signals.astype(float),
            "position_fraction": pos_frac.astype(float),
            "true_log_return": true_logret.astype(float),
            "true_simple_return": simple_ret.astype(float),
        }
    )

    df_out.to_csv(filename, index=False)
    print(f"Saved {split_name} predictions to {filename} (rows={len(df_out)})")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = Config()

    print(f"Using device: {device}")
    print(f"CSV path: {cfg.csv_path}")

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        feature_stats,
        target_stats,
        ts_train,
        ts_val,
        ts_test,
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

    best_state_dict = None
    best_epoch = 0
    best_val_log_ret = -1e9
    best_val_equity_final = cfg.initial_capital
    epochs_without_improvement = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_sign_acc = evaluate_loss_and_sign_acc(model, val_loader, criterion)

        # Validation PnL simulation
        val_preds, val_true_norm = collect_predictions(model, val_loader)
        if val_preds.size == 0:
            val_equity_curve = np.array([cfg.initial_capital], dtype=np.float32)
            val_log_ret = 0.0
        else:
            val_equity_curve, val_log_ret = simulate_equity(val_preds, val_true_norm, target_stats, cfg)

        val_equity_final = float(val_equity_curve[-1])

        improved = False
        if val_log_ret > best_val_log_ret + cfg.early_stopping_min_delta:
            improved = True

        if improved:
            best_val_log_ret = val_log_ret
            best_val_equity_final = val_equity_final
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_sign_acc={val_sign_acc:.3f} | "
            f"val_equity_final={val_equity_final:.2f} | "
            f"val_log_ret={val_log_ret:.4f}"
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_log_ret={best_val_log_ret:.4f}."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"Restored best model from epoch {best_epoch} "
            f"(val_log_ret={best_val_log_ret:.4f}, "
            f"val_equity_final={best_val_equity_final:.2f})."
        )
    else:
        print("Warning: no improvement found during training; using last epoch model.")

    # -----------------------------
    # Final test evaluation (PnL)
    # -----------------------------
    test_loss, test_sign_acc = evaluate_loss_and_sign_acc(model, test_loader, criterion)
    test_preds, test_true_norm = collect_predictions(model, test_loader)
    if test_preds.size == 0:
        test_equity_curve = np.array([cfg.initial_capital], dtype=np.float32)
        test_log_ret = 0.0
    else:
        test_equity_curve, test_log_ret = simulate_equity(test_preds, test_true_norm, target_stats, cfg)

    test_equity_final = float(test_equity_curve[-1])

    print(
        f"TEST | loss={test_loss:.6f} | "
        f"sign_acc={test_sign_acc:.3f} | "
        f"equity_final={test_equity_final:.2f} | "
        f"log_ret={test_log_ret:.4f}"
    )

    # Dump per-row predictions for inspection (validation and test)
    val_preds_final, val_true_norm_final = collect_predictions(model, val_loader)
    dump_predictions_to_csv(
        "val",
        val_preds_final,
        val_true_norm_final,
        ts_val,
        target_stats,
        cfg,
        "val_predictions.csv",
    )
    dump_predictions_to_csv(
        "test",
        test_preds,
        test_true_norm,
        ts_test,
        target_stats,
        cfg,
        "test_predictions.csv",
    )

    torch.save(model.state_dict(), "price_lstm_trader.pt")
    print("Saved model to price_lstm_trader.pt")


if __name__ == "__main__":
    main()
