import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#config section 
@dataclass
class Config:
    csv_path: str = "data/eurusd_hourly.csv"
    time_column: str = "time"
    close_column: str = "close"

    sequence_length: int = 24
    horizon: int = 1
    max_return: float = 0.01

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 0.001
    hidden_size: int = 64
    num_layers: int = 2


cfg = Config()

class PriceDataset(Dataset):
    #Input: sequence of normalized close prices
    #Target: normalised 1 step ahead returns
    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.cfg = cfg

        close = df[cfg.close_column].astype(float).values
        self.close = close

        #normalize close
        close_norm = (close - close.mean()) / (close.std() + 1e-8)
        self.features = close_norm.reshape(-1,1)

        self.sequence_length = cfg.sequence_length
        self.horizon = cfg.horizon
        self.max_return = cfg.max_return

        #last index usable for prediction
        self.max_t = len(close) - self.horizon - 1
        if self.max_t <= self.sequence_length:
            raise ValueError("Not enough data to build sequence")
        
    def __len__(self):
        return self.max_t - self.sequence_length + 1
    
    def __getitem__(self, index):
        t = index + self.sequence_length - 1
        start = t - self.sequence_length + 1
        end = t + 1

        x_sequence = self.features[start:end]

        p_t = self.close[t]
        p_future = self.close[t + self.horizon]

        raw_return = (p_future - p_t) / p_t
        normalized_return = np.clip(raw_return / self.max_return, -1.0, 1.0)
        return torch.tensor(x_sequence, dtype=torch.float32), torch.tensor(normalized_return, dtype=torch.float32)

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
        self.out_activation = nn.Tanh()  #normalize to [-1, 1]

    def forward(self, x):
        #x meaning: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]              #(batch, hidden_size)
        out = self.fc(h_last)         #(batch, 1)
        out = self.out_activation(out)
        return out.squeeze(-1)        #(batch,)


def load_csv(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    #Checks data
    for col in [cfg.time_column, cfg.close_column]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in CSV. Found: {list(df.columns)}")

    #sort by time
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
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    correct_sign = 0
    total_sign = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        sign_pred = torch.sign(preds)
        sign_true = torch.sign(y)
        correct_sign += torch.sum(sign_pred == sign_true).item()
        total_sign += y.numel()

    avg_loss = total_loss / n
    sign_acc = correct_sign / total_sign
    return avg_loss, sign_acc


def main():
    print(f"Using device: {device}")

    df = load_csv(cfg)
    dataset = PriceDataset(df, cfg)
    print("Dataset length:", len(dataset))

    # ----- time-based split -----
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

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ----- model setup -----
    input_dim = 1  # only close price
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
        val_loss, val_sign_acc = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_sign_acc={val_sign_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_sign_acc = eval_epoch(model, test_loader, criterion)
    print(f"TEST | loss={test_loss:.6f} | sign_acc={test_sign_acc:.3f}")

    torch.save(model.state_dict(), "price_lstm_regressor.pt")
    print("Saved model to price_lstm_regressor.pt")

if __name__ == "__main__":
    main()