"""
lstm_ae.py — LSTM autoencoder for sequence-based anomaly detection.
Architecture (small on purpose — CPU-friendly, hard to overfit on ~1700 windows):
    Input:    (batch, 30, 2)
    Encoder:  LSTM hidden=16, 1 layer, batch_first=True
    Latent:   final hidden state (batch, 16)
    Decoder:  LSTM hidden=16, fed latent for each of 30 timesteps
    Output:   Linear (16 -> 2), giving back (batch, 30, 2)
    Loss:     MSE between input and output windows

Training:
    Per ticker, 80/20 train/val split by date (no shuffling — temporal data).
    Adam optimizer, lr=1e-3. Early stopping on val loss with patience=5.
    Up to 50 epochs. Each ticker takes ~5-10 minutes on CPU.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

WINDOW_SIZE = 30
FEATURE_COLS = ["return", "volume_ratio"]


# ---------------- Data prep ----------------

def make_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Slice a per-ticker feature DataFrame into overlapping 30-day windows.

    Returns:
        X: array of shape (n_windows, window_size, n_features)
        end_dates: DatetimeIndex of the LAST day in each window (the day the window "represents")
    """
    # Drop rows where any feature is NaN (early days when rolling features aren't ready)
    df = df[FEATURE_COLS].dropna()

    if len(df) < window_size:
        raise ValueError(f"Not enough rows ({len(df)}) for window_size={window_size}")

    n_windows = len(df) - window_size + 1
    n_features = len(FEATURE_COLS)

    X = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    end_dates = []

    for i in range(n_windows):
        X[i] = df.iloc[i:i + window_size].values
        end_dates.append(df.index[i + window_size - 1])

    return X, pd.DatetimeIndex(end_dates)


def standardize(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Per-feature z-score using TRAIN-only statistics (no leakage from val).

    Returns scaled X_train, scaled X_val, and a stats dict to use later.
    """
    # Compute mean and std across all windows × timesteps, per feature
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return X_train_s.astype(np.float32), X_val_s.astype(np.float32), stats


# ---------------- Model ----------------

class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, n_features: int = 2, hidden_size: int = 16, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_size, n_features)
        returns: reconstruction of same shape
        """
        # Encode
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch, hidden)
        latent = h_n.squeeze(0)         # (batch, hidden)

        # Repeat latent for each timestep to feed into decoder
        decoder_input = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        # decoder_input: (batch, window_size, hidden)

        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        # decoder_output: (batch, window_size, hidden)

        # Project back to feature space
        out = self.output_layer(decoder_output)  # (batch, window_size, n_features)
        return out


# ---------------- Training ----------------

def train_one_ticker(
    ticker: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    val_frac: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Train an LSTM AE on one ticker. Save the model + scaling stats to disk."""
    csv_path = PROC_DIR / f"{ticker}.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    X, end_dates = make_windows(df)

    # Temporal split: first 80% train, last 20% val
    n_train = int(len(X) * (1 - val_frac))
    X_train_raw, X_val_raw = X[:n_train], X[n_train:]
    train_dates, val_dates = end_dates[:n_train], end_dates[n_train:]

    if verbose:
        print(f"\n[{ticker}] windows: total={len(X)}, train={len(X_train_raw)}, val={len(X_val_raw)}")
        print(f"[{ticker}] train dates: {train_dates[0].date()} -> {train_dates[-1].date()}")
        print(f"[{ticker}] val dates:   {val_dates[0].date()} -> {val_dates[-1].date()}")

    # Standardize using train stats only
    X_train, X_val, stats = standardize(X_train_raw, X_val_raw)

    # Wrap in DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(X_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    torch.manual_seed(42)
    model = LSTMAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping on val loss
    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        for x, _ in train_loader:
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
        train_loss = train_loss_sum / len(train_ds)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                recon = model(x)
                loss = criterion(recon, x)
                val_loss_sum += loss.item() * x.size(0)
        val_loss = val_loss_sum / len(val_ds)
        val_losses.append(val_loss)

        if verbose:
            print(f"[{ticker}] epoch {epoch + 1:2d}: train={train_loss:.5f}  val={val_loss:.5f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[{ticker}] early stopping at epoch {epoch + 1} (best val={best_val:.5f})")
                break

    # Load best state and save
    if best_state is not None:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"lstm_ae_{ticker}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "stats": stats,
        "config": {
            "window_size": WINDOW_SIZE,
            "feature_cols": FEATURE_COLS,
            "hidden_size": model.hidden_size,
        },
    }, model_path)

    return {
        "ticker": ticker,
        "n_train_windows": len(X_train),
        "n_val_windows": len(X_val),
        "best_val_loss": best_val,
        "epochs_trained": len(train_losses),
        "model_path": str(model_path),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ---------------- Inference (Day 9) ----------------

def load_trained_model(ticker: str) -> tuple[LSTMAutoencoder, dict, dict]:
    """Load a saved per-ticker model. Returns (model in eval mode, stats, config)."""
    model_path = MODELS_DIR / f"lstm_ae_{ticker}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model for {ticker} at {model_path}. "
            f"Run `python -m src.methods.lstm_ae` (Day 8 training) first."
        )
    checkpoint = torch.load(model_path, weights_only=False)
    hidden_size = checkpoint["config"]["hidden_size"]
    model = LSTMAutoencoder(hidden_size=hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["stats"], checkpoint["config"]


def compute_window_errors(model: LSTMAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-window reconstruction MSE. Input X already standardized. Returns 1D array of length n_windows."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X.astype(np.float32))
        recon = model(x_tensor).numpy()
    # MSE per window: mean across (timesteps × features)
    return ((X - recon) ** 2).mean(axis=(1, 2))


def lstm_ae_flags(
    features: pd.DataFrame,
    threshold_k: float = 2.5,
    val_frac: float = 0.2,
) -> pd.Series:
    """
    Flag anomalies using a pre-trained LSTM autoencoder.

    Strategy:
        1. Load the model trained for THIS ticker (Day 8 output).
        2. Slice features into 30-day windows, standardize using saved train stats.
        3. Compute reconstruction MSE for each window.
        4. Threshold = mean_train_error + threshold_k * std_train_error.
        5. Flag the END date of any window whose error > threshold.

    Args:
        features: per-ticker feature DataFrame (must have 'return' and 'volume_ratio').
                  Per-ticker because evaluate.py calls us once per ticker.
        threshold_k: how many SDs above train-mean to count as anomaly.
        val_frac: must match training value (default 0.2). Used to split errors into
                  train_errors (for threshold calibration) and full errors (for flagging).

    Returns:
        pd.Series of bool indexed by features.index. True where the
        END date of the window had reconstruction error above the threshold.

    Notes on the design choice (see NOTES.md Day 9):
        We flag the END date of high-error windows ("the day that broke the pattern"),
        not the middle day and not the per-timestep maximum. This matches the
        sequence-level framing — we're not detecting point outliers, we're detecting
        windows that don't look like training patterns.
    """
    # Identify ticker from the DataFrame's filename context — fall back: detect by trying each model.
    # Cleaner: evaluate.py knows the ticker, but our interface contract is (df) -> series.
    # Solution: store the ticker on the dataframe as df.attrs, set by evaluate.py.
    ticker = features.attrs.get("ticker")
    if ticker is None:
        raise ValueError(
            "lstm_ae_flags requires features.attrs['ticker'] to be set. "
            "evaluate.py needs to set this when loading per-ticker data."
        )

    # Load trained model + its saved stats
    model, stats, config = load_trained_model(ticker)
    window_size = config["window_size"]
    feature_cols = config["feature_cols"]
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    # Build windows
    X_raw, end_dates = make_windows(features, window_size=window_size)
    X = (X_raw - mean) / std

    # Compute per-window errors
    all_errors = compute_window_errors(model, X)

    # Calibrate threshold on the TRAIN portion only (same split as training)
    n_train = int(len(all_errors) * (1 - val_frac))
    train_errors = all_errors[:n_train]
    threshold = train_errors.mean() + threshold_k * train_errors.std()

    # Flag end dates of high-error windows
    high_error_mask = all_errors > threshold

    flags = pd.Series(False, index=features.index)
    for is_high, end_date in zip(high_error_mask, end_dates):
        if is_high and end_date in flags.index:
            flags.loc[end_date] = True

    return flags


def lstm_ae_flags_pct(
    features: pd.DataFrame,
    flag_rate: float = 0.02,
    use_full_window: bool = True,
) -> pd.Series:
    """
    Variant of lstm_ae_flags that uses a PERCENTILE threshold instead of mean+k*std.

    Rationale: train-period errors may be systematically smaller than test-period
    errors (different volatility regimes), so a threshold calibrated on train.mean+k*std
    can over-flag the test period. Percentile thresholding is robust to that.

    Args:
        features: per-ticker DataFrame with features.attrs['ticker'] set.
        flag_rate: target fraction of windows to flag (mirror's IF's contamination).
                   0.02 = flag top 2% most anomalous windows.
        use_full_window: if True, calibrate threshold using errors on BOTH train and
                         val windows together. False = train errors only (matches
                         lstm_ae_flags behavior, exposes the regime-shift problem).
    """
    ticker = features.attrs.get("ticker")
    if ticker is None:
        raise ValueError("features.attrs['ticker'] must be set")

    model, stats, config = load_trained_model(ticker)
    window_size = config["window_size"]
    feature_cols = config["feature_cols"]
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    X_raw, end_dates = make_windows(features, window_size=window_size)
    X = (X_raw - mean) / std
    all_errors = compute_window_errors(model, X)

    # Calibrate threshold from errors
    if use_full_window:
        # Use all errors — robust to train/test regime shift
        threshold = np.quantile(all_errors, 1 - flag_rate)
    else:
        # Use only train errors — original (broken) behavior
        n_train = int(len(all_errors) * 0.8)
        threshold = np.quantile(all_errors[:n_train], 1 - flag_rate)

    high_error_mask = all_errors > threshold

    flags = pd.Series(False, index=features.index)
    for is_high, end_date in zip(high_error_mask, end_dates):
        if is_high and end_date in flags.index:
            flags.loc[end_date] = True

    return flags


# ---------------- Smoke test ----------------

if __name__ == "__main__":
    import time
    TICKERS = ["AAPL", "AMC", "GME", "NVDA", "TSLA"]
    summary = []
    for ticker in TICKERS:
        t0 = time.time()
        result = train_one_ticker(ticker, epochs=50, verbose=False)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        summary.append(result)
        print(f"{ticker:6s}  best_val={result['best_val_loss']:.4f}  "
              f"epochs={result['epochs_trained']:3d}  time={elapsed:5.1f}s  "
              f"reduction={100*(1 - result['val_losses'][-1]/result['val_losses'][0]):.1f}%")

    print(f"\nTotal time: {sum(r['elapsed_s'] for r in summary):.1f}s")
    print(f"Models saved to: {MODELS_DIR}/")