import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import JitterLSTM

# ─────────────────────────── Config ───────────────────────────
X_PATH      = "X_dataset.npy"
Y_PATH      = "y_dataset.npy"
MODEL_PATH  = "model.pth"

SEQ_LEN     = 20
INPUT_SIZE  = 5         # time, rtt, delay, jitter, loss
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3

BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
TRAIN_RATIO = 0.80

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────── Data Loading ─────────────────────
def load_dataset(x_path, y_path):
    X = np.load(x_path).astype(np.float32)   # (N, 20, 5)
    y = np.load(y_path).astype(np.float32)   # (N,)

    print(f"\nLoaded X: {X.shape}  y: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Label {int(cls)}: {cnt} samples ({cnt/len(y)*100:.1f}%)")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)   # (N, 1) for BCEWithLogitsLoss

    return TensorDataset(X_t, y_t)


# ─────────────────────────── Class Weights ────────────────────
def compute_pos_weight(y_path):
    """Compute pos_weight = count(negatives) / count(positives) for BCEWithLogitsLoss."""
    y = np.load(y_path)
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    if n_pos == 0:
        raise ValueError("No positive samples found — collect more data with spike_mode active.")
    pw = n_neg / n_pos
    print(f"\nClass balance -> neg: {n_neg}, pos: {n_pos}  |  pos_weight: {pw:.2f}")
    return torch.tensor([pw], dtype=torch.float32).to(DEVICE)


# ─────────────────────────── Training Loop ────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)             # (batch, 1) raw logits
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

        # Accuracy: sigmoid > 0.5 → predicted positive
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total   += len(X_batch)

    return total_loss / total, correct / total


# ─────────────────────────── Evaluation Loop ──────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total   += len(X_batch)

            # Confusion matrix components for recall/precision
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return total_loss / total, correct / total, precision, recall, f1


# ─────────────────────────── Main ─────────────────────────────
def main():
    # 1. Load dataset
    dataset    = load_dataset(X_PATH, Y_PATH)
    pos_weight = compute_pos_weight(Y_PATH)

    # 2. Train / test split
    n_train = int(len(dataset) * TRAIN_RATIO)
    n_test  = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"\nTrain samples: {n_train}  |  Test samples: {n_test}")

    # 3. Build model
    model = JitterLSTM(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(DEVICE)
    print(f"\nModel architecture:\n{model}")

    # 4. Loss & Optimiser
    # BCEWithLogitsLoss with pos_weight handles class imbalance natively
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Optional: reduce LR on plateau to avoid stuck training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 5. Training loop
    print("\n" + "="*65)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'F1':>6}")
    print("="*65)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, prec, rec, f1 = evaluate(model, test_loader, criterion)

        scheduler.step(vl_loss)

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc*100:>8.2f}%  "
              f"{vl_loss:>8.4f}  {vl_acc*100:>6.2f}%  {f1:>6.4f}")

        # Save best checkpoint by F1 (handles imbalance better than accuracy)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_PATH)

    print("="*65)
    print(f"\nBest F1 on test set : {best_f1:.4f}")
    print(f"Model saved to      : {MODEL_PATH}")

    # 6. Final evaluation on the best saved model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, final_acc, final_prec, final_rec, final_f1 = evaluate(model, test_loader, criterion)
    print(f"\n-- Final Test Results --")
    print(f"  Accuracy  : {final_acc*100:.2f}%")
    print(f"  Precision : {final_prec:.4f}")
    print(f"  Recall    : {final_rec:.4f}")
    print(f"  F1 Score  : {final_f1:.4f}")


if __name__ == "__main__":
    main()
