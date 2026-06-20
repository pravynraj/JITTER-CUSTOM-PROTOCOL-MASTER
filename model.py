import torch
import torch.nn as nn


class JitterLSTM(nn.Module):
    """
    LSTM-based binary classifier for jitter spike prediction.

    Input shape : (batch_size, seq_len, input_size)
    Output shape: (batch_size, 1)  — raw logit (use with BCEWithLogitsLoss)
    """

    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.3):
        """
        Args:
            input_size  : Number of features per time-step (default 5: time, rtt, delay, jitter, loss)
            hidden_size : Number of LSTM hidden units
            num_layers  : Number of stacked LSTM layers
            dropout     : Dropout between LSTM layers (ignored when num_layers == 1)
        """
        super(JitterLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # --- LSTM encoder ---
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,          # expects (batch, seq, feature)
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),            # single logit output
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        Returns: Tensor of shape (batch_size, 1) — raw logits
        """
        # LSTM forward — only take the last time-step's hidden state
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden  = lstm_out[:, -1, :]   # (batch, hidden_size)

        logits = self.classifier(last_hidden)  # (batch, 1)
        return logits
