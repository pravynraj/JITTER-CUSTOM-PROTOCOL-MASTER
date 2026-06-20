import pandas as pd
import matplotlib.pyplot as plt

# ── Load Data ──────────────────────────────────────────────────
df = pd.read_csv("realtime_log.csv")

# Validate required columns
required = ["timestamp", "rtt", "jitter", "predicted_prob", "prediction"]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in realtime_log.csv: {missing}")

print(f"Loaded {len(df)} rows from realtime_log.csv")

# ── Plot 1: RTT vs Time ────────────────────────────────────────
plt.figure()
plt.plot(df["timestamp"], df["rtt"], color="steelblue")
plt.title("RTT vs Time")
plt.xlabel("Time")
plt.ylabel("RTT (ms)")
plt.grid()

# ── Plot 2: Jitter vs Time ─────────────────────────────────────
plt.figure()
plt.plot(df["timestamp"], df["jitter"], color="darkorange")
plt.title("Jitter vs Time")
plt.xlabel("Time")
plt.ylabel("Jitter (ms)")
plt.grid()

# ── Plot 3: Spike Prediction vs Time ──────────────────────────
plt.figure()
plt.plot(df["timestamp"], df["prediction"], color="crimson", drawstyle="steps-post")
plt.title("Spike Prediction vs Time")
plt.xlabel("Time")
plt.ylabel("Prediction (0/1)")
plt.yticks([0, 1], ["Normal", "Spike"])
plt.grid()

# ── Plot 4: Predicted Probability vs Time ─────────────────────
plt.figure()
plt.plot(df["timestamp"], df["predicted_prob"], color="mediumseagreen")
plt.axhline(y=0.26, color="red", linestyle="--", linewidth=0.8, label="Threshold (0.26)")
plt.title("Predicted Probability vs Time")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.grid()

# ── Show All ───────────────────────────────────────────────────
plt.tight_layout()
plt.show()
