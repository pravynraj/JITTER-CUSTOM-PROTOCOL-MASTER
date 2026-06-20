import csv
import socket
import struct
import time
import random
import threading
import torch
import numpy as np
from collections import deque
from model import JitterLSTM

REALTIME_LOG = "realtime_log.csv"

# ── Constants ──────────────────────────────────────────────────
SERVER_IP      = "127.0.0.1"
SERVER_PORT    = 5005
PACKET_FMT     = "!Id"          # unsigned int (seq) + double (timestamp)
TIMEOUT        = 0.5            # ACK wait timeout in seconds

NORMAL_INTERVAL = 0.020         # 20ms — normal sending rate
SLOW_INTERVAL   = 0.050         # 50ms — throttled rate when spike predicted


# ── Model Loading ──────────────────────────────────────────────
model = JitterLSTM()
model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
model.eval()

print("Model loaded successfully")


# ── Sliding Window Buffer ──────────────────────────────────────
# Each entry: [time, rtt, delay, jitter, loss]  — 5 features to match training data
window = deque(maxlen=20)

for _ in range(20):
    window.append([0.0, 0.0, 0.0, 0.0, 0.0])

print("Window initialized:", len(window))


# ── Real-time Prediction ───────────────────────────────────────
def predict_spike(window):
    """
    Runs the LSTM model on the current sliding window.

    Args:
        window: deque of 20 packets, each a list of 5 features:
                [time, rtt, delay, jitter, loss]
    Returns:
        1 if a jitter spike is predicted, 0 otherwise
    """
    # Convert deque → numpy (20, 5) → tensor (1, 20, 5)
    data   = np.array(window, dtype=np.float32)
    tensor = torch.tensor(data).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)              # raw logit (1, 1)

    prob = torch.sigmoid(output).item()
    print(f"  Predicted probability: {prob:.4f}")

    # Rule-based safety override: flag spike if latest jitter already exceeds 50ms
    latest_jitter = window[-1][3]

    # Normalize jitter: cap at 150ms ceiling so extreme spikes don't dominate unbounded
    jitter_norm = min(latest_jitter / 150, 1.0)

    # Weighted hybrid score: 60% ML confidence + 40% normalized real-time jitter
    score = 0.6 * prob + 0.4 * jitter_norm
    print(f"  Hybrid score:          {score:.4f} (jitter_norm={jitter_norm:.2f})")

    # Smooth threshold: score > 0.26 triggers throttle
    result = 1 if score > 0.26 else 0
    return result, prob   # return prob so callers can log it


# ── Adaptive UDP Sender ────────────────────────────────────────
class RealtimeSender:
    """
    AI-driven adaptive UDP sender.
    - Sends packets to receiver.py
    - After each ACK, updates the sliding window and runs inference
    - Slows down transmission when a jitter spike is predicted
    """

    def __init__(self, ip=SERVER_IP, port=SERVER_PORT):
        self.target  = (ip, port)
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(TIMEOUT)

        self.seq           = 0
        self.running       = False
        self.lock          = threading.Lock()

        # Per-packet state for metric computation
        self.prev_recv_time = None
        self.prev_delay     = 0.0

        # Pending packets awaiting ACK: seq -> send_time
        self.unacked = {}

        # ── CSV log setup ──────────────────────────────────────
        self._log_file = open(REALTIME_LOG, "w", newline="")
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(["timestamp", "seq", "rtt", "delay",
                             "jitter", "predicted_prob", "prediction"])
        self._log_file.flush()

    # ── Send Loop (separate thread) ────────────────────────────
    def _send_loop(self, interval_ref):
        """Transmits packets at the current adaptive interval."""
        while self.running:
            t_send = time.time()
            packet = struct.pack(PACKET_FMT, self.seq, t_send)

            with self.lock:
                self.unacked[self.seq] = t_send

            try:
                self.sock.sendto(packet, self.target)
            except Exception as e:
                if self.running:
                    print(f"  Send error: {e}")

            self.seq += 1

            # Respect current adaptive interval
            elapsed = time.time() - t_send
            sleep_t = interval_ref[0] - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Main Loop (receive + predict + adapt) ──────────────────
    def run(self):
        self.running = True

        # interval_ref is a mutable list so the send thread picks up changes live
        interval_ref = [NORMAL_INTERVAL]

        send_thread = threading.Thread(
            target=self._send_loop, args=(interval_ref,), daemon=True
        )
        send_thread.start()

        print(f"\nSending to {self.target[0]}:{self.target[1]} — press Ctrl+C to stop.\n")

        try:
            while self.running:
                try:
                    data, _ = self.sock.recvfrom(1024)
                    recv_time = time.time()

                    if len(data) != struct.calcsize(PACKET_FMT):
                        continue

                    seq, send_time = struct.unpack(PACKET_FMT, data)

                    with self.lock:
                        if seq not in self.unacked:
                            continue        # duplicate or already expired
                        del self.unacked[seq]

                    # ── Compute metrics ────────────────────────
                    rtt_ms    = (recv_time - send_time) * 1000.0
                    delay_ms  = 0.0
                    if self.prev_recv_time is not None:
                        delay_ms = (recv_time - self.prev_recv_time) * 1000.0
                    jitter_ms = abs(delay_ms - self.prev_delay)

                    self.prev_recv_time = recv_time
                    self.prev_delay     = delay_ms

                    t_norm = recv_time - 1.7e9   # keep value small / in range

                    # ── Update sliding window ──────────────────
                    window.append([t_norm, rtt_ms, delay_ms, jitter_ms, 0.0])

                    print(f"[seq={seq:>5}] RTT={rtt_ms:>7.1f}ms  "
                          f"delay={delay_ms:>7.1f}ms  jitter={jitter_ms:>7.1f}ms")

                    # ── AI Prediction ──────────────────────────
                    # Capture prob separately for logging
                    result, prob = predict_spike(window)

                    # ── Log to CSV ─────────────────────────────
                    self._csv.writerow([
                        f"{recv_time:.6f}", seq,
                        f"{rtt_ms:.3f}", f"{delay_ms:.3f}",
                        f"{jitter_ms:.3f}", f"{prob:.4f}", result
                    ])
                    self._log_file.flush()

                    if result == 1:
                        interval_ref[0] = SLOW_INTERVAL
                        print("  WARNING Spike predicted -> slowing down "
                              f"(interval={SLOW_INTERVAL*1000:.0f}ms)\n")
                    else:
                        interval_ref[0] = NORMAL_INTERVAL
                        print("  OK Normal network "
                              f"(interval={NORMAL_INTERVAL*1000:.0f}ms)\n")

                except socket.timeout:
                    # Mark long-overdue packets as lost (simple check)
                    now = time.time()
                    with self.lock:
                        expired = [s for s, t in self.unacked.items()
                                   if now - t > TIMEOUT]
                        for s in expired:
                            del self.unacked[s]
                    continue

                except Exception as e:
                    if self.running:
                        print(f"  Receive error: {e}")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.sock.close()
        self._log_file.close()
        print(f"\nSender stopped. Log saved to {REALTIME_LOG}")


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    sender = RealtimeSender()
    sender.run()
