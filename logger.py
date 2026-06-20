import csv
import threading
import os

class MetricsLogger:
    """
    Handles logging of metrics to a CSV file in a thread-safe manner.
    """
    def __init__(self, filepath="metrics.csv"):
        self.filepath = filepath
        self.lock = threading.Lock()
        
        # Initialize the CSV file and write the header
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['time', 'seq', 'rtt', 'delay', 'jitter', 'loss'])
        self.file.flush()

    def log(self, time_val, seq, rtt, delay, jitter, loss):
        """
        Logs a single row of metrics.
        - time_val: current timestamp (seconds since epoch)
        - seq: sequence number
        - rtt: Round Trip Time in ms
        - delay: inter-arrival delay in ms
        - jitter: difference between consecutive delays in ms
        - loss: 1 if lost, 0 otherwise
        """
        # Format metrics to 3 decimal places; leave blank if packet was lost
        rtt_str = f"{rtt:.3f}" if not loss else ""
        delay_str = f"{delay:.3f}" if not loss else ""
        jitter_str = f"{jitter:.3f}" if not loss else ""
        
        loss_val = 1 if loss else 0
        
        with self.lock:
            self.writer.writerow([
                f"{time_val:.6f}",
                seq,
                rtt_str,
                delay_str,
                jitter_str,
                loss_val
            ])
            self.file.flush()

    def close(self):
        with self.lock:
            if not self.file.closed:
                self.file.close()
