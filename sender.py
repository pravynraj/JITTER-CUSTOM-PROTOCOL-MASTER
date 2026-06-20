import socket
import struct
import time
import threading
from logger import MetricsLogger

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005
PACKET_FMT = "!Id"
SEND_INTERVAL = 0.02  # 20ms
TIMEOUT = 1.0         # 1 second timeout for packet loss

class Sender:
    """
    UDP Sender that transmits packets every 20ms, receives ACKs, and logs metrics.
    """
    def __init__(self, target_ip=SERVER_IP, target_port=SERVER_PORT):
        self.target_address = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        
        self.logger = MetricsLogger("sender_log.csv")
        self.unacked = {}  # Maps seq -> send_time
        
        self.lock = threading.Lock()
        self.running = False
        self.seq = 0
        
        # State variables for metrics calculation
        self.prev_recv_time = None
        self.prev_delay = 0.0

    def start(self):
        self.running = True
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.loss_thread = threading.Thread(target=self._check_loss_loop, daemon=True)
        
        self.send_thread.start()
        self.recv_thread.start()
        self.loss_thread.start()

    def stop(self):
        self.running = False
        # Give threads a moment to finish cleanly
        time.sleep(0.5)
        self.sock.close()
        self.logger.close()

    def _send_loop(self):
        """
        Transmits a packet exactly every SEND_INTERVAL seconds.
        """
        while self.running:
            send_time = time.time()
            packet = struct.pack(PACKET_FMT, self.seq, send_time)
            
            with self.lock:
                self.unacked[self.seq] = send_time
                
            try:
                self.sock.sendto(packet, self.target_address)
            except Exception as e:
                if self.running:
                    print(f"Send error: {e}")
                    
            self.seq += 1
            
            # Maintain the 20ms send interval
            elapsed = time.time() - send_time
            if elapsed < SEND_INTERVAL:
                time.sleep(SEND_INTERVAL - elapsed)

    def _recv_loop(self):
        """
        Listens for ACKs and calculates RTT, delay (inter-arrival), and jitter.
        """
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                recv_time = time.time()
                
                if len(data) == struct.calcsize(PACKET_FMT):
                    seq, send_time = struct.unpack(PACKET_FMT, data)
                    
                    with self.lock:
                        if seq in self.unacked:
                            del self.unacked[seq]
                        else:
                            # Packet is a duplicate or was already marked lost
                            continue
                            
                    # Calculate Metrics
                    rtt_ms = (recv_time - send_time) * 1000.0
                    
                    # Inter-arrival delay: difference in receive times between consecutive ACKs
                    delay_ms = 0.0
                    if self.prev_recv_time is not None:
                        delay_ms = (recv_time - self.prev_recv_time) * 1000.0
                        
                    # Jitter: absolute difference between consecutive inter-arrival delays
                    jitter_ms = abs(delay_ms - self.prev_delay)
                    
                    # Update state
                    self.prev_recv_time = recv_time
                    self.prev_delay = delay_ms
                    
                    self.logger.log(
                        time_val=recv_time,
                        seq=seq,
                        rtt=rtt_ms,
                        delay=delay_ms,
                        jitter=jitter_ms,
                        loss=0
                    )
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")

    def _check_loss_loop(self):
        """
        Periodically checks for packets that haven't been ACKed within the TIMEOUT.
        """
        while self.running:
            time.sleep(0.5)
            curr_time = time.time()
            lost_seqs = []
            
            with self.lock:
                for seq, send_time in list(self.unacked.items()):
                    if curr_time - send_time > TIMEOUT:
                        lost_seqs.append(seq)
                        del self.unacked[seq]
            
            # Log lost packets
            for seq in lost_seqs:
                self.logger.log(
                    time_val=curr_time,
                    seq=seq,
                    rtt=0.0,
                    delay=0.0,
                    jitter=0.0,
                    loss=1
                )

if __name__ == "__main__":
    sender = Sender()
    try:
        print(f"Starting sender. Sending to {SERVER_IP}:{SERVER_PORT}")
        print("Press Ctrl+C to stop.")
        sender.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping sender...")
        sender.stop()
