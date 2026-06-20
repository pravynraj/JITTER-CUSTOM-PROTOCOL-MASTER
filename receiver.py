import socket
import struct
import time
import random

SERVER_IP = "0.0.0.0"
SERVER_PORT = 5005
# Format: !Id -> Network byte order, Unsigned int (4 bytes) sequence, Double (8 bytes) timestamp
PACKET_FMT = "!Id"

class Receiver:
    """
    UDP Receiver that listens for packets and sends ACKs back.
    Includes network condition simulation (delay, jitter, loss, spikes).
    """
    def __init__(self, ip=SERVER_IP, port=SERVER_PORT):
        self.address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.address)
        self.running = False
        self.spike_mode = False
        self.spike_remaining = 0

    def start(self):
        self.running = True
        print(f"Receiver listening on {self.address[0]}:{self.address[1]}")
        try:
            while self.running:
                data, addr = self.sock.recvfrom(1024)
                
                # Check if the packet is the correct size
                if len(data) == struct.calcsize(PACKET_FMT):
                    # 1. Simulate Packet Loss (5-10% probability)
                    if random.random() < 0.07:
                        continue  # Drop packet, do not send ACK

                    # 2. Determine delay based on current mode
                    if self.spike_mode:
                        # --- SPIKE MODE ---
                        # Apply high delay (150–300ms) + wide jitter (±80ms)
                        base   = random.uniform(0.150, 0.300)
                        jitter = random.uniform(-0.080, 0.080)
                        delay  = max(0.0, base + jitter)   # clamp: never negative

                        # Decrement counter and check if spike episode is over
                        self.spike_remaining -= 1
                        if self.spike_remaining <= 0:
                            self.spike_mode = False
                            print("[SPIKE END] Returning to normal mode")
                    else:
                        # --- NORMAL MODE ---
                        # Apply base delay (40ms) + mild jitter (±20ms)
                        delay = max(0.0, 0.040 + random.uniform(-0.020, 0.020))

                        # Randomly trigger a new spike episode (10% probability)
                        if random.random() < 0.10:
                            self.spike_mode = True
                            self.spike_remaining = random.randint(5, 10)
                            print(f"[SPIKE START] Spike mode for {self.spike_remaining} packets")

                    # Apply the simulated network delay
                    if delay > 0:
                        time.sleep(delay)

                    # We echo the packet back as an ACK payload.
                    # This allows the sender to extract the seq and original send timestamp.
                    self.sock.sendto(data, addr)
                    
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            if self.running:
                print(f"Receiver error: {e}")

    def stop(self):
        self.running = False
        self.sock.close()
        print("Receiver stopped.")

if __name__ == "__main__":
    receiver = Receiver()
    receiver.start()
