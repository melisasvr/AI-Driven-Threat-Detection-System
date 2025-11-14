import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scapy.all import sniff, IP
from sklearn.ensemble import IsolationForest
import threading
import seaborn as sns

# Use seaborn theme
sns.set_theme(style="darkgrid")

# -----------------------------
# 1. TRAIN A SIMPLE MODEL
# -----------------------------
normal_data = np.random.normal(0, 1, size=(1000, 5))
abnormal_data = np.random.normal(4, 1, size=(100, 5))
X = np.vstack([normal_data, abnormal_data])

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# -----------------------------
# 2. PACKET FEATURE EXTRACTION
# -----------------------------
def extract_features(packet):
    try:
        pkt_len = len(packet)
        ttl = packet[IP].ttl if packet.haslayer(IP) else 0
        proto = packet[IP].proto if packet.haslayer(IP) else 0
        sport = packet.sport if hasattr(packet, "sport") else 0
        dport = packet.dport if hasattr(packet, "dport") else 0

        return np.array([pkt_len, ttl, proto, sport, dport])
    except:
        return np.array([0,0,0,0,0])

# -----------------------------
# 3. LIVE SCORE BUFFER
# -----------------------------
live_scores = []
lock = threading.Lock()

def handle_packet(packet):
    global live_scores
    feats = extract_features(packet).reshape(1, -1)
    score = -model.decision_function(feats)[0]  # higher = more anomalous

    with lock:
        live_scores.append(score)
        if len(live_scores) > 200:
            live_scores = live_scores[-200:]

# Start sniffing in background (daemon)
threading.Thread(target=lambda: sniff(prn=handle_packet, store=False),
                 daemon=True).start()

# -----------------------------
# 4. LIVE GRAPH
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

def animate(i):
    with lock:
        if not live_scores:
            return

        ax.clear()
        ax.plot(live_scores, marker="o")

        ax.set_title("Real-Time Network Anomaly Score")
        ax.set_xlabel("Packet #")
        ax.set_ylabel("Anomaly Score (Higher = Suspicious)")

        # Stable Y-axis limits
        recent_max = max(live_scores[-50:])
        ax.set_ylim(0, recent_max + 1)

ani = FuncAnimation(fig, animate, interval=500, cache_frame_data=False)

plt.tight_layout()
plt.show()
