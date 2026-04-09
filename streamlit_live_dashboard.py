import streamlit as st
import numpy as np
import pandas as pd
from scapy.all import sniff, IP
from sklearn.ensemble import IsolationForest
import threading, time

# -----------------------------
# TRAIN BASELINE MODEL
# -----------------------------
normal = np.random.normal(0, 1, (1000, 5))
abnormal = np.random.normal(4, 1, (100, 5))
X = np.vstack([normal, abnormal])

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# -----------------------------
# LIVE SCORE BUFFER
# -----------------------------
scores = []
packet_log = []

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(pkt):
    try:
        return np.array([
            len(pkt),
            pkt[IP].ttl if pkt.haslayer(IP) else 0,
            pkt[IP].proto if pkt.haslayer(IP) else 0,
            pkt.sport if hasattr(pkt, "sport") else 0,
            pkt.dport if hasattr(pkt, "dport") else 0
        ])
    except:
        return np.zeros(5)

# -----------------------------
# PACKET HANDLER
# -----------------------------
def handle_packet(pkt):
    feat = extract_features(pkt).reshape(1, -1)
    score = -model.decision_function(feat)[0]  # Higher = more anomaly

    scores.append(score)
    if len(scores) > 200:
        scores.pop(0)

    packet_log.append({
        "Length": feat[0][0],
        "TTL": feat[0][1],
        "Proto": feat[0][2],
        "Sport": feat[0][3],
        "Dport": feat[0][4],
        "Anomaly": score
    })

    if len(packet_log) > 200:
        packet_log.pop(0)

# -----------------------------
# START SNIFFER THREAD
# -----------------------------
threading.Thread(
    target=lambda: sniff(prn=handle_packet, store=False),
    daemon=True
).start()

# -----------------------------
# STREAMLIT DASHBOARD UI
# -----------------------------
st.set_page_config(page_title="Live Network Threat Detector", layout="wide")

st.title("🔴 Real-Time Network Anomaly Detection Dashboard")

col1, col2 = st.columns(2)

# Real-time anomaly plot
with col1:
    st.subheader("📈 Live Anomaly Score")
    graph_area = st.line_chart()

# Packet stats table
with col2:
    st.subheader("📋 Recent Packet Stats")
    table_area = st.empty()

st.markdown("---")
alert_box = st.empty()

# -----------------------------
# LIVE UPDATE LOOP
# -----------------------------
while True:
    if scores:
        graph_area.add_rows({"Anomaly Score": [scores[-1]]})

        # Alert if anomaly very high
        if scores[-1] > 4:
            alert_box.error(f"⚠ HIGH ANOMALY DETECTED: Score = {scores[-1]:.2f}")
        elif scores[-1] > 2:
            alert_box.warning(f"⚠ Suspicious Traffic: Score = {scores[-1]:.2f}")
        else:
            alert_box.success(f"Normal Traffic: Score = {scores[-1]:.2f}")

    # Update table
    if packet_log:
        df = pd.DataFrame(packet_log)
        table_area.dataframe(df, height=400)

    time.sleep(0.2)
