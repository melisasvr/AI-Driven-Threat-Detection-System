#!/usr/bin/env python3
"""
anomaly_detection_with_graphs.py
Improved anomaly detection demo:
 - Synthetic dataset (or load your own CSV by uncommenting)
 - Models: IsolationForest, LocalOutlierFactor, OneClassSVM (optional Autoencoder)
 - Plots: PCA scatter, anomaly score histograms, confusion matrices, ROC curves
 - Optional: template for live packet capture with scapy (commented)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
import warnings
warnings.filterwarnings("ignore")

# Optional autoencoder
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except Exception:
    HAS_TF = False

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_synthetic_dataset(n_normal=1000, n_anom=100, n_features=20, seed=42):
    np.random.seed(seed)
    normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    anomalous = np.random.normal(loc=5.0, scale=1.0, size=(n_anom, n_features))
    X = np.vstack([normal, anomalous])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anom)])  # 0 normal, 1 anomaly
    return pd.DataFrame(X), pd.Series(y, name="label")

def load_csv_dataset(path, label_col="label"):
    """
    Load a CSV file. It must contain a label column with 0 = normal, 1 = anomaly.
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")
    y = df[label_col]
    X = df.drop(columns=[label_col])
    return X, y

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def train_isolation_forest(X_train, contamination=0.1, random_state=42):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    model.fit(X_train)
    return model

def train_lof(X_train, contamination=0.1, n_neighbors=20):
    # Note: LOF is unsupervised but sklearn API returns -1/1 for fit_predict
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(X_train)  # with novelty=True we can call predict on new data
    return lof

def train_ocsvm(X_train, nu=0.1, kernel="rbf"):
    oc = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
    oc.fit(X_train)
    return oc

def build_autoencoder(input_dim, encoding_dim=8):
    # Simple dense autoencoder (only if Keras available)
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(int(input_dim/2), encoding_dim*2), activation="relu")(inp)
    x = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(max(int(input_dim/2), encoding_dim*2), activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    ae = models.Model(inputs=inp, outputs=out)
    ae.compile(optimizer="adam", loss="mse")
    return ae

def autoencoder_train_predict(X_train, X_test, threshold=None, epochs=50, batch_size=32):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed; autoencoder option not available.")
    ae = build_autoencoder(X_train.shape[1], encoding_dim=max(4, X_train.shape[1]//4))
    ae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    recon = ae.predict(X_test)
    mse = np.mean(np.square(X_test - recon), axis=1)
    if threshold is None:
        # estimate threshold from train reconstructions (e.g., mean + 3*std)
        recon_train = ae.predict(X_train)
        mse_train = np.mean(np.square(X_train - recon_train), axis=1)
        threshold = np.mean(mse_train) + 3 * np.std(mse_train)
    preds = np.where(mse > threshold, 1, 0)
    return preds, mse, threshold

def model_scores_to_labels(model, X, model_name="iforest"):
    """
    For each model, produce anomaly score and predicted labels in canonical form:
     - scores: higher -> more normal for some models, so standardize sign when needed
     - labels: 1 = anomaly, 0 = normal
    """
    if model_name == "iforest":
        # decision_function: higher -> more normal; predict: -1 anomaly, 1 normal
        scores = model.decision_function(X)  # higher == more normal
        raw_pred = model.predict(X)
        labels = np.where(raw_pred == -1, 1, 0)
        # Convert to anomaly score where higher -> more anomalous
        anomaly_score = -scores
    elif model_name == "lof":
        # For LOF with novelty=True, use score_samples: higher -> more normal
        scores = model.score_samples(X)
        raw_pred = model.predict(X)
        labels = np.where(raw_pred == -1, 1, 0)
        anomaly_score = -scores
    elif model_name == "ocsvm":
        scores = model.score_samples(X)  # higher -> more normal
        raw_pred = model.predict(X)
        labels = np.where(raw_pred == -1, 1, 0)
        anomaly_score = -scores
    else:
        raise ValueError("Unknown model_name")
    return labels, anomaly_score

def evaluate_and_save_plots(X_train, X_test, y_train, y_test, models_dict):
    """
    models_dict: {"iforest":model_instance, "lof":..., "ocsvm":...}
    Produces: PCA scatter, histogram of scores, confusion matrices, ROC plot.
    """
    # Precompute PCA for plotting
    pca = PCA(n_components=2)
    X_full = np.vstack([X_train, X_test])
    pca.fit(X_full)
    X_test_2d = pca.transform(X_test)
    results = {}

    # For ROC
    plt.figure(figsize=(8,6))
    plt.title("ROC Curves")
    fpr_tpr_present = False

    for name, model in models_dict.items():
        if name == "autoencoder":
            # handled separately if present
            preds_ae, scores_ae, th = model  # stored tuple (preds, scores, thresh)
            y_pred = preds_ae
            anomaly_score = scores_ae  # higher -> more anomalous (MSE)
        else:
            y_pred, anomaly_score = model_scores_to_labels(model, X_test, model_name=name)

        # Save results for summary
        results[name] = {"y_pred": y_pred, "score": anomaly_score}

        # 1) PCA scatter (color by predicted label; marker by ground truth)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X_test_2d[:,0], y=X_test_2d[:,1],
                        hue=y_pred, style=y_test,
                        palette={0:"tab:blue", 1:"tab:red"},
                        markers={0:"o",1:"X"}, s=40, alpha=0.7)
        plt.title(f"PCA Scatter — predictions by {name}")
        plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
        plt.legend(title="pred / true")
        fname = os.path.join(OUTPUT_DIR, f"pca_scatter_{name}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

        # 2) Score histogram
        plt.figure(figsize=(8,5))
        plt.hist(anomaly_score[y_test==0], bins=50, alpha=0.6, label="normal")
        plt.hist(anomaly_score[y_test==1], bins=50, alpha=0.6, label="anomaly")
        plt.title(f"Anomaly Score Distribution — {name}")
        plt.xlabel("Anomaly score (higher -> more anomalous)")
        plt.ylabel("Frequency")
        plt.legend()
        fname = os.path.join(OUTPUT_DIR, f"score_hist_{name}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

        # 3) Confusion matrix & classification report
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {name}")
        fname = os.path.join(OUTPUT_DIR, f"confusion_{name}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"=== Model: {name} ===")
        print(classification_report(y_test, y_pred, digits=4))
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}\n")

        # 4) ROC (works because we have true labels)
        try:
            fpr, tpr, _ = roc_curve(y_test, anomaly_score)
            roc_auc = auc(fpr, tpr)
            plt.figure(1)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            fpr_tpr_present = True
        except Exception as e:
            print(f"Could not compute ROC for {name}: {e}")

    if fpr_tpr_present:
        plt.figure(1)
        plt.plot([0,1], [0,1], "k--", linewidth=0.8)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC — Model Comparison")
        plt.legend(loc="lower right")
        fname = os.path.join(OUTPUT_DIR, "roc_comparison.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"All plots saved in folder: {OUTPUT_DIR}")
    return results

def main():
    # --- Load or create dataset ---
    # Option A: synthetic
    X_df, y_series = create_synthetic_dataset(n_normal=1000, n_anom=150, n_features=20, seed=42)

    # Option B: load CSV with label column 'label'
    # X_df, y_series = load_csv_dataset("path_to_your_dataset.csv", label_col="label")

    # --- Train-test split ---
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42, stratify=y_series)

    # --- Preprocess ---
    X_train_s, X_test_s, scaler = preprocess(X_train_df, X_test_df)

    # --- Contamination estimate: use proportion of anomalies in training labels (if labels known)
    # In real unlabeled use-case, set this based on domain knowledge.
    prop_anom_in_train = np.clip(y_train.mean(), 0.001, 0.5)
    contamination = float(prop_anom_in_train)  # e.g., 0.1

    # --- Train models ---
    print("Training IsolationForest...")
    iforest = train_isolation_forest(X_train_s, contamination=contamination)

    print("Training LocalOutlierFactor (novelty=True)...")
    lof = train_lof(X_train_s, contamination=contamination, n_neighbors=20)

    print("Training OneClassSVM...")
    ocsvm = train_ocsvm(X_train_s, nu=contamination, kernel="rbf")

    models = {"iforest": iforest, "lof": lof, "ocsvm": ocsvm}

    # Optional Autoencoder
    if HAS_TF:
        print("Training Autoencoder (this may take time)...")
        try:
            preds_ae, scores_ae, th = autoencoder_train_predict(X_train_s, X_test_s, threshold=None, epochs=60, batch_size=64)
            models["autoencoder"] = (preds_ae, scores_ae, th)
            print(f"Autoencoder threshold={th:.5f}")
        except Exception as e:
            print("Autoencoder training failed:", e)

    # --- Evaluate & plot ---
    results = evaluate_and_save_plots(X_train_s, X_test_s, y_train.values, y_test.values, models)

    # Example: access results for a model
    # results["iforest"]["y_pred"], results["iforest"]["score"]

    # --- Optional: Live packet capture template (commented) ---

    def extract_features_from_packet(pkt):
        # build a feature vector compatible with your trained model (must match training features)
        feat = []
        try:
            feat.append(len(pkt))  # packet length
            if pkt.haslayer("IP"):
                feat.append(pkt["IP"].ttl)
                feat.append(pkt["IP"].len if hasattr(pkt["IP"], "len") else 0)
            # add more features: protocol, flags, payload entropy etc.
        except Exception:
            pass
        return np.array(feat)

    live_buffer = []
    def handle_packet(pkt):
        fv = extract_features_from_packet(pkt)
        if fv.shape[0] == X_train_df.shape[1]:
            fv_scaled = scaler.transform(fv.reshape(1,-1))
            score = iforest.decision_function(fv_scaled)
            label = iforest.predict(fv_scaled)
            if label == -1:
                print("ANOMALY detected (live)!")
            # append to live_buffer for plotting
            live_buffer.append((-score).item())

    # sniff(prn=handle_packet, store=False)

if __name__ == "__main__":
    main()
