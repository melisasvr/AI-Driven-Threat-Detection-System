# AI-Driven Threat Detection System

An AI-driven threat detection system that analyzes network traffic and identifies potential anomalies using machine learning. The project uses an Isolation Forest model to distinguish between normal and abnormal traffic patterns.

## Project Overview

This project demonstrates how machine learning can be applied to network security by generating synthetic traffic data, training an anomaly detection model, and evaluating suspicious activity within network patterns.

## Features

- Generates synthetic network traffic data for training and testing.
- Trains an Isolation Forest model for anomaly detection.
- Evaluates model performance using test data.
- Generates synthetic network packets and saves them as a `.pcap` file.
- Serves as a foundation for more advanced threat detection systems.

## Project Structure

- `ai_threat_detection.py`: Main script that trains and tests the anomaly detection model using synthetic network traffic data.
- `generate_synthetic_network_data.py`: Generates synthetic network traffic data for training and testing.
- `generate_synthetic_traffic.py`: Generates synthetic network packets and saves them in `.pcap` format.

## Prerequisites

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `scapy`

## Installation
- Install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn scapy
```

## Usage

### 1. Generate Synthetic Network Data

Run the following command to create the synthetic dataset:

```bash
python generate_synthetic_network_data.py
```

This will generate a CSV file named `synthetic_network_data.csv` containing both normal and abnormal traffic data.

### 2. Train and Test the Model

Train the Isolation Forest model and evaluate its performance:

```bash
python ai_threat_detection.py
```

- The script will output training and testing results, along with a classification report.

### 3. Generate Synthetic Network Packets

- Create synthetic network packets and save them as a `.pcap` file:

```bash
python generate_synthetic_traffic.py
```

- The generated `.pcap` file can be used for further analysis or for testing network monitoring tools.

## How It Works

### `ai_threat_detection.py`
- Generates synthetic normal and abnormal network traffic data.
- Splits the dataset into training and testing sets.
- Trains the Isolation Forest model on the training data.
- Evaluates the model on test data.
- Outputs performance metrics and classification results.

### `generate_synthetic_network_data.py`
- Creates synthetic network traffic data with both normal and abnormal patterns.
- Saves the generated data as a CSV file.

### `generate_synthetic_traffic.py`
- Generates synthetic network packets using random IP addresses and TCP ports.
- Saves the packets as a `.pcap` file for analysis.

## Potential Enhancements

- To make the project more robust and production-ready, consider the following improvements:

- Integrate real network traffic data for more realistic evaluation.
- Compare multiple machine learning models.
- Add feature engineering to improve detection quality.
- Build a real-time monitoring pipeline for live traffic analysis.
- Develop a GUI to visualize detection results.

## Author

**Melisa Sever**

- Project repository: [AI-Driven Threat Detection System](https://github.com/melisasvr/AI-Driven-Threat-Detection-System)

## Collaboration
- This project was originally started by Melisa Sever and later developed in collaboration with Saarthak Tripathi.
