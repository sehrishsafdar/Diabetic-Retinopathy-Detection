# 👁️ Federated Learning for Diabetic Retinopathy Detection using Vision Transformer (ViT)

This repository implements a Federated Learning system for **Diabetic Retinopathy Detection** using Vision Transformer (ViT) models. The system allows multiple clients to collaboratively train a global model without sharing sensitive patient data, thus preserving privacy while achieving high diagnostic accuracy.

---

## 📁 Project Structure
├── client/ # Client-side training scripts
├── data/ # Dataset handling and preprocessing
├── models/ # Vision Transformer model definitions
├── server/ # Server-side aggregation logic
├── evaluate_global_model.py # Evaluates the global model on test data
├── plot_metrics.py # Plots accuracy, loss, ROC curve etc.
├── requirements.txt # List of dependencies
├── train.py # Main federated training loop
├── train_simple.py # Centralized (non-federated) training baseline
├── training_metrics.csv # Logs training metrics across rounds
└── README.md # Documentation


---

## 🧠 Overview

This project simulates a **federated learning environment** to train a diabetic retinopathy classifier using retinal fundus images. Multiple clients (e.g., hospitals or clinics) train the model on their local data, and a central server aggregates their model weights to form a **global Vision Transformer model**.

### ⚙️ System Architecture

- Each **client** trains a Vision Transformer on local data.
- The **server** collects model weights and aggregates using **FedAvg**.
- The **global model** is evaluated after each round on a central test set.

---

## ✨ Key Features

- 🔐 **Federated Learning** for privacy-preserving model training
- 🧠 **Vision Transformer (ViT)** backbone for image classification
- 📊 **Training Metrics Logging** – loss, accuracy, AUC, per round
- 📈 **Global Model Evaluation** on test data
- 📉 **Metrics Plotting** using built-in visualization script

---

## ⚙️ Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
#🔍 Main Dependencies:
torch

torchvision

timm

scikit-learn

pandas

matplotlib

seaborn

opencv-python

📚 Dataset
This project assumes the availability of preprocessed diabetic retinopathy datasets (e.g., APTOS, EyePACS, IDRiD).

You may place your dataset in the data/ directory following this structure:

python-repl
Copy
Edit
data/
├── client_1/
│   ├── images/
│   └── labels.csv
├── client_2/
│   ├── images/
│   └── labels.csv
...

🚀 How to Run
▶️ Centralized (Baseline) Training
python train_simple.py
🌐 Federated Training
python train.py
🧪 Evaluate Global Model
python evaluate_global_model.py
📈 Plot Training Metrics
python plot_metrics.py
🧪 client training
python train.py client 0
