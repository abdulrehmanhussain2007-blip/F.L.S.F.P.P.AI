# Federated Learning for Privacy-Preserving Handwritten Character Recognition

## 📌 Project Overview

This project implements a **Federated Learning system using PyTorch** to train a global machine learning model across multiple simulated clients without directly sharing their raw training data.

Using the FEMNIST dataset, the system learns to recognize handwritten digits and letters from different writers. Each client trains a local model on its own data, while a central server aggregates local model updates using the **Federated Averaging (FedAvg)** algorithm.

The project demonstrates how collaborative machine learning can be performed while keeping training data decentralized.

## 🎯 Objectives

* Implement a federated learning workflow using PyTorch.
* Train local models on separate client datasets.
* Apply FedAvg to aggregate model parameters.
* Use FEMNIST to simulate realistic, writer-specific data distributions.
* Evaluate global model performance across federated communication rounds.
* Explore privacy challenges and limitations in distributed machine learning.

## 🧠 Technologies Used

* Python
* PyTorch
* FEMNIST Dataset
* NumPy
* Matplotlib
* Federated Averaging (FedAvg)

## 📂 Dataset

**FEMNIST — Federated Extended MNIST**

FEMNIST contains handwritten digits and letters, with data naturally associated with different writers.

Dataset: https://huggingface.co/datasets/flwrlabs/femnist

## ⚙️ System Architecture

1. **Client initialization:** Divide the training data among clients, preferably by writer.
2. **Local training:** Each client trains a copy of the global model on its local data.
3. **Model update transmission:** Clients send model parameters or updates to the server.
4. **Server aggregation:** FedAvg combines client models into a new global model.
5. **Global evaluation:** Evaluate the aggregated model and track performance across rounds.
6. **Iteration:** Repeat local training and aggregation until the desired number of rounds is reached.

## 🔬 Key Features

* Decentralized training simulation
* Multiple federated clients
* Local model optimization
* FedAvg aggregation
* Writer-based data partitioning
* Global accuracy and loss tracking
* Comparison with centralized training
* Potential extensions for differential privacy and secure aggregation

## 🚀 Getting Started

### Install dependencies

```bash
pip install torch torchvision datasets numpy matplotlib
```

### Run the project

```bash
python main.py
```

Make sure the FEMNIST dataset loader and client partitioning are configured for the dataset version you use.

## 📊 Evaluation Metrics

* Global test accuracy
* Training loss across communication rounds
* Per-client performance
* Communication rounds required for convergence
* Comparison with a centralized training baseline

## 🔐 Privacy Considerations

Federated learning reduces the need to transfer raw training data to a central server. However, model updates can still leak information. Differential privacy and secure aggregation are possible future enhancements.

## 🔮 Future Improvements

* Integrate Flower for a more realistic federated learning simulation.
* Support non-IID client distributions.
* Implement differential privacy.
* Explore secure aggregation.
* Compare FedAvg with alternative aggregation algorithms.
* Evaluate communication efficiency and convergence.

## 👨‍💻 Author

Developed as an AI/ML project exploring federated optimization, decentralized training, and privacy-aware machine learning.
