# Netflix Recommender System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

This project implements a production-style recommendation system inspired by real-world streaming platforms. It focuses on ranking-based evaluation, model comparison, and experimentation rather than simple prediction accuracy.

---

## Dataset
**MovieLens 20M** is used as implicit feedback by treating ratings $\ge$ 4.0 as positive signals. This mirrors real-world systems where explicit negative feedback is rare.

---

## Models
I implemented a hybrid architecture to balance performance and interpretability:
* **Popularity-based baseline**: Essential for cold-start scenarios and benchmarking.
* **Matrix Factorization (ALS)**: Efficient latent factor learning using Alternating Least Squares.
* **Neural Collaborative Filtering (NCF)**: A Deep Learning approach using PyTorch embeddings and MLP to capture non-linear interactions.

---

## Evaluation
Offline evaluation focuses on **Ranking Metrics** rather than RMSE, as these better reflect how users interact with top-ranked recommendations:
* **Precision@K**: The proportion of recommended items that are relevant.
* **Recall@K**: The proportion of relevant items found in the top-K.
* **NDCG@K**: A measure of ranking quality that penalizes relevant items appearing lower in the list.

---

## Online Simulation
A custom **Epsilon-Greedy A/B Simulator** demonstrates the trade-off between exploration and exploitation.
* **Goal**: Highlight the gap between offline performance (test set accuracy) and online reward (user clicks).
* **Method**: "Replays" historical data as a live stream to test bandit strategies.

---

## Key Insights
* **Popularity is a strong and necessary baseline**: It performs surprisingly well and is crucial for new users.
* **ALS offers an excellent performance/interpretability tradeoff**: Faster to train than deep learning while providing high-quality vectors.
* **Offline gains do not always translate to online improvements**: A model with better NDCG might learn slower in a bandit setting due to poor exploration.

---

## Limitations
* **No real-time user feedback**: The simulation uses historical replay, not live users.
* **Simplified reward modeling**: Interactions are binary (click/no-click) rather than watch-time weighted.
* **No distributed training or serving**: Currently runs on a single node (though served via FastAPI).

---

## Future Work
* **Cold-start modeling**: Integrate content-based embeddings (BERT on movie plots) for new items.
* **Causal uplift modeling**: Estimate the *incremental* impact of a recommendation.
* **Distributed inference**: Scale the serving layer using Docker/Kubernetes.
* **Vector Search**: Move to a vector database (FAISS/Pinecone) for retrieving from millions of items.

---

## Installation & Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```text
MOVIE RECOMMENDER/
│
├── api/
│   └── app.py               # FastAPI serving layer with embedded UI
│
├── data/
│   ├── raw/                 # MovieLens 20M dataset
│   └── processed/           # Processed interaction matrices
│
├── models/
│   ├── matrix_factorization.py  # ALS/SVD implementation
│   ├── neural_cf.py             # PyTorch Neural Collaborative Filtering
│   └── popularity.py            # Baseline model
│
├── evaluation/
│   ├── metrics.py           # Custom implementations of NDCG, Precision, Recall
│   └── offline_eval.py      # Script for batch model evaluation
│
├── simulation/
│   └── ab_simulator.py      # Epsilon-Greedy Bandit logic
│
├── notebooks/
│   ├── exploratory_analysis.ipynb   # EDA on Long-Tail distributions
│   └── ab_simulation_analysis.ipynb # A/B Testing & Regret visualizations
│
├── config.yaml              # Hyperparameters and paths
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```