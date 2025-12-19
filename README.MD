# Netflix Recommender System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

This project implements a production-style recommendation system inspired by real-world streaming platforms. It focuses on item-based similarity, real-time inference, and accurate retrieval using K-Nearest Neighbors (KNN).

---

## Dataset
**MovieLens 20M** (or a subset thereof) is used to build the interaction matrix. The system processes raw `ratings.csv` data to establish relationships between users and items based on historical viewing preferences.

---

## Models
I implemented a robust **Item-Based Collaborative Filtering** architecture:
* **K-Nearest Neighbors (KNN)**: Uses `Cosine Similarity` on a sparse interaction matrix to find movies that are mathematically similar to a user's favorites.
* **Scikit-Learn Implementation**: Leverages the `NearestNeighbors` algorithm with a `brute` force approach for exact precision.
* **Sparse Matrix Optimization**: Utilizes `scipy.sparse.csr_matrix` to efficiently handle millions of user-item interactions in memory.

---

## Evaluation
The system evaluates recommendations based on relevance and similarity metrics:
* **Cosine Similarity**: Measures the angle between item vectors to determine closeness.
* **Hit Rate**: The percentage of recommended items that align with held-out user preferences.
* **Visual Inspection**: Qualitative assessment via the Netflix-style dashboard to ensure recommendations make intuitive sense (e.g., "Toy Story" yields other animation films).

---

## Online Simulation
The project includes a **Real-Time Web Interface** to simulate the user experience:
* **Goal**: Bridge the gap between static offline metrics and real-time user interaction.
* **Method**: A React-style frontend (embedded in FastAPI) that allows for instant "User ID" lookup and visualizes the inference speed and recommendation quality live.

---

## Key Insights
* **Item-Based > User-Based**: For this dataset, finding similar *items* proved more stable and interpretable than finding similar *users*.
* **Sparsity Handling**: The use of Compressed Sparse Row (CSR) matrices was critical; standard DataFrames consumed too much memory.
* **Inference Speed**: Despite the large dataset, the KNN model (once fitted) provides sub-millisecond query times suitable for production APIs.

---

## Limitations
* **Cold Start Problem**: New users or items with zero ratings cannot be recommended effectively until interaction data is gathered.
* **Scalability**: The `brute` force algorithm is accurate but computationally expensive as the item catalog grows into the millions.
* **Single-Source Recommendations**: Currently heavily weighs the user's top-rated item; aggregating neighbors from multiple top items would improve diversity.

---

## Future Work
* **Approximate Nearest Neighbors (ANN)**: Migrate to libraries like FAISS or Annoy for faster retrieval at scale.
* **Hybrid Filtering**: Incorporate movie metadata (genres, tags) to solve the cold-start problem.
* **Vector Database**: Store embeddings in Pinecone or Milvus for distributed retrieval.

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
│   ├── knn.py                   # KNN Model Logic
│   ├── matrix_factorization.py  # Scikit-Learn SVD Implementation
│   └── neural_cf.py             # PyTorch Neural Collaborative Filtering
    └── popularity.py            # Baseline model
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

