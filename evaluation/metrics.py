import numpy as np

def precision_at_k(recs, truth, k):
    return len(set(recs[:k]) & set(truth)) / k

def recall_at_k(recs, truth, k):
    return len(set(recs[:k]) & set(truth)) / len(truth)

def ndcg_at_k(recs, truth, k):
    score = 0.0
    for i, item in enumerate(recs[:k]):
        if item in truth:
            score += 1 / np.log2(i + 2)
    return score
