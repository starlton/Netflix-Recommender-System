from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from tqdm import tqdm

def evaluate(model_fn, interactions, k=10):
    metrics = {"precision": [], "recall": [], "ndcg": []}

    for user in tqdm(interactions.user_id.unique()):
        truth = interactions[interactions.user_id == user].item_id.tolist()
        recs = model_fn(user, k)

        metrics["precision"].append(precision_at_k(recs, truth, k))
        metrics["recall"].append(recall_at_k(recs, truth, k))
        metrics["ndcg"].append(ndcg_at_k(recs, truth, k))

    return {m: sum(v) / len(v) for m, v in metrics.items()}
