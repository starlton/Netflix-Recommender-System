from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from tqdm import tqdm
import pandas as pd
import numpy as np

def evaluate(model, interactions, k=10):
    metrics = {"precision": [], "recall": [], "ndcg": []}
    
    # Handle column renaming for evaluation
    df = interactions.copy()
    if 'userId' in df.columns:
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})

    # Evaluate on a sample (e.g., 100 users) to keep it fast for testing
    unique_users = df['user_id'].unique()
    sample_users = np.random.choice(unique_users, size=min(100, len(unique_users)), replace=False)
    
    print(f"Evaluating on {len(sample_users)} users...")
    
    for user in tqdm(sample_users):
        # Ground Truth
        truth = df[df['user_id'] == user]['item_id'].tolist()
        
        # Prediction
        recs = model.recommend(user, k)
        
        if not recs:
            continue

        metrics["precision"].append(precision_at_k(recs, truth, k))
        metrics["recall"].append(recall_at_k(recs, truth, k))
        metrics["ndcg"].append(ndcg_at_k(recs, truth, k))

    results = {m: sum(v) / len(v) if v else 0 for m, v in metrics.items()}
    return results

if __name__ == "__main__":
    import yaml
    from models.knn import train
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Load Ratings
    df = pd.read_csv(config['data']['interactions_path'])
    # Train
    model = train(df, config['model']['knn'])
    # Eval
    print(evaluate(model, df, k=10))
