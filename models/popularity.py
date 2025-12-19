def train(interactions): 
    """
    Train popularity-based recommender.
    """
    return (
        interactions.groupby("item_id") 
        .size()
        .sort_values(ascending=False) 
    )

def recommend(model, k=10): 
    """
    Recommend top-k popular movies
    """

    return model.head(k).index.tolist() 