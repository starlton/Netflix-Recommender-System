import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def train(interactions, config): 
    """ 
    Train Matrix Factorization model using Scikit-Learn SVD
    (Replaces implicit ALS)
    """
    n_users = interactions.user_id.max() + 1
    n_items = interactions.item_id.max() + 1

    matrix = csr_matrix(
        ([1] * len(interactions),
         (interactions.user_id, interactions.item_id)),
        shape=(n_users, n_items)
    )
    
    svd = TruncatedSVD(
        n_components=config["factors"],
        n_iter=config["iterations"],
        random_state=123
    )

    svd.fit(matrix) 
    return svd, matrix 

def recommend(model, matrix, user_id, k=10): 
    user_row = matrix.getrow(user_id)
    user_latent = model.transform(user_row)
    scores = np.dot(user_latent, model.components_).flatten()
    
    seen_indices = user_row.indices
    scores[seen_indices] = -np.inf
    
    top_items = np.argsort(scores)[-k:][::-1]
    return top_items.tolist()
