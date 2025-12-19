import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def train(interactions, config): 
    """ 
    Train Matrix Factorization model using Scikit-Learn SVD
    (Replaces implicit ALS)
    """
    
    # Determine matrix shape
    n_users = interactions.user_id.max() + 1
    n_items = interactions.item_id.max() + 1

    # Create sparse interaction matrix
    matrix = csr_matrix(
        ([1] * len(interactions),
         (interactions.user_id, interactions.item_id)),
        shape=(n_users, n_items)
    )
    
    # Use TruncatedSVD as a standard alternative to ALS
    svd = TruncatedSVD(
        n_components=config["factors"],
        n_iter=config["iterations"],
        random_state=123
    )

    svd.fit(matrix) 
    return svd, matrix 

def recommend(model, matrix, user_id, k=10): 
    """
    Recommend movies for a user using the SVD model
    """

    # Get the user's interaction history
    user_row = matrix.getrow(user_id)
    
    # Transform user vector into latent space
    user_latent = model.transform(user_row)
    
    # Calculate scores for all items (dot product of user latent and item components)
    scores = np.dot(user_latent, model.components_).flatten()
    
    # Filter out items the user has already interacted with
    seen_indices = user_row.indices
    scores[seen_indices] = -np.inf
    
    # Get top k items (sorting indices by score in descending order)
    top_items = np.argsort(scores)[-k:][::-1]
    
    return top_items.tolist()