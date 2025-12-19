import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def train(interactions, config):
    """
    Trains the KNN model exactly like Recommender_System.ipynb.
    1. Pivots data into a Matrix (Movies x Users)
    2. Fits NearestNeighbors model
    """
    print("Step 1: Pivoting data (this matches your notebook)...")
    
    # Create the Matrix: Rows=Movies, Cols=Users, Values=Ratings
    # We use a sparse matrix for speed, but the logic is identical to pivot_table
    user_ids = interactions['userId'].unique()
    movie_ids = interactions['movieId'].unique()

    # Mappers to convert real IDs to Matrix Indices
    user_mapper = {id: i for i, id in enumerate(user_ids)}
    movie_mapper = {id: i for i, id in enumerate(movie_ids)}
    movie_inv_mapper = {i: id for i, id in enumerate(movie_ids)}

    user_index = [user_mapper[i] for i in interactions['userId']]
    movie_index = [movie_mapper[i] for i in interactions['movieId']]

    # Create CSR Matrix (Item-Based CF)
    matrix = csr_matrix(
        (interactions['rating'], (movie_index, user_index)), 
        shape=(len(movie_ids), len(user_ids))
    )
    
    print("Step 2: Fitting KNN Model...")
    model = NearestNeighbors(
        n_neighbors=config["k_neighbors"],
        metric=config["metric"],
        algorithm=config["algorithm"]
    )
    model.fit(matrix)
    
    return model, matrix, movie_mapper, movie_inv_mapper

def recommend(model_resources, user_id, interactions_df, k=10):
    """
    Recommends movies by finding the user's favorite movie
    and finding neighbors to that movie.
    """
    model, matrix, movie_mapper, movie_inv_mapper = model_resources
    
    # 1. Get user's history
    user_history = interactions_df[interactions_df['userId'] == user_id]
    
    if user_history.empty:
        return []

    # 2. Find their #1 favorite movie (Highest rated)
    # This matches the notebook's strategy of "Item-Based" recommendation
    favorite_movie_id = user_history.sort_values('rating', ascending=False).iloc[0]['movieId']
    
    if favorite_movie_id not in movie_mapper:
        return []

    # 3. Find neighbors to that movie
    movie_idx = movie_mapper[favorite_movie_id]
    movie_vec = matrix[movie_idx].reshape(1, -1)
    
    # Get K+1 neighbors (because the first one is the movie itself)
    distances, indices = model.kneighbors(movie_vec, n_neighbors=k+1)
    
    # 4. Convert indices back to Movie IDs
    recommendations = []
    for i in range(1, len(indices.flatten())):
        idx = indices.flatten()[i]
        movie_id = movie_inv_mapper[idx]
        recommendations.append(movie_id)
        
    return recommendations