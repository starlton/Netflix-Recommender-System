import torch
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List, Optional
import os

# Import your actual model
# (Make sure an __init__.py exists in 'models/' or python path is set)
from models.neural_cf import NeuralCF

app = FastAPI()

# --- Global Variables to hold data in memory ---
model: Optional[NeuralCF] = None
movies_df: Optional[pd.DataFrame] = None
all_movie_ids: Optional[torch.Tensor] = None

# --- Configuration ---
# Update these paths to match where your files actually are
MOVIES_CSV_PATH = "movies.csv"
N_USERS = 138494  # From MovieLens 20M metadata (or calculate dynamically)
N_ITEMS = 131263  # From MovieLens 20M metadata

@app.on_event("startup")
def load_resources():
    """
    Load the model and data once when the server starts.
    """
    global model, movies_df, all_movie_ids

    print("Loading Movie Data...")
    try:
        # Load movie titles
        movies_df = pd.read_csv(MOVIES_CSV_PATH)
        # Create a tensor of all movie IDs for prediction
        # (Filter to ensure IDs are within model's range)
        valid_movie_ids = movies_df[movies_df['movieId'] < N_ITEMS]['movieId'].values
        all_movie_ids = torch.tensor(valid_movie_ids, dtype=torch.long)
        print(f"Loaded {len(movies_df)} movies.")
    except Exception as e:
        print(f"WARNING: Could not load movies.csv: {e}")
        # Fallback for demo if file missing
        movies_df = pd.DataFrame(columns=["movieId", "title"])
        all_movie_ids = torch.tensor([], dtype=torch.long)

    print("Initializing NeuralCF Model...")
    # Initialize the model architecture
    model = NeuralCF(n_users=N_USERS, n_items=N_ITEMS)
    
    # In a real production scenario, you would load trained weights here:
    # model.load_state_dict(torch.load("models/saved_weights.pth"))
    # model.eval() 
    
    print("System Ready.")

# --- The Modern UI (HTML/Tailwind) ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StreamLine | AI Recommender</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #141414; color: #fff; }
        .glass { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .movie-card { transition: all 0.3s ease; }
        .movie-card:hover { transform: scale(1.05); z-index: 10; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
        .red-gradient { background: linear-gradient(to right, #E50914, #B20710); }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #141414; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #E50914; }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center">

    <header class="w-full p-6 flex justify-between items-center glass sticky top-0 z-50">
        <div class="flex items-center gap-2">
            <div class="text-3xl font-extrabold tracking-tighter text-red-600">NETFLIX<span class="text-white font-light text-xl opacity-80">REC</span></div>
        </div>
        <div class="flex items-center gap-4">
            <div class="text-xs font-mono text-gray-400 bg-black/30 px-3 py-1 rounded-full border border-gray-700">
                Model: <span class="text-blue-400">NeuralCF (PyTorch)</span>
            </div>
            <div class="h-3 w-3 rounded-full bg-green-500 animate-pulse"></div>
        </div>
    </header>

    <main class="w-full max-w-7xl mt-12 px-6 flex flex-col items-center flex-grow">
        
        <div class="text-center mb-16 space-y-6 max-w-2xl animate-fade-in-up">
            <h1 class="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
                Next Level Picks.
            </h1>
            <p class="text-gray-400 text-lg md:text-xl font-light">
                Enter your User ID to generate real-time inferences using Deep Learning.
            </p>
            
            <div class="flex items-center justify-center mt-8 relative group">
                <div class="absolute -inset-1 bg-gradient-to-r from-red-600 to-purple-600 rounded-full blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
                <div class="relative flex items-center bg-black rounded-full p-1 border border-gray-700">
                    <input type="number" id="userIdInput" placeholder="User ID (e.g. 1)" 
                           class="bg-transparent text-white px-6 py-4 rounded-full w-64 text-center focus:outline-none placeholder-gray-600 text-xl font-mono">
                    <button onclick="getRecommendations()" 
                            class="red-gradient hover:brightness-110 text-white font-bold py-4 px-8 rounded-full shadow-lg transition-all transform hover:scale-105 active:scale-95 flex items-center gap-2">
                        <span>Generate</span>
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clip-rule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <div id="resultsArea" class="hidden w-full pb-20">
            <div class="flex items-center justify-between mb-8 border-b border-gray-800 pb-4">
                <h2 class="text-2xl font-semibold flex items-center gap-2">
                    <span class="text-red-600">I</span> Top Picks For You
                </h2>
                <span class="text-gray-500 text-sm" id="inferenceTime">Inference: 0ms</span>
            </div>
            
            <div id="moviesGrid" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6">
                </div>
        </div>
        
        <div id="loader" class="hidden mt-12 flex flex-col items-center">
            <div class="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-red-600 mb-4"></div>
            <p class="text-gray-500 animate-pulse">Running Forward Pass...</p>
        </div>

    </main>

    <footer class="w-full py-8 text-center text-gray-600 text-sm border-t border-gray-900 mt-auto">
        <p>Built with PyTorch & FastAPI</p>
    </footer>

    <script>
        async function getRecommendations() {
            const userId = document.getElementById('userIdInput').value;
            const grid = document.getElementById('moviesGrid');
            const resultsArea = document.getElementById('resultsArea');
            const loader = document.getElementById('loader');
            const timeLabel = document.getElementById('inferenceTime');

            if (!userId) { alert("Please enter a User ID"); return; }

            // Reset UI
            resultsArea.classList.add('hidden');
            loader.classList.remove('hidden');
            grid.innerHTML = '';

            const startTime = performance.now();

            try {
                // Fetch from YOUR API
                const response = await fetch(`/recommend/${userId}`);
                const data = await response.json();
                
                const endTime = performance.now();
                timeLabel.innerText = `Inference: ${(endTime - startTime).toFixed(0)}ms`;

                if (!data.recommendations || data.recommendations.length === 0) {
                     alert("No recommendations found or User ID out of range.");
                     loader.classList.add('hidden');
                     return;
                }

                data.recommendations.forEach((movie, index) => {
                    // Use a placeholder image if specific movie posters aren't available
                    // We generate a deterministic color/gradient based on ID for variety
                    const hue = (movie.id * 137) % 360; 
                    
                    const card = `
                        <div class="movie-card glass rounded-xl overflow-hidden relative group aspect-[2/3]">
                            <div class="absolute inset-0 bg-gray-800 flex items-center justify-center">
                                <span class="text-7xl font-bold opacity-10 select-none text-white">${index + 1}</span>
                                <div class="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-90"></div>
                            </div>
                            
                            <div class="absolute bottom-0 p-4 w-full">
                                <div class="flex items-center gap-2 mb-2">
                                    <span class="bg-red-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
                                        ${(movie.score * 100).toFixed(0)}% MATCH
                                    </span>
                                </div>
                                <h3 class="font-bold text-sm md:text-base leading-tight text-white mb-1 line-clamp-2" title="${movie.title}">
                                    ${movie.title}
                                </h3>
                                <p class="text-xs text-gray-400 truncate">${movie.genres || "Genre N/A"}</p>
                            </div>
                            
                            <div class="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 px-2 py-1 rounded text-[10px] font-mono border border-gray-700">
                                ID: ${movie.id}
                            </div>
                        </div>
                    `;
                    grid.innerHTML += card;
                });

                loader.classList.add('hidden');
                resultsArea.classList.remove('hidden');

            } catch (error) {
                console.error(error);
                alert("Error fetching recommendations. Check console for details.");
                loader.classList.add('hidden');
            }
        }
    </script>
</body>
</html>
"""

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the modern frontend Dashboard."""
    return html_content

@app.get("/recommend/{user_id}")
async def recommend(user_id: int):
    """
    Real-time inference using the NeuralCF model.
    """
    global model, movies_df, all_movie_ids

    # 1. Input Validation
    if model is None or movies_df is None:
        return {"error": "Model not initialized. Check server logs."}
    
    # 2. Prepare Data for PyTorch
    # Create a tensor of the user_id repeated N times (once for each movie)
    # shape: [n_items]
    user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long)
    
    # 3. Model Inference (Forward Pass)
    # We turn off gradient calculation for inference (faster/less memory)
    with torch.no_grad():
        # returns shape: [n_items] (predicted score for every movie)
        predictions = model(user_tensor, all_movie_ids)
    
    # 4. Post-processing (Ranking)
    # Get top 10 indices with highest scores
    top_k = 10
    top_scores, top_indices = torch.topk(predictions, top_k)
    
    # Convert tensor indices back to Movie IDs
    recommended_ids = all_movie_ids[top_indices].numpy()
    recommended_scores = top_scores.numpy()
    
    # 5. Fetch Metadata (Titles, Genres)
    results = []
    for rank, movie_id in enumerate(recommended_ids):
        # Look up movie details in pandas df
        meta = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        results.append({
            "id": int(movie_id),
            "title": meta['title'],
            "genres": meta.get('genres', 'Unknown'),
            "score": float(recommended_scores[rank])  # Confidence score
        })

    return {
        "user_id": user_id, 
        "recommendations": results
    }