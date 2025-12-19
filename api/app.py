import pandas as pd
import yaml
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import Optional

# --- CHANGED: Import KNN instead of NeuralCF ---
import models.knn as knn

app = FastAPI()

# --- Global Variables ---
model_resources = None
movies_df = None
interactions_df = None
config = None

@app.on_event("startup")
def load_resources():
    global model_resources, movies_df, interactions_df, config

    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load Data
    print("Loading Data...")
    movies_df = pd.read_csv(config['data']['movies_path'])
    interactions_df = pd.read_csv(config['data']['interactions_path'])
    
    # 3. Train KNN Model
    # We pass the config specific to KNN
    model_resources = knn.train(interactions_df, config['model'])
    print("KNN Model Trained & Ready.")

# --- The Modern UI (Identical to before) ---
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
        .movie-card:hover { transform: scale(1.05); z-index: 10; cursor: pointer; }
        .red-gradient { background: linear-gradient(to right, #E50914, #B20710); }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center">
    <header class="w-full p-6 flex justify-between items-center glass sticky top-0 z-50">
        <div class="text-3xl font-extrabold text-red-600">NETFLIX<span class="text-white font-light text-xl opacity-80">REC</span></div>
        <div class="text-xs text-gray-400 border border-gray-700 px-3 py-1 rounded-full">Model: <span class="text-blue-400">KNN (Item-Based)</span></div>
    </header>
    <main class="w-full max-w-7xl mt-12 px-6 flex flex-col items-center flex-grow">
        <div class="text-center mb-16 space-y-6 max-w-2xl">
            <h1 class="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">Next Level Picks.</h1>
            <div class="flex items-center justify-center mt-8 relative group">
                <div class="relative flex items-center bg-black rounded-full p-1 border border-gray-700">
                    <input type="number" id="userIdInput" placeholder="User ID (e.g. 1)" class="bg-transparent text-white px-6 py-4 rounded-full w-64 text-center focus:outline-none text-xl font-mono">
                    <button onclick="getRecommendations()" class="red-gradient text-white font-bold py-4 px-8 rounded-full shadow-lg transition-all hover:scale-105">Generate</button>
                </div>
            </div>
        </div>
        <div id="resultsArea" class="hidden w-full pb-20">
            <div id="moviesGrid" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6"></div>
        </div>
    </main>
    <script>
        async function getRecommendations() {
            const userId = document.getElementById('userIdInput').value;
            const grid = document.getElementById('moviesGrid');
            const resultsArea = document.getElementById('resultsArea');
            if (!userId) { alert("Please enter a User ID"); return; }
            
            grid.innerHTML = '';
            resultsArea.classList.remove('hidden');

            try {
                const response = await fetch(`/recommend/${userId}`);
                const data = await response.json();
                
                if (!data.recommendations || data.recommendations.length === 0) {
                     alert("No recommendations found. Try a different User ID."); return;
                }

                data.recommendations.forEach((movie, index) => {
                    const card = `
                        <div class="movie-card glass rounded-xl overflow-hidden relative group aspect-[2/3]">
                            <div class="absolute inset-0 bg-gray-800 flex items-center justify-center">
                                <span class="text-7xl font-bold opacity-10 text-white">${index + 1}</span>
                            </div>
                            <div class="absolute bottom-0 p-4 w-full bg-gradient-to-t from-black to-transparent">
                                <h3 class="font-bold text-sm text-white line-clamp-2">${movie.title}</h3>
                                <p class="text-xs text-gray-400">${movie.genres || "Genre N/A"}</p>
                            </div>
                        </div>`;
                    grid.innerHTML += card;
                });
            } catch (error) { console.error(error); alert("Error fetching data"); }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int):
    """
    Uses the KNN model to find movies similar to the user's favorite.
    """
    if model_resources is None:
        return {"error": "Model not ready"}

    # --- Use the new KNN logic ---
    k = config['evaluation']['k']
    rec_ids = knn.recommend(model_resources, user_id, interactions_df, k=k)
    
    results = []
    for mid in rec_ids:
        # Lookup metadata
        meta = movies_df[movies_df['movieId'] == mid]
        if not meta.empty:
            row = meta.iloc[0]
            results.append({
                "id": int(mid),
                "title": row['title'],
                "genres": row.get('genres', 'Unknown')
            })
            
    return {"user_id": user_id, "recommendations": results}
