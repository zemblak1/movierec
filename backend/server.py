import os
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ml-100k")

ratings = pd.read_csv(
    os.path.join(DATA_DIR, "u.data"),
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    usecols=["user_id", "movie_id", "rating"],
)

movies = pd.read_csv(
    os.path.join(DATA_DIR, "u.item"),
    sep="|",
    encoding="latin-1",
    names=["movie_id", "title"] + [f"col_{i}" for i in range(22)],
    usecols=["movie_id", "title"],
)

def fix_title(t):
    match = re.search(r"^(.*?)(?:,\s+(The|A|An))(\s+\(\d{4}\))?$", t)
    if match:
        main, article, year = match.groups()
        return f"{article} {main}{year or ''}"
    return t

movies["title"] = movies["title"].apply(fix_title)


# ---------------------------------------------------------------------------
# Pivot matrix & KNN model
# ---------------------------------------------------------------------------
user_movie_matrix = ratings.pivot_table(
    index="user_id", columns="movie_id", values="rating"
).fillna(0)

model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=5)
model.fit(user_movie_matrix.values)

# Pre-compute popularity (sum of ratings per movie)
popularity = ratings.groupby("movie_id")["rating"].sum().sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/movies/popular")
def popular():
    """Return the top N most popular movies (by total rating sum)."""
    n = int(request.args.get("n", 50))
    top_ids = popularity.head(n).index.tolist()
    result = movies[movies["movie_id"].isin(top_ids)].copy()
    # Preserve popularity order
    result["sort_order"] = result["movie_id"].map(
        {mid: idx for idx, mid in enumerate(top_ids)}
    )
    result = result.sort_values("sort_order")
    return jsonify(
        [
            {"movie_id": int(row.movie_id), "title": row.title}
            for row in result.itertuples()
        ]
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Accept  { ratings: { movie_id: rating } }  (all keys as strings is fine).
    Build a rating vector, find the K nearest neighbours, and return movies
    those neighbours rated highly that the quiz user has NOT already rated.
    """
    ratings_input = request.json.get("ratings", {})

    # Build the quiz-user vector (0 for unseen movies)
    user_vector = np.zeros(user_movie_matrix.shape[1])
    col_list = user_movie_matrix.columns.tolist()

    rated_movie_ids = set()
    for mid_str, rating_val in ratings_input.items():
        mid = int(mid_str)
        rating_val = float(rating_val)
        if rating_val == 0:
            continue  # "Haven't seen it"
        rated_movie_ids.add(mid)
        if mid in col_list:
            idx = col_list.index(mid)
            user_vector[idx] = rating_val

    if not rated_movie_ids:
        return jsonify([])

    # Find 10 nearest neighbours
    k = min(10, user_movie_matrix.shape[0])
    distances, indices = model.kneighbors(
        user_vector.reshape(1, -1), n_neighbors=k
    )

    # Aggregate highly-rated movies from neighbours
    neighbour_rows = user_movie_matrix.iloc[indices[0]]
    # Weight by similarity (1 - cosine distance)
    similarities = 1 - distances[0]
    weighted_scores = neighbour_rows.T.dot(similarities)

    # Convert to a Series with movie_id index
    score_series = pd.Series(weighted_scores, index=user_movie_matrix.columns)

    # Exclude movies the user already rated
    score_series = score_series.drop(labels=list(rated_movie_ids), errors="ignore")

    # Keep only movies with a positive score
    score_series = score_series[score_series > 0].sort_values(ascending=False)

    # Take top 10 recommendations
    top_recs = score_series.head(10)
    rec_ids = top_recs.index.tolist()
    rec_scores = top_recs.values.tolist()

    rec_movies = movies[movies["movie_id"].isin(rec_ids)].copy()
    rec_movies["score"] = rec_movies["movie_id"].map(dict(zip(rec_ids, rec_scores)))
    rec_movies = rec_movies.sort_values("score", ascending=False)

    return jsonify(
        [
            {
                "title": row.title,
                "similarity_score": round(float(row.score), 4),
            }
            for row in rec_movies.itertuples()
        ]
    )


if __name__ == "__main__":
    print("Movie Recommender API running on http://localhost:5000")
    app.run(port=5000, debug=True)
