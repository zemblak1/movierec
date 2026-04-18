import pandas as pd
from sklearn.neighbors import NearestNeighbors


ratings = pd.read_csv(
    "ml-100k/u1.base",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    usecols=["user_id", "movie_id", "rating"],
)

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
).fillna(0)

target_user_id = 123
liked_movies = ratings.loc[
    (ratings["user_id"] == target_user_id) & (ratings["rating"] >= 4)
]

if len(liked_movies) < 11:
    raise ValueError("Target user does not have enough liked movies for this test.")

seed_movies = liked_movies.sample(n=10, random_state=42)

quiz_ratings = seed_movies.set_index("movie_id")["rating"].to_dict()

hidden_like_ids = set(
    liked_movies.loc[
        ~liked_movies["movie_id"].isin(seed_movies["movie_id"]),
        "movie_id",
    ]
)

subset_matrix = (
    user_movie_matrix[list(quiz_ratings)]
    .loc[lambda df: (df != 0).any(axis=1)]
    .drop(index=target_user_id, errors="ignore")
)

model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
model.fit(subset_matrix)

user_vector = pd.Series(quiz_ratings).reindex(subset_matrix.columns, fill_value=0)
distances, indices = model.kneighbors(user_vector.to_frame().T, n_neighbors=1)

matched_user_id = subset_matrix.index[indices[0, 0]]
similarity = 1 - distances[0, 0]

recommendations = (
    ratings.loc[
        (ratings["user_id"] == matched_user_id)
        & (ratings["rating"] >= 4)
        & (~ratings["movie_id"].isin(quiz_ratings))
    ]
    .drop_duplicates("movie_id")
    .sort_values("rating", ascending=False)
    .head(10)
    .merge(movies, on="movie_id", how="left")[["movie_id", "title", "rating"]]
)

hits = set(recommendations["movie_id"]) & hidden_like_ids
hit_movies = movies[movies["movie_id"].isin(hits)][["movie_id", "title"]]

print("Quiz Ratings:")
print(quiz_ratings)
print(f"\nMatched user: {matched_user_id}")
print(f"Similarity: {similarity:.4f}")

print("\nRecommended movies:")
print(recommendations.to_string(index=False))

print("\nNumber of Hidden Liked Movies:", len(hidden_like_ids))
print("Number of Hits:", len(hits))
print("Successes:", bool(hits))

if hits:
    print("\nMatched Movies:")
    print(hit_movies.to_string(index=False))
else:
    print("\nNo recommended movies matched the user's hidden liked movies.")
