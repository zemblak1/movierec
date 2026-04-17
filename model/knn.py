import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
import os
import zipfile
import urllib.request
import pandas as pd

# load data

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    usecols=["user_id", "movie_id", "rating"]
)

# pivot the table so the index is user id for fitting into knn

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0) # 0 for no ratings

# mock user input for test (remove later)

mock_user_ratings = {
    242: 3,
    302: 3,
    377: 1,
    51:  4,
    346: 2,
    474: 5,
    265: 3,
    465: 4,
    451: 2,
    86:  5
}

rated_movie_ids = list(mock_user_ratings.keys())
subset_matrix = user_movie_matrix[rated_movie_ids]

# drop users who rated none of these 10 movies
subset_matrix = subset_matrix.loc[(subset_matrix != 0).any(axis=1)]

model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
model.fit(subset_matrix)

user_vector = pd.Series(mock_user_ratings).reindex(
    subset_matrix.columns, fill_value=0
)

# return matching user

neighbors = 1  # only need top 1 match for now

distances, indices = model.kneighbors(
    user_vector.values.reshape(1, -1),
    n_neighbors=neighbors
)

indices  = indices[0] # remove apparent nested array
distances = distances[0]

matched_users = []
for i in range(len(indices)):
    user_id    = user_movie_matrix.index[indices[i]]
    similarity = 1 - distances[i]   # convert distance to similarity
    matched_users.append({"user_id": user_id, "similarity": similarity})

best_match = matched_users[0]
print(f'your best match is user:{best_match['user_id']}')

