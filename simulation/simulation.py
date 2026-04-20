# Simulate a short movie quiz, match the quiz taker to a similar user,
# and measure whether the resulting recommendations recover held-out likes.


from pathlib import Path
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# These are the main parameter settings that are used within the script. Change these if you want.
DATA_DIR = Path(__file__).resolve().parent.parent / "ml-100k"
FOLDS = ["u1", "u2", "u3", "u4", "u5"]

#Change these specific parameters to find how the success rate changes!!
TRIALS_PER_USER = 20
QUIZ_SIZE = 20
LIKE_THRESHOLD = 4
TOP_K_RECOMMENDATIONS = 20
BASE_SEED = 20


minimum_liked_movies = max(10, QUIZ_SIZE)

# Print the settings once so each run is easy to compare.
print(f"Simulation Settings")
print(f"Data directory: {DATA_DIR}")
print(f"Folds: {', '.join(FOLDS)}")
print(f"Trials per user: {TRIALS_PER_USER}")
print(f"Quiz size: {QUIZ_SIZE}")
print(f"Like threshold: >= {LIKE_THRESHOLD}")
print(f"Top recommendations kept: {TOP_K_RECOMMENDATIONS}")
print(f"Base seed: {BASE_SEED}")
print(f"Eligible users need at least {minimum_liked_movies} liked movies in base and at least 1 liked movie in test.")
print()


overall_eligible_users = 0
overall_trials = 0
overall_successes = 0
overall_misses = 0
overall_no_neighbor = 0
overall_no_recommendations = 0


# Run the same evaluation flow across each MovieLens train/test fold.
for fold_index, fold_name in enumerate(FOLDS):
    base_path = DATA_DIR / f"{fold_name}.base"
    test_path = DATA_DIR / f"{fold_name}.test"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing base ratings file: {base_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test ratings file: {test_path}")

    # Load the train and test ratings for this fold.
    base_ratings = pd.read_csv(
        base_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=["user_id", "movie_id", "rating"],
    )

    test_ratings = pd.read_csv(
        test_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=["user_id", "movie_id", "rating"],
    )

    # Build the user-movie matrix used for nearest-neighbor matching.
    user_movie_matrix = base_ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
    ).fillna(0)

    # Keep only liked movies so the quiz and evaluation use positive feedback.
    liked_base_ratings = base_ratings.loc[base_ratings["rating"] >= LIKE_THRESHOLD]
    liked_test_ratings = test_ratings.loc[test_ratings["rating"] >= LIKE_THRESHOLD]

    liked_base_counts = liked_base_ratings.groupby("user_id").size()
    liked_test_counts = liked_test_ratings.groupby("user_id").size()

    # Only evaluate users who have enough likes to form a quiz and a test target.
    eligible_user_ids = sorted(
        user_id
        for user_id in liked_base_counts.index
        if liked_base_counts[user_id] >= minimum_liked_movies
        and liked_test_counts.get(user_id, 0) >= 1
    )

    fold_trials = 0
    fold_successes = 0
    fold_misses = 0
    fold_no_neighbor = 0
    fold_no_recommendations = 0

    # Try multiple quiz samples for each eligible user.
    for target_user_id in eligible_user_ids:
        liked_movies = liked_base_ratings.loc[
            liked_base_ratings["user_id"] == target_user_id
        ]
        # Hold out liked test movies as the ground truth for success checks.
        hidden_like_ids = set(
            liked_test_ratings.loc[
                liked_test_ratings["user_id"] == target_user_id,
                "movie_id",
            ]
        )

        for trial_index in range(TRIALS_PER_USER):
            fold_trials += 1

            # Draw one quiz from the target user's liked training movies.
            trial_seed = (
                BASE_SEED
                + (fold_index * 1_000_000)
                + (target_user_id * 1_000)
                + trial_index
            )
            seed_movies = liked_movies.sample(n = QUIZ_SIZE, random_state = trial_seed)
            quiz_ratings = seed_movies.set_index("movie_id")["rating"].to_dict()

            # Compare users only on the movies that appear in this quiz.
            subset_matrix = (
                user_movie_matrix[list(quiz_ratings)]
                .loc[lambda df: (df != 0).any(axis=1)]
                .drop(index=target_user_id, errors="ignore")
            )

            if subset_matrix.empty:
                fold_no_neighbor += 1
                continue

            model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
            model.fit(subset_matrix)

            # Align the quiz answers with the subset matrix before searching.
            user_vector = pd.Series(quiz_ratings).reindex(subset_matrix.columns, fill_value=0)
            distances, indices = model.kneighbors(user_vector.to_frame().T, n_neighbors=1)

            matched_user_id = int(subset_matrix.index[indices[0, 0]])

            # Recommend the matched user's liked movies that were not in the quiz.
            recommendations = (
                base_ratings.loc[
                    (base_ratings["user_id"] == matched_user_id)
                    & (base_ratings["rating"] >= LIKE_THRESHOLD)
                    & (~base_ratings["movie_id"].isin(quiz_ratings))
                ]
                .drop_duplicates("movie_id")
                .sort_values(["rating", "movie_id"], ascending=[False, True])
                .head(TOP_K_RECOMMENDATIONS)
            )

            if recommendations.empty:
                fold_no_recommendations += 1
                continue

            # Count a hit when a recommendation matches a held-out liked movie.
            hits = set(recommendations["movie_id"]) & hidden_like_ids

            if hits:
                fold_successes += 1
            else:
                fold_misses += 1

    # Summarize this fold before moving to the next one.
    fold_failures = fold_trials - fold_successes
    fold_success_rate = 0 if fold_trials == 0 else fold_successes / fold_trials

    overall_eligible_users += len(eligible_user_ids)
    overall_trials += fold_trials
    overall_successes += fold_successes
    overall_misses += fold_misses
    overall_no_neighbor += fold_no_neighbor
    overall_no_recommendations += fold_no_recommendations

    print(f"{fold_name}")
    print(f"  eligible users: {len(eligible_user_ids)}")
    print(f"  executed trials: {fold_trials}")
    print(f"  successes: {fold_successes}")
    print(f"  failures: {fold_failures}")
    print(f"  misses: {fold_misses}")
    print(f"  no neighbor found: {fold_no_neighbor}")
    print(f"  no recommendations: {fold_no_recommendations}")
    print(f"  success rate: {fold_success_rate:.2%}")
    print()


# Print the final totals across all folds.
overall_failures = overall_trials - overall_successes
overall_success_rate = 0 if overall_trials == 0 else overall_successes / overall_trials

print("Overall")
print(f"  folds evaluated: {len(FOLDS)}")
print(f"  eligible users: {overall_eligible_users}")
print(f"  executed trials: {overall_trials}")
print(f"  successes: {overall_successes}")
print(f"  failures: {overall_failures}")
print(f"  misses: {overall_misses}")
print(f"  no neighbor found: {overall_no_neighbor}")
print(f"  no recommendations: {overall_no_recommendations}")
print(f"  success rate: {overall_success_rate:.2%}")
