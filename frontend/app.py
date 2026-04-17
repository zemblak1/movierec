from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Label, Input, Button, Static
from textual.containers import VerticalScroll, Container
from textual import work
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import random
import os

class MovieModel:
    def __init__(self):
        self.ready = False
    
    def load(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, "ml-100k", "u.data")
        item_path = os.path.join(base_dir, "ml-100k", "u.item")

        try:
            self.ratings = pd.read_csv(
                data_path,
                sep="\\t",
                names=["user_id", "movie_id", "rating", "timestamp"],
                usecols=["user_id", "movie_id", "rating"]
            )
            self.movies = pd.read_csv(
                item_path,
                sep="|",
                encoding="latin-1",
                names=["movie_id", "title"] + [f"col_{i}" for i in range(22)],
                usecols=["movie_id", "title"]
            )
        except Exception as e:
            return str(e)

        self.user_movie_matrix = self.ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        ).fillna(0)

        self.model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
        self.model.fit(self.user_movie_matrix.values)

        ratingstats = self.ratings.groupby("movie_id")["rating"].sum()
        ratingstats = ratingstats.sort_values(ascending=False).head(50)
        self.popular_movie_ids = ratingstats.index.tolist()

        self.ready = True
        return None

    def get_questions(self, count=10):
        selected_ids = random.sample(self.popular_movie_ids, count)
        questions = []
        for mid in selected_ids:
            title = self.movies[self.movies["movie_id"] == mid]["title"].values[0]
            questions.append({"id": mid, "title": title})
        return questions

    def get_recommendations(self, user_ratings_dict):
        user_vector = pd.Series(user_ratings_dict).reindex(
            self.user_movie_matrix.columns, fill_value=0
        )
        
        distances, indices = self.model.kneighbors(
            user_vector.values.reshape(1, -1),
            n_neighbors=1
        )
        
        best_match_user_idx = indices[0][0]
        best_match_user_id = self.user_movie_matrix.index[best_match_user_idx]
        
        matched_user_ratings = self.ratings[self.ratings["user_id"] == best_match_user_id]
        top_ratings = matched_user_ratings[matched_user_ratings["rating"] >= 4]
        
        seen_movie_ids = set(user_ratings_dict.keys())
        top_ratings = top_ratings[~top_ratings["movie_id"].isin(seen_movie_ids)]
        
        top_movie_ids = top_ratings.sort_values("rating", ascending=False).head(10)["movie_id"].tolist()
        
        recommended_movies = []
        for mid in top_movie_ids:
            title = self.movies[self.movies["movie_id"] == mid]["title"].values[0]
            recommended_movies.append(title)
            
        return recommended_movies

class MovieRecApp(App):
    TITLE = "MovieRec"
    CSS = """
    #loading {
        content-align: center middle;
        height: 100%;
    }
    #questions-container {
        padding: 1 2;
    }
    .question-block {
        margin-bottom: 1;
    }
    .title {
        text-style: bold;
        color: white;
    }
    #questions-list {
        display: none;
    }
    #results {
        display: none;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Loading Movie Recommendation Model... Please Wait", id="loading")
        
        with VerticalScroll(id="questions-list"):
            yield Container(id="questions-container")
            yield Button("Get Recommendations", id="submit-btn", variant="primary")
            
        with VerticalScroll(id="results"):
            yield Label("Recommended Movies for You:\\n", classes="title")
            yield Static("", id="results-text")

        yield Footer()

    def on_mount(self):
        self.movie_model = MovieModel()
        self.load_model()

    @work(thread=True)
    def load_model(self):
        err = self.movie_model.load()
        self.post_message(self.ModelLoaded(err))

    class ModelLoaded(App.Message):
        def __init__(self, err: str | None) -> None:
            self.err = err
            super().__init__()

    def on_movie_rec_app_model_loaded(self, message: ModelLoaded):
        if message.err:
            self.query_one("#loading", Label).update(f"Error loading model: {message.err}")
        else:
            self.query_one("#loading").display = False
            self.query_one("#questions-list").display = True
            self.questions = self.movie_model.get_questions(10)
            
            container = self.query_one("#questions-container")
            for i, q in enumerate(self.questions):
                title = q['title']
                with container.app.batch_update():
                    container.mount(Label(f"Question {i+1} (0 if you havent seen it)"))
                    container.mount(Label(f"Rating of {title}: "))
                    container.mount(Input(placeholder="0-5", type="integer", id=f"input_{q['id']}", classes="question-block"))

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "submit-btn":
            user_ratings = {}
            for q in self.questions:
                mid = q['id']
                val = self.query_one(f"#input_{mid}", Input).value
                if val:
                    try:
                        parsed = int(val)
                        if parsed >= 0 and parsed <= 5:
                            user_ratings[mid] = parsed
                    except ValueError:
                        pass
            
            self.query_one("#questions-list").display = False
            self.query_one("#results").display = True
            self.query_one("#results-text", Static).update("Computing nearest neighbor...")
            self.compute_recs(user_ratings)

    @work(thread=True)
    def compute_recs(self, user_ratings):
        recs = self.movie_model.get_recommendations(user_ratings)
        self.post_message(self.RecsComputed(recs))

    class RecsComputed(App.Message):
        def __init__(self, recs: list) -> None:
            self.recs = recs
            super().__init__()

    def on_movie_rec_app_recs_computed(self, message: RecsComputed):
        out = "\\n".join([f"'{r}'" for r in message.recs])
        if not out:
            out = "No recommendations found."
        self.query_one("#results-text", Static).update(out)

if __name__ == "__main__":
    MovieRecApp().run()