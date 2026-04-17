import { useState, useEffect, useCallback } from 'react';
import './App.css';
import MovieCard from './components/MovieCard';
import ResultCard from './components/ResultCard';

const API_BASE = 'http://localhost:5000';

function shuffleAndPick(arr, n) {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

export default function App() {
  const [moviePool, setMoviePool] = useState([]);
  const [quizMovies, setQuizMovies] = useState([]);
  const [ratings, setRatings] = useState({});
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState(null);

  // ─── Fetch popular movies on mount ───
  useEffect(() => {
    async function fetchPopular() {
      try {
        const res = await fetch(`${API_BASE}/movies/popular?n=50`);
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        const picked = shuffleAndPick(data, 10);
        setQuizMovies(picked);
        // Fill pool with the rest
        const remaining = data.filter(
          (m) => !picked.some((p) => p.movie_id === m.movie_id)
        );
        setMoviePool(remaining);
        // Initialise all ratings to 0 (haven't seen)
        const initial = {};
        picked.forEach((m) => {
          initial[m.movie_id] = 0;
        });
        setRatings(initial);
      } catch (err) {
        setError(
          'Could not connect to the backend. Make sure the Flask server is running on http://localhost:5000'
        );
      } finally {
        setInitialLoading(false);
      }
    }
    fetchPopular();
  }, []);

  // ─── Rate a movie ───
  const handleRate = useCallback((movieId, value) => {
    setRatings((prev) => ({ ...prev, [movieId]: value }));
  }, []);

  // ─── Skip a movie ───
  const handleSkip = useCallback((movieId) => {
    setMoviePool((prevPool) => {
      if (prevPool.length === 0) return prevPool;
      const newPool = [...prevPool];
      const nextMovie = newPool.shift();

      setQuizMovies((prevQuiz) =>
        prevQuiz.map((m) => (m.movie_id === movieId ? nextMovie : m))
      );

      setRatings((prevRatings) => ({
        ...prevRatings,
        [nextMovie.movie_id]: 0,
      }));

      return newPool;
    });
  }, []);

  // ─── Submit ratings ───
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setRecommendations(null);

    // Build the ratings payload — only include movies that were actually rated
    const payload = {};
    Object.entries(ratings).forEach(([movieId, rating]) => {
      payload[movieId] = rating;
    });

    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratings: payload }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      // Attach max score for bar normalisation
      const maxScore = data.length > 0 ? data[0].similarity_score : 1;
      const enriched = data.map((r) => ({ ...r, _maxScore: maxScore }));

      setRecommendations(enriched);
    } catch (err) {
      setError('Failed to get recommendations. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  // ─── Reset ───
  const handleReset = () => {
    setRecommendations(null);
    setError(null);
    // Re-fetch new movies
    setInitialLoading(true);
    fetch(`${API_BASE}/movies/popular?n=50`)
      .then((res) => res.json())
      .then((data) => {
        const picked = shuffleAndPick(data, 10);
        setQuizMovies(picked);
        const remaining = data.filter(
          (m) => !picked.some((p) => p.movie_id === m.movie_id)
        );
        setMoviePool(remaining);
        const initial = {};
        picked.forEach((m) => {
          initial[m.movie_id] = 0;
        });
        setRatings(initial);
      })
      .catch(() =>
        setError('Could not connect to the backend.')
      )
      .finally(() => setInitialLoading(false));
  };

  // ─── Count rated movies ───
  const ratedCount = Object.values(ratings).filter((v) => v > 0).length;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header" id="app-header">
        <div className="header__logo">
          <div className="header__icon">🎬</div>
          <div>
            <div className="header__title">MovieRec</div>
            <div className="header__subtitle">Powered by KNN • MovieLens 100K</div>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="hero" id="hero-section">
        <h1 className="hero__title">
          Discover your next{' '}
          <span className="hero__title-accent">favorite film</span>
        </h1>
        <p className="hero__description">
          Rate a handful of popular movies and our machine-learning model will
          find your taste twin among 100,000 real ratings.
        </p>
      </section>

      {/* ── Error ── */}
      {error && (
        <div className="error-banner" id="error-banner" role="alert">
          ⚠️ {error}
        </div>
      )}

      {/* ── Quiz Section ── */}
      {!recommendations && (
        <section className="section" id="quiz-section">
          <div className="section__header">
            <div className="section__icon section__icon--quiz">🎯</div>
            <div>
              <h2 className="section__title">Rate these movies</h2>
              <p className="section__subtitle">
                Use the stars or mark "Haven't seen it"
              </p>
            </div>
          </div>

          {initialLoading ? (
            <div className="spinner-container">
              <div className="spinner" />
              <span className="spinner-container__text">Loading movies…</span>
            </div>
          ) : (
            <>
              <div className="movie-grid" id="movie-grid">
                {quizMovies.map((movie, i) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    index={i}
                    rating={ratings[movie.movie_id] || 0}
                    onRate={(val) => handleRate(movie.movie_id, val)}
                    onSkip={() => handleSkip(movie.movie_id)}
                  />
                ))}
              </div>

              <div className="actions" id="actions-area">
                <p className="actions__counter">
                  You've rated <strong>{ratedCount}</strong> of{' '}
                  {quizMovies.length} movies
                </p>
                <button
                  className="cta-button"
                  id="get-recommendations-btn"
                  onClick={handleSubmit}
                  disabled={loading || ratedCount === 0}
                >
                  {loading ? (
                    <>
                      <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                      Analyzing…
                    </>
                  ) : (
                    <>🍿 Get Recommendations</>
                  )}
                </button>
              </div>
            </>
          )}
        </section>
      )}

      {/* ── Loading State ── */}
      {loading && (
        <div className="spinner-container" id="loading-spinner">
          <div className="spinner" />
          <span className="spinner-container__text">
            Finding your taste twin among 943 users…
          </span>
        </div>
      )}

      {/* ── Results Section ── */}
      {recommendations && !loading && (
        <section className="section results" id="results-section">
          <div className="section__header">
            <div className="section__icon section__icon--results">✨</div>
            <div>
              <h2 className="section__title">Your recommendations</h2>
              <p className="section__subtitle">
                Based on the {ratedCount} movie{ratedCount !== 1 ? 's' : ''} you
                rated
              </p>
            </div>
          </div>

          {recommendations.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
              No recommendations found. Try rating more movies!
            </p>
          ) : (
            <div className="results-list" id="results-list">
              {recommendations.map((rec, i) => (
                <ResultCard key={rec.title} rec={rec} index={i} />
              ))}
            </div>
          )}

          <div style={{ textAlign: 'center' }}>
            <button
              className="reset-button"
              id="try-again-btn"
              onClick={handleReset}
            >
              🔄 Try Again with Different Movies
            </button>
          </div>
        </section>
      )}

      {/* ── Footer ── */}
      <footer className="footer" id="app-footer">
        MovieRec — Built with React + Flask + scikit-learn • MovieLens 100K
        dataset
      </footer>
    </div>
  );
}
