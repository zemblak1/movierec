import StarRating from './StarRating';

export default function MovieCard({ movie, index, rating, onRate, onSkip }) {
  const isRated = rating > 0;

  return (
    <div
      className={`movie-card${isRated ? ' movie-card--rated' : ''}`}
      style={{ animationDelay: `${index * 0.06}s` }}
      id={`movie-card-${movie.movie_id}`}
    >
      <span className="movie-card__number">{index + 1}</span>
      <h3 className="movie-card__title">{movie.title}</h3>

      <StarRating value={rating} onChange={onRate} />

      <button
        className="havent-seen-btn"
        onClick={() => onSkip()}
        type="button"
      >
        <span>👁️‍🗨️</span>
        Haven't seen it
      </button>
    </div>
  );
}
