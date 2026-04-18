export default function ResultCard({ rec, index }) {
  const rankClass =
    index === 0
      ? 'result-card__rank--gold'
      : index === 1
        ? 'result-card__rank--silver'
        : index === 2
          ? 'result-card__rank--bronze'
          : 'result-card__rank--default';

  // Normalise the bar width relative to the top score (first element)
  const maxScore = 1; // we'll pass this in from parent if needed
  const barPercent = Math.min(100, (rec.similarity_score / (rec._maxScore || 1)) * 100);

  return (
    <div
      className="result-card"
      style={{ animationDelay: `${index * 0.08}s` }}
      id={`result-${index}`}
    >
      <div className={`result-card__rank ${rankClass}`}>
        {index + 1}
      </div>

      <div className="result-card__info">
        <div className="result-card__title">{rec.title}</div>
        <div className="result-card__score">
          Match score: {rec.similarity_score.toFixed(2)}
        </div>
      </div>

      <div className="result-card__bar-container">
        <div
          className="result-card__bar"
          style={{ width: `${barPercent}%` }}
        />
      </div>
    </div>
  );
}
