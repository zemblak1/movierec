import { useState } from 'react';

export default function StarRating({ value, onChange }) {
  const [hovered, setHovered] = useState(0);

  return (
    <div className="star-rating">
      {[1, 2, 3, 4, 5].map((star) => {
        const isActive = star <= value;
        const isHovered = star <= hovered;
        let className = 'star-rating__star';
        if (isActive) className += ' star-rating__star--active';
        else if (isHovered) className += ' star-rating__star--hovered';

        return (
          <span
            key={star}
            className={className}
            onClick={() => onChange(star === value ? 0 : star)}
            onMouseEnter={() => setHovered(star)}
            onMouseLeave={() => setHovered(0)}
            role="button"
            aria-label={`Rate ${star} stars`}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onChange(star === value ? 0 : star);
              }
            }}
          >
            ★
          </span>
        );
      })}
      {value > 0 && (
        <span className="star-rating__label">{value}/5</span>
      )}
    </div>
  );
}
