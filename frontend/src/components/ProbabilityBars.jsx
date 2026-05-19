const EMOTION_COLORS = {
  anger: "var(--anger)",
  disgust: "var(--disgust)",
  fear: "var(--fear)",
  joy: "var(--joy)",
  neutral: "var(--neutral)",
  sadness: "var(--sadness)",
  surprise: "var(--surprise)",
};

export default function ProbabilityBars({ probabilities, highlightLabel }) {
  if (!probabilities) return null;

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="probability-bars">
      <div className="section-title">Emotion Distribution</div>
      {sorted.map(([emotion, value]) => {
        const isHighlight = emotion === highlightLabel;
        return (
          <div
            key={emotion}
            className={`bar-row${isHighlight ? "" : " dimmed"}`}
          >
            <span className="bar-emotion">{emotion}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${(value * 100).toFixed(1)}%`,
                  background: EMOTION_COLORS[emotion] || "#666",
                }}
              />
            </div>
            <span className="bar-pct">{Math.round(value * 100)}%</span>
          </div>
        );
      })}
    </div>
  );
}
