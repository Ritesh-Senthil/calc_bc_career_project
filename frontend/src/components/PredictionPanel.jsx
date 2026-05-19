const EMOTION_COLORS = {
  anger: "var(--anger)",
  disgust: "var(--disgust)",
  fear: "var(--fear)",
  joy: "var(--joy)",
  neutral: "var(--neutral)",
  sadness: "var(--sadness)",
  surprise: "var(--surprise)",
};

export default function PredictionPanel({ title, prediction }) {
  if (!prediction) return null;

  const color = EMOTION_COLORS[prediction.label] || "var(--text)";
  const pct = Math.round(prediction.confidence * 100);

  return (
    <div className="prediction-panel">
      <div className="panel-title">{title}</div>
      <div className="emotion-label" style={{ color }}>
        {prediction.label}
      </div>
      <div className="confidence">{pct}%</div>
    </div>
  );
}
