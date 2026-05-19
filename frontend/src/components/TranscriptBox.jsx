export default function TranscriptBox({ transcript }) {
  const isEmpty =
    !transcript || transcript === "[transcription unavailable]";

  return (
    <div className="transcript-box">
      <div className="label">Transcript</div>
      <div className={`text${isEmpty ? " text--empty" : ""}`}>
        {isEmpty ? "No transcript available" : transcript}
      </div>
    </div>
  );
}
