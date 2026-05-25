export default function TranscriptBox({ transcript, isLive }) {
  const isEmpty =
    !transcript || transcript === "[transcription unavailable]";

  return (
    <div className={`transcript-box${isLive ? " live" : ""}`}>
      <div className="label">
        {isLive ? "Live Transcript" : "Transcript"}
      </div>
      <div className={`text${isEmpty ? " text--empty" : ""}`}>
        {isEmpty
          ? isLive
            ? "Listening\u2026"
            : "No transcript available"
          : transcript}
        {isLive && !isEmpty && <span className="cursor-blink" />}
      </div>
    </div>
  );
}
