import { useState, useEffect, useCallback } from "react";
import { Volume2, Loader2, Wifi, WifiOff } from "lucide-react";
import { analyzeAudio, checkHealth } from "./api";
import Recorder from "./components/Recorder";
import TranscriptBox from "./components/TranscriptBox";
import PredictionPanel from "./components/PredictionPanel";
import ProbabilityBars from "./components/ProbabilityBars";

export default function App() {
  const [recording, setRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);

  useEffect(() => {
    checkHealth()
      .then(() => setBackendConnected(true))
      .catch(() => setBackendConnected(false));
  }, []);

  const speakResponse = useCallback((text) => {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }, []);

  const handleRecordingComplete = useCallback(
    async (blob) => {
      setRecording(false);
      setAnalyzing(true);
      setError(null);

      try {
        const data = await analyzeAudio(blob);
        setResult(data);
        if (autoSpeak && data.spoken_response) {
          speakResponse(data.spoken_response);
        }
      } catch (err) {
        setError(err.message || "Something went wrong");
      } finally {
        setAnalyzing(false);
      }
    },
    [autoSpeak, speakResponse]
  );

  return (
    <div className="app">
      <header className="header">
        <h1>MoodMirror</h1>
        <p className="subtitle">
          Multimodal emotion recognition from voice
        </p>
        <div
          className={`connection-status${backendConnected ? " connected" : ""}`}
        >
          {backendConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span className="dot" />
          {backendConnected ? "Backend connected" : "Backend offline"}
        </div>
      </header>

      <Recorder
        onRecordingComplete={handleRecordingComplete}
        disabled={analyzing}
      />

      {analyzing && (
        <div className="analyzing">
          <Loader2 size={18} className="spinner" />
          Analyzing your recording&hellip;
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="results-grid">
          <TranscriptBox transcript={result.transcript} />

          <div className="predictions-row">
            <PredictionPanel
              title="Text Analysis"
              prediction={result.text_prediction}
            />
            <PredictionPanel
              title="Vocal Analysis"
              prediction={result.audio_prediction}
            />
            <PredictionPanel
              title="Combined"
              prediction={result.fusion_prediction}
            />
          </div>

          {result.fusion_prediction && (
            <ProbabilityBars
              probabilities={result.fusion_prediction.probabilities}
              highlightLabel={result.fusion_prediction.label}
            />
          )}

          {result.spoken_response && (
            <div className="response-section">
              <div className="label">Response</div>
              <p className="response-text">{result.spoken_response}</p>
              <div className="response-actions">
                <button
                  className="btn btn--primary"
                  onClick={() => speakResponse(result.spoken_response)}
                >
                  <Volume2 size={15} />
                  Speak
                </button>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={autoSpeak}
                    onChange={(e) => setAutoSpeak(e.target.checked)}
                  />
                  <span className="toggle-switch" />
                  Auto-speak
                </label>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
