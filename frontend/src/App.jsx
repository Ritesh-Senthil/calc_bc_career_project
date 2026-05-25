import { useState, useEffect, useCallback, useRef } from "react";
import { Volume2, Loader2, Wifi, WifiOff, Send } from "lucide-react";
import { analyzeAudio, analyzeText, checkHealth } from "./api";
import Recorder from "./components/Recorder";
import TranscriptBox from "./components/TranscriptBox";
import PredictionPanel from "./components/PredictionPanel";
import ProbabilityBars from "./components/ProbabilityBars";

export default function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);

  const [recording, setRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [textInput, setTextInput] = useState("");

  const [speaking, setSpeaking] = useState(false);
  const [spokenCharIndex, setSpokenCharIndex] = useState(0);
  const utteranceRef = useRef(null);

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
    utteranceRef.current = utterance;

    utterance.onstart = () => {
      setSpeaking(true);
      setSpokenCharIndex(0);
    };
    utterance.onboundary = (e) => {
      if (typeof e.charIndex === "number") {
        setSpokenCharIndex(e.charIndex + (e.charLength || 0));
      }
    };
    utterance.onend = () => {
      setSpeaking(false);
      setSpokenCharIndex(text.length);
    };
    utterance.onerror = () => {
      setSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, []);

  const handleResult = useCallback(
    (data) => {
      setResult(data);
      if (autoSpeak && data.spoken_response) {
        speakResponse(data.spoken_response);
      }
    },
    [autoSpeak, speakResponse]
  );

  const handleLiveTranscript = useCallback((text) => {
    setLiveTranscript(text);
    setRecording(true);
  }, []);

  const handleRecordingComplete = useCallback(
    async (blob) => {
      setRecording(false);
      setAnalyzing(true);
      setError(null);
      setResult(null);
      window.speechSynthesis.cancel();
      setSpeaking(false);

      try {
        const data = await analyzeAudio(blob, liveTranscript);
        handleResult(data);
      } catch (err) {
        setError(err.message || "Something went wrong");
      } finally {
        setAnalyzing(false);
      }
    },
    [liveTranscript, handleResult]
  );

  const handleTextSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      const text = textInput.trim();
      if (!text) return;

      setAnalyzing(true);
      setError(null);
      setResult(null);
      window.speechSynthesis.cancel();
      setSpeaking(false);

      try {
        const data = await analyzeText(text);
        handleResult(data);
      } catch (err) {
        setError(err.message || "Something went wrong");
      } finally {
        setAnalyzing(false);
      }
    },
    [textInput, handleResult]
  );

  const responseText = result?.spoken_response || "";

  return (
    <div className="app">
      <header className="header">
        <h1>MoodMirror</h1>
        <p className="subtitle">
          Emotion recognition from voice and text
        </p>
        <div
          className={`connection-status${backendConnected ? " connected" : ""}`}
        >
          {backendConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span className="dot" />
          {backendConnected ? "Backend connected" : "Backend offline"}
        </div>
      </header>

      <div className="input-section">
        <Recorder
          onRecordingComplete={handleRecordingComplete}
          onLiveTranscript={handleLiveTranscript}
          disabled={analyzing}
        />

        {recording && (
          <TranscriptBox transcript={liveTranscript} isLive />
        )}

        <div className="divider">
          <span>or</span>
        </div>

        <form className="text-input-form" onSubmit={handleTextSubmit}>
          <textarea
            className="text-input"
            placeholder="Type something to analyze its emotion…"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            disabled={analyzing || recording}
            rows={3}
          />
          <button
            className="btn btn--primary text-submit"
            type="submit"
            disabled={analyzing || recording || !textInput.trim()}
          >
            <Send size={15} />
            Analyze Text
          </button>
        </form>
      </div>

      {analyzing && (
        <div className="analyzing">
          <Loader2 size={18} className="spinner" />
          Analyzing&hellip;
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="results-grid">
          <TranscriptBox transcript={result.transcript} />

          <PredictionPanel
            title="Detected Emotion"
            prediction={result.prediction}
          />

          <ProbabilityBars
            probabilities={result.prediction.probabilities}
            highlightLabel={result.prediction.label}
          />

          {responseText && (
            <div className={`response-section${speaking ? " is-speaking" : ""}`}>
              <div className="label">
                Response
                {speaking && <span className="speaking-badge">speaking</span>}
              </div>
              <p className="response-text">
                {speaking ? (
                  <>
                    <span className="spoken">{responseText.slice(0, spokenCharIndex)}</span>
                    <span className="unspoken">{responseText.slice(spokenCharIndex)}</span>
                  </>
                ) : (
                  responseText
                )}
              </p>
              <div className="response-actions">
                {speaking ? (
                  <button className="btn" onClick={stopSpeaking}>
                    Stop
                  </button>
                ) : (
                  <button
                    className="btn btn--primary"
                    onClick={() => speakResponse(responseText)}
                  >
                    <Volume2 size={15} />
                    Speak
                  </button>
                )}
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
