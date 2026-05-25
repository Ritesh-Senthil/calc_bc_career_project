import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, Square } from "lucide-react";

function formatTime(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

const PREFERRED_MIME = "audio/webm";
const MAX_DURATION = 30;

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

export default function Recorder({ onRecordingComplete, onLiveTranscript, disabled }) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const intervalRef = useRef(null);
  const recognitionRef = useRef(null);
  const finalTranscriptRef = useRef("");

  const cleanup = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch {}
      recognitionRef.current = null;
    }
    recorderRef.current = null;
    chunksRef.current = [];
    finalTranscriptRef.current = "";
    setElapsed(0);
    setIsRecording(false);
  }, []);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const startRecording = useCallback(async () => {
    setPermissionDenied(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (onLiveTranscript) onLiveTranscript("");
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported(PREFERRED_MIME)
        ? PREFERRED_MIME
        : undefined;
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeType || "audio/webm",
        });
        cleanup();
        if (blob.size > 0) onRecordingComplete(blob);
      };

      recorder.start(250);
      setIsRecording(true);

      if (SpeechRecognitionAPI && onLiveTranscript) {
        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        finalTranscriptRef.current = "";

        recognition.onresult = (event) => {
          let interim = "";
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const text = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscriptRef.current += text;
            } else {
              interim += text;
            }
          }
          onLiveTranscript(finalTranscriptRef.current + interim);
        };

        recognition.onerror = () => {};
        recognition.onend = () => {
          if (recorderRef.current?.state === "recording") {
            try {
              recognition.start();
            } catch {}
          }
        };

        try {
          recognition.start();
          recognitionRef.current = recognition;
        } catch {}
      }

      let sec = 0;
      intervalRef.current = setInterval(() => {
        sec += 1;
        setElapsed(sec);
        if (sec >= MAX_DURATION && recorderRef.current?.state === "recording") {
          recorderRef.current.stop();
        }
      }, 1000);
    } catch {
      setPermissionDenied(true);
      cleanup();
    }
  }, [onRecordingComplete, onLiveTranscript, cleanup]);

  const stopRecording = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch {}
      recognitionRef.current = null;
    }
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  return (
    <div className="recorder">
      {!isRecording ? (
        <button
          className="record-btn"
          onClick={startRecording}
          disabled={disabled}
          aria-label="Start recording"
        >
          <Mic size={26} />
        </button>
      ) : (
        <>
          <button
            className="stop-btn"
            onClick={stopRecording}
            aria-label="Stop recording"
          >
            <Square size={20} />
          </button>
          <div className="recording-indicator">
            <span className="pulse-dot" />
            <span className="timer">{formatTime(elapsed)}</span>
          </div>
        </>
      )}
      {permissionDenied && (
        <p className="permission-error">
          Microphone access denied. Please allow microphone permissions and try
          again.
        </p>
      )}
    </div>
  );
}
