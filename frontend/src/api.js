async function postAnalyze(formData) {
  const res = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let message = `Analysis failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) message = body.detail;
    } catch {}
    throw new Error(message);
  }

  return res.json();
}

export async function analyzeAudio(audioBlob, browserTranscript = "") {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.webm");
  if (browserTranscript) {
    formData.append("transcript", browserTranscript);
  }
  return postAnalyze(formData);
}

export async function analyzeText(text) {
  const formData = new FormData();
  formData.append("transcript", text);
  return postAnalyze(formData);
}

export async function checkHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Backend unavailable");
  return res.json();
}
