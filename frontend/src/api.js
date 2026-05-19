export async function analyzeAudio(audioBlob) {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.webm");

  const res = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let message = `Analysis failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) message = body.detail;
    } catch {
      /* response wasn't JSON */
    }
    throw new Error(message);
  }

  return res.json();
}

export async function checkHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Backend unavailable");
  return res.json();
}
