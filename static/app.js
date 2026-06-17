const form = document.querySelector("#tag-form");
const textarea = document.querySelector("#sentence");
const message = document.querySelector("#message");
const predictionBody = document.querySelector("#prediction-body");
const tokenCount = document.querySelector("#token-count");
const oovCount = document.querySelector("#oov-count");
const sampleButton = document.querySelector("#sample-button");

const samples = [
  "The student reads a book.",
  "Natural language processing helps computers understand text.",
  "Atlanta's recent election produced no evidence of irregularities."
];

let sampleIndex = 0;

function formatProbability(value) {
  return `${(Number(value) * 100).toFixed(4)}%`;
}

function renderPredictions(data) {
  tokenCount.textContent = `${data.token_count} token${data.token_count === 1 ? "" : "s"}`;
  oovCount.textContent = `${data.oov_count} OOV`;

  if (!data.tokens.length) {
    predictionBody.innerHTML = '<tr><td colspan="4" class="empty">No tokens found.</td></tr>';
    return;
  }

  predictionBody.innerHTML = data.tokens.map((item) => `
    <tr>
      <td>${escapeHtml(item.word)}</td>
      <td><span class="tag">${escapeHtml(item.tag)}</span></td>
      <td><span class="${item.known ? "known" : "oov"}">${item.known ? "Known" : "OOV"}</span></td>
      <td>${formatProbability(item.emission_probability)}</td>
    </tr>
  `).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textarea.value.trim();
  message.textContent = "Running Viterbi decoder...";

  try {
    const response = await fetch("/api/tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Tagging failed.");
    }

    renderPredictions(data);
    message.textContent = "Prediction complete.";
  } catch (error) {
    message.textContent = error.message;
  }
});

sampleButton.addEventListener("click", () => {
  sampleIndex = (sampleIndex + 1) % samples.length;
  textarea.value = samples[sampleIndex];
  textarea.focus();
});
