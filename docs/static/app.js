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

const tokenPattern = /[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]/g;
let sampleIndex = 0;
let model = null;
let vocab = null;

function normalizeWord(word) {
  return word.toLowerCase();
}

function tokenize(text) {
  return text.match(tokenPattern) || [];
}

function formatProbability(value) {
  return `${(Number(value) * 100).toFixed(4)}%`;
}

function emissionProbability(tag, word) {
  const normalized = normalizeWord(word);
  const known = vocab.has(normalized);
  const denominator = model.tag_counts[tag] + model.alpha * (model.vocab_size + 1);
  const count = known ? (model.emission_counts[tag][normalized] || 0) : 0;
  return {
    probability: (count + model.alpha) / denominator,
    known
  };
}

function predict(words) {
  if (!words.length) {
    return [];
  }

  const tags = model.tags;
  const viterbi = [];
  const backpointer = [];
  viterbi.push({});
  backpointer.push({});

  for (const tag of tags) {
    const emission = emissionProbability(tag, words[0]).probability;
    viterbi[0][tag] = Math.log((model.pi[tag] || 0) + model.epsilon) + Math.log(emission + model.epsilon);
    backpointer[0][tag] = null;
  }

  for (let position = 1; position < words.length; position += 1) {
    viterbi.push({});
    backpointer.push({});

    for (const currentTag of tags) {
      const emission = emissionProbability(currentTag, words[position]).probability;
      let bestPreviousTag = null;
      let bestScore = Number.NEGATIVE_INFINITY;

      for (const previousTag of tags) {
        const transition = model.transition_probs[previousTag][currentTag] || 0;
        const score = viterbi[position - 1][previousTag]
          + Math.log(transition + model.epsilon)
          + Math.log(emission + model.epsilon);

        if (score > bestScore) {
          bestScore = score;
          bestPreviousTag = previousTag;
        }
      }

      viterbi[position][currentTag] = bestScore;
      backpointer[position][currentTag] = bestPreviousTag;
    }
  }

  let bestFinalTag = tags[0];
  for (const tag of tags) {
    if (viterbi[viterbi.length - 1][tag] > viterbi[viterbi.length - 1][bestFinalTag]) {
      bestFinalTag = tag;
    }
  }

  const bestSequence = [bestFinalTag];
  for (let position = words.length - 1; position > 0; position -= 1) {
    bestSequence.push(backpointer[position][bestSequence[bestSequence.length - 1]]);
  }
  bestSequence.reverse();

  return words.map((word, index) => {
    const tag = bestSequence[index];
    const emission = emissionProbability(tag, word);
    return {
      word,
      tag,
      known: emission.known,
      emission_probability: emission.probability
    };
  });
}

function renderPredictions(tokens) {
  const oovTotal = tokens.filter((item) => !item.known).length;
  tokenCount.textContent = `${tokens.length} token${tokens.length === 1 ? "" : "s"}`;
  oovCount.textContent = `${oovTotal} OOV`;

  if (!tokens.length) {
    predictionBody.innerHTML = '<tr><td colspan="4" class="empty">No tokens found.</td></tr>';
    return;
  }

  predictionBody.innerHTML = tokens.map((item) => `
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

async function loadModel() {
  const response = await fetch("model.json");
  if (!response.ok) {
    throw new Error("Could not load model.json.");
  }
  model = await response.json();
  vocab = new Set(model.vocab);
  message.textContent = "Static model loaded. Ready to tag.";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!model) {
    message.textContent = "Model is still loading.";
    return;
  }

  const words = tokenize(textarea.value.trim());
  const predictions = predict(words);
  renderPredictions(predictions);
  message.textContent = "Prediction complete.";
});

sampleButton.addEventListener("click", () => {
  sampleIndex = (sampleIndex + 1) % samples.length;
  textarea.value = samples[sampleIndex];
  textarea.focus();
});

loadModel().catch((error) => {
  message.textContent = error.message;
});
