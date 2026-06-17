# HMM POS Tagger Mini Project

This project implements a Part-of-Speech (POS) tagger from scratch using a first-order Hidden Markov Model (HMM). It trains and evaluates on the Brown Corpus with the universal POS tagset.

The project includes two web interfaces:

- `docs/`: a static GitHub Pages version that runs fully in the browser using an exported model JSON file.
- `app.py`: a Flask version that trains/loads the model from the bundled Brown corpus files at server startup.

The implementation covers:

- Brown Corpus loading and reproducible train/test splitting
- Initial, transition, and emission probability estimation
- Laplace smoothing for emissions and unseen words
- Log-space Viterbi decoding
- Word-level accuracy evaluation
- Out-of-vocabulary (OOV) analysis
- Error analysis for common POS tag mistakes
- Interactive web-based POS tagging

## Project Structure

```text
.
|-- docs/                           # Static GitHub Pages web app
|   |-- index.html                  # Static web interface
|   |-- model.json                  # Exported HMM model for browser use
|   |-- static/                     # Static CSS and JavaScript decoder
|   `-- report-image/               # Images used by the static web app
|-- app.py                          # Flask web application
|-- hmm_model.py                    # Reusable HMM model and Brown corpus loader
|-- requirements.txt                # Flask app dependencies
|-- project1.ipynb                  # Main implementation notebook
|-- project1-result-check.ipynb     # Notebook with final result/report-check cells
|-- Miniproject_HMM_NLP.pdf         # Original project specification
|-- static/                         # Flask app CSS and JavaScript
|-- templates/                      # Flask HTML templates
|-- nltk_data/                      # Bundled Brown Corpus and universal tagset data
`-- doc/
    `-- latex-report/
        |-- main.tex                # LaTeX project report
        |-- image/                  # Report logo/image assets
        `-- resultImage/            # Generated result figures
```

## Quick Start

### Run the GitHub Pages Version Locally

Use this version when you want the same behavior as GitHub Pages. It does not need Flask, NLTK, or any Python packages.

```bash
cd docs
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Stop the local server with `Ctrl + C`.

Do not open `docs/index.html` by double-clicking it, because browsers may block loading `model.json` from the local filesystem. Use the local server command above.

### Host on GitHub Pages

The easiest public hosting option is the static app in `docs/`. It runs the HMM decoder in browser JavaScript and loads the exported model from `docs/model.json`, so GitHub Pages can host it without Flask or Python.

After pushing the `docs/` folder to GitHub:

1. Open your GitHub repository.
2. Go to **Settings** > **Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select your main branch and the `/docs` folder.
5. Click **Save**.

GitHub will publish the site at a URL like:

```text
https://YOUR_USERNAME.github.io/YOUR_REPO/
```

## Optional Flask Web App

The Flask version is useful if you want a Python server and `/api/tag` endpoint. It reads the bundled Brown corpus files directly from `nltk_data/`, so it does not require `nltk` at runtime.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app from the project root:

```bash
flask --app app run
```

Then open:

```text
http://127.0.0.1:5000
```

The tagging API is available at:

```text
POST /api/tag
```

Example request body:

```json
{
  "text": "The student reads a book."
}
```

## Notebook Requirements

Use Python 3 with these packages for the notebooks:

- `nltk`
- `jupyter`
- `matplotlib` for the result-check notebook and report figures

Install them with:

```bash
pip install nltk jupyter matplotlib
```

The repository includes the required NLTK data under `nltk_data/`. If NLTK does not find it automatically, set:

```bash
export NLTK_DATA="$PWD/nltk_data"
```

The notebooks also call `nltk.download("brown")` and `nltk.download("universal_tagset")`, so they can download the same data if internet access is available.

## Run the Notebooks

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open and run:

1. `project1.ipynb` for the main HMM POS tagger implementation.
2. `project1-result-check.ipynb` to review final outputs and report-ready summaries.

The main notebook follows this workflow:

1. Load the Brown Corpus using the universal tagset.
2. Shuffle sentences with `random.seed(42)`.
3. Split the data into 80% training and 20% testing.
4. Estimate HMM parameters:
   - initial probabilities
   - transition probabilities
   - emission probabilities
5. Apply Laplace smoothing to emissions and add an `<OOV>` token.
6. Decode test sentences with log-space Viterbi.
7. Evaluate word-level accuracy and analyze mistakes.

## Reported Results

Saved notebook outputs report:

| Metric | Value |
| --- | ---: |
| Total sentences | 57,340 |
| Training sentences | 45,872 |
| Testing sentences | 11,468 |
| POS tags | 12 |
| Vocabulary size | 45,153 |
| Test words | 232,177 |
| Correct words | 217,777 |
| Wrong words | 14,400 |
| Accuracy | 93.8% |
| OOV words | 5,050 |
| OOV rate | 2.18% |

Most common saved mistake types:

| True tag | Predicted tag | Count |
| --- | --- | ---: |
| VERB | NOUN | 1,337 |
| NOUN | DET | 1,113 |
| NOUN | ADJ | 986 |
| NOUN | `.` | 709 |
| PRT | ADP | 670 |
| NOUN | VERB | 637 |
| NOUN | PRON | 609 |
| VERB | DET | 560 |
| ADJ | NOUN | 558 |
| VERB | ADJ | 551 |

## Report

The LaTeX report is in `doc/latex-report/main.tex`, with generated result images in `doc/latex-report/resultImage/`.

To build the report from the LaTeX directory:

```bash
cd doc/latex-report
pdflatex main.tex
```

Run `pdflatex` twice if references or figure placement need another pass.

## Review Notes

- The core implementation matches the assignment goal: it trains an HMM from frequency counts and decodes with Viterbi in log-space.
- The split is reproducible because the notebook uses `random.seed(42)` before shuffling.
- Emission probabilities are smoothed, but initial and transition probabilities are not smoothed. The decoder compensates with `EPSILON` before taking logs, but a cleaner model would smooth these distributions explicitly.
- The static app in `docs/` is the recommended hosting path for GitHub Pages.
- The Flask app keeps reusable Python model logic in `hmm_model.py` for server-side use.
- The repository includes both zipped and extracted NLTK data, which improves offline use but increases repository size.

## Possible Improvements

- Add a script to regenerate `docs/model.json` whenever the Python model changes.
- Add transition and initial probability smoothing.
- Add sentence-level accuracy and confusion-matrix reporting.
- Add a simple CLI for tagging custom sentences without opening Jupyter or Flask.
