# HMM POS Tagger Mini Project

This project implements a Part-of-Speech (POS) tagger from scratch using a first-order Hidden Markov Model (HMM). It trains and evaluates on the NLTK Brown Corpus with the universal POS tagset.

The project now also includes a Flask web tool for trying the trained HMM tagger on custom sentences and reviewing the model details, dataset statistics, and saved evaluation charts.

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
|-- app.py                          # Flask web application
|-- hmm_model.py                    # Reusable HMM model and Brown corpus loader
|-- requirements.txt                # Web app dependency list
|-- project1.ipynb                  # Main implementation notebook
|-- project1-result-check.ipynb     # Notebook with final result/report-check cells
|-- Miniproject_HMM_NLP.pdf         # Original project specification
|-- static/                         # Web app CSS and JavaScript
|-- templates/                      # Flask HTML templates
|-- nltk_data/                      # Bundled Brown Corpus and universal tagset data
`-- doc/
    `-- latex-report/
        |-- main.tex                # LaTeX project report
        |-- image/                  # Report logo/image assets
        `-- resultImage/            # Generated result figures
```

## Requirements

### Web Application

Use Python 3 with Flask:

```bash
pip install -r requirements.txt
```

The web app reads the bundled Brown corpus files directly from `nltk_data/`, so it does not require `nltk` at runtime.

### Notebooks

Use Python 3 with these packages:

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

## How to Run

### Run the Web Tool

From the project root:

```bash
flask --app app run
```

Then open:

```text
http://127.0.0.1:5000
```

The web interface includes:

- A sentence input box for custom POS tagging
- Word-by-word predicted universal POS tags
- Known/OOV status for each token
- Emission probability shown for each predicted tag
- Model details and reported notebook result charts

The tagging API is also available at:

```text
POST /api/tag
```

Example request body:

```json
{
  "text": "The student reads a book."
}
```

### Run the Notebooks

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
- The web app moves the reusable model logic into `hmm_model.py`, while the notebooks remain as the original implementation/report workflow.
- `requirements.txt` currently contains the Flask dependency needed for the web tool.
- The repository includes both zipped and extracted NLTK data, which improves offline use but increases repository size.

## Possible Improvements

- Add a notebook-to-module synchronization workflow so future model changes do not need to be copied manually.
- Add a production WSGI configuration for deployment.
- Add transition and initial probability smoothing.
- Add sentence-level accuracy and confusion-matrix reporting.
- Add a simple CLI for tagging custom sentences without opening Jupyter or Flask.
