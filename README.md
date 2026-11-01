# 10K-MetricsBot

Numeric question answering and risk classification for SEC 10-K filings.
Built for the AIML lab ESE (Experiments 04, 05, 06/08).

![UI](docs/ui.png)

## The problem

Annual reports run to hundreds of pages of dense accounting text. Analysts need exact figures
("What were total net sales?") and a way to tag management statements by risk type, without a model
inventing numbers.

## What it does

| Experiment | Component | Details |
|---|---|---|
| Exp 04 (classification) | Risk classifier | DistilBERT fine-tuned to label sentences as Credit Risk, Liquidity Risk, Operational Risk or Market Growth |
| Exp 05 (QA) | Extractive vs generative | DistilBERT (SQuAD) copies exact spans from the filing. FLAN-T5-small generates text. Both compared on 20 numeric questions |
| Exp 06/08 (deployment) | FastAPI + Docker | `POST /query-filing` with Swagger UI, a web page, and a Dockerfile |

**Design rule:** numbers in the API response come only from the extractive model, so every answer is
literal text from the filing. FLAN-T5 is used only for a short restatement of the source sentence.

![Swagger](docs/swagger.png)

## Data

- Apple Inc. Form 10-K, fiscal year ended Sept 27, 2025 (about 42k words)
- Bank of America Form 10-K, fiscal year 2025 (about 198k words)
- Downloaded from SEC EDGAR as PDF and converted with `src/pdf_to_txt.py`.
- Classifier data: 597 sentences (about 150 per class), labelled by keyword rules (`src/prepare_data.py`).
  Labels are weak labels. <Edit this line if you manually reviewed or corrected them.>

## Results

### Risk classifier (120 held-out sentences)

Accuracy 0.90, macro F1 0.90.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit Risk | 0.90 | 0.90 | 0.90 |
| Liquidity Risk | 0.97 | 1.00 | 0.98 |
| Operational Risk | 0.90 | 0.93 | 0.92 |
| Market Growth | 0.82 | 0.77 | 0.79 |

Accuracy by source: Bank of America 0.91, Apple 0.88. Full report: `eval/classifier_report.txt`.

### Extractive vs generative QA (20 questions)

Correct answers. A answer counts as correct if the first number in it equals the gold figure.
"Oracle passage" means each model is handed a passage known to contain the answer, which isolates the reader.

| Setting | Question type | n | Extractive | Generative |
|---|---|---|---|---|
| End-to-end (retrieval + reader) | Table figures | 11 | 3 | 3 |
| End-to-end (retrieval + reader) | Facts stated in sentences | 9 | 7 | 6 |
| Oracle passage | Table figures | 11 | 3 | 3 |
| Oracle passage | Facts stated in sentences | 9 | 6 | 7 |

Numbers that did not appear in the passage the model was given: extractive 0 in every setting,
generative 1 (for example `15.408,095` for Apple's diluted EPS in an earlier run).
Details: `eval/qa_summary.txt`, `eval/qa_results.json`.

**Findings**

- Neither model is clearly more accurate. With 20 questions, differences of one or two answers are noise.
- Both models do far better on facts stated in sentences than on table figures. PDF extraction flattens
  tables into number strings, so the model cannot tell which column is the current year.
- The extractive model can only return text that exists in the filing, so it cannot invent a number.
  It can still return a real number from the wrong place (for example a prior-year column).
  The generative model can produce numbers that exist nowhere in the source.

## Limitations

- Small evaluation set (20 questions, 2 filings). Results are indicative, not conclusive.
- Table figures are weak because of flattened PDF tables. Converting table rows to "label: value" text is future work.
- Classifier labels come from keywords, and no hand-written test set was run, so the 0.90 score partly measures agreement with those rules.
- The classifier has no "none of these" class, so every passage gets a risk label, even results-of-operations text.
- FLAN-T5-small is a weak summarizer. It mostly restates the source sentence.
- Not financial advice. Always check answers against the source passage.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

1. Put the filing PDFs in `data/` as `tech_10k.pdf` and `bank_10k.pdf`, then convert them:
   `python src/pdf_to_txt.py`
   (The converted `.txt` files are already in the repo.)
2. Build the labelled dataset: `python src/prepare_data.py`
3. Train the risk classifier (about 4 minutes on CPU, saves to `models/risk_classifier`):
   `python src/train_classifier.py`
4. Evaluate QA: `python src/qa_compare.py`, then `python src/report.py`
5. Start the service: `uvicorn app.main:app`
   - Web page: http://127.0.0.1:8000/
   - Swagger UI: http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query-filing -H "Content-Type: application/json" \
  -d '{"question": "How many full-time equivalent employees did Apple have as of September 27, 2025?", "filing_name": "tech_10k"}'
```

## Docker

Train the classifier first (step 3 above), because the image copies `models/risk_classifier`.

```bash
docker build -t 10k-metricsbot .
docker run -p 8000:8000 10k-metricsbot
```

## Project structure

```
app/main.py            FastAPI service (/query-filing, /classify-risk, /filings, /health)
app/static/index.html  Web page
src/prepare_data.py    Weak-labelling of sentences
src/train_classifier.py  DistilBERT fine-tuning (Exp 04)
src/retrieval.py       BM25 passage retrieval
src/qa_compare.py      Extractive vs generative QA and evaluation (Exp 05)
src/report.py          Table vs prose breakdown
src/pdf_to_txt.py      PDF to text
eval/                  Test questions, results and reports
Dockerfile             Container image (Exp 06/08)
```

Author: <Swayam Mahindrakar>, <SPIT / EXTC>