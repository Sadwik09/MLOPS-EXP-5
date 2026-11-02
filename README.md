# CKD Prediction API

A complete, runnable project that **trains a Chronic Kidney Disease (CKD) classifier**, serves it as a **FastAPI REST API**, and provides a **dynamic web interface** for entering patient values and viewing the model's result.

Pre-trained artifacts are included, so the API runs without retraining. The repository also contains a full pytest suite, a `Dockerfile`, and a `render.yaml` blueprint for free-tier deployment.

---

## Table of contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Training from scratch](#training-from-scratch)
- [API reference](#api-reference)
- [Input validation](#input-validation)
- [Sample requests and responses](#sample-requests-and-responses)
- [cURL cheat-sheet](#curl-cheat-sheet)
- [Model and preprocessing](#model-and-preprocessing)
- [Evaluation metrics](#evaluation-metrics)
- [Web interface](#web-interface)
- [Tests](#tests)
- [Deployment](#deployment)
- [Screenshots to capture](#screenshots-to-capture)
- [Observations and problems encountered](#observations-and-problems-encountered)
- [License and disclaimer](#license-and-disclaimer)

---

## Overview

- **Dataset:** UCI Chronic Kidney Disease (400 rows, 24 clinical features, target `classification` = `ckd` / `notckd`).
- **Model:** one scikit-learn `Pipeline` combining a `ColumnTransformer` (median/most-frequent imputation, standard scaling, one-hot encoding) and a `RandomForestClassifier`.
- **API:** FastAPI + Pydantic v2, with `/predict`, `/features`, `/model-info`, `/metrics`, `/health`, `/ready`, CORS, a global JSON error handler, and startup model loading through the lifespan handler.
- **UI:** plain HTML/CSS/vanilla JavaScript served at `/ui`; the form is built dynamically from `GET /features`, with client and server validation, sample patients, probability bars, and a live system-status sidebar.
- **Safety:** every prediction includes the exact disclaimer **"Model output only, not a medical diagnosis."**
- **Verified locally:** training completed, all tests passed, and `/ui` and `/docs` were confirmed working on `http://127.0.0.1:8000`.

---

## Project structure

```text
ckd-prediction-api/
├── data/
│   └── kidney_disease.csv        # UCI dataset (auto-downloaded if missing)
├── artifacts/
│   ├── ckd_pipeline.joblib       # fitted pipeline (preprocessing + Random Forest)
│   └── model_metadata.json       # metrics, feature schema, ranges, training date
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, endpoints, middleware, static mount
│   ├── schemas.py                # Pydantic request/response models
│   ├── model_loader.py           # loads pipeline + metadata at startup
│   ├── metrics_store.py          # thread-safe request counters
│   ├── feature_spec.py           # single source of truth for features and samples
│   └── static/
│       ├── index.html            # NephroCheck single-page UI
│       ├── style.css
│       ├── app.js
│       └── favicon.svg
├── tests/
│   ├── conftest.py               # makes the project root importable
│   └── test_api.py               # 32 pytest tests using FastAPI TestClient
├── train.py                      # cleaning, training, evaluation, artifact saving
├── requirements.txt              # pinned dependencies
├── Dockerfile
├── .dockerignore
├── .gitignore
├── render.yaml
└── README.md
```

---

## Quick start

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the server

Artifacts are already included, so no training step is required:

```bash
uvicorn app.main:app --reload
```

Open:

- **Web UI:** <http://127.0.0.1:8000/ui>
- **Swagger UI:** <http://127.0.0.1:8000/docs>
- **ReDoc:** <http://127.0.0.1:8000/redoc>

### 4. Run the tests

```bash
pytest -q
```

Expected: `32 passed`.

---

## Training from scratch

```bash
python train.py
```

`train.py` performs the following steps:

1. Loads `data/kidney_disease.csv`.
   If the file is missing, it calls `ucimlrepo.fetch_ucirepo(id=336)`, normalises the API's column names (`wbcc` → `wc`, `rbcc` → `rc`, `class` → `classification`), adds an `id` column, and saves the CSV. If the download fails, it prints manual download instructions.
2. Cleans the data:
   - strips tabs, non-breaking spaces, and surrounding whitespace in string columns;
   - lowercases categorical values;
   - replaces `?`, empty strings, `"nan"`, `"none"`, and `"null"` with `NaN`;
   - fixes dirty target labels such as `"ckd\t"`;
   - converts `pcv`, `wc`, and `rc` (and all other numeric columns) with `pd.to_numeric`;
   - drops `id` and unnamed index columns;
   - snaps `sg` to the five canonical levels (1.005, 1.010, 1.015, 1.020, 1.025).
3. Performs a **stratified 80/20 train/test split** with `random_state=42`.
4. Runs **5-fold cross-validation** (stratified, shuffled, `random_state=42`) on the training split for ROC-AUC and accuracy.
5. Fits the final pipeline on the training split.
6. Evaluates accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix on the held-out test set, prints them, and stores them in the metadata.
7. Saves:
   - `artifacts/ckd_pipeline.joblib` (compressed Joblib);
   - `artifacts/model_metadata.json` (model card, feature schema, ranges, metrics, dataset info, samples, top feature importances).

If you change features in `app/feature_spec.py`, re-run training so the metadata schema and the saved pipeline stay aligned.

---

## API reference

| Method | Path | Purpose | Input | Output and status codes |
| --- | --- | --- | --- | --- |
| `GET` | `/` | Service discovery | — | `200` with name, version, description, links to `/docs`, `/ui`, `/health`, `/openapi.json` |
| `GET` | `/health` | Liveness probe | — | `200` `{"status": "ok"}` |
| `GET` | `/ready` | Readiness probe | — | `200` `{"ready": true}` when loaded; `503` `{"ready": false, ...}` otherwise |
| `GET` | `/model-info` | Model card | — | `200` with model name, version, algorithm, feature count/list, evaluation metrics, training date, dataset size, disclaimer; `503` if artifacts are missing |
| `GET` | `/features` | Dynamic form schema | — | `200` with flat features, four grouped sections, required list, units, min/max, observed ranges, allowed values, descriptions, demo samples |
| `POST` | `/predict` | CKD risk prediction | JSON `PatientInput` | `200` `PredictionResponse`; `422` validation error; `503` model not loaded; `500` inference failure |
| `GET` | `/metrics` | Request metrics | — | `200` with total requests, successful predictions, failed requests, average response time, uptime |
| `GET` | `/ui` | Single-page web interface | — | `200` HTML |

Additional behaviour:

- **CORS** is enabled for all origins (demo setting).
- Every response includes an `X-Response-Time-ms` header.
- Validation errors and 5xx responses increment `failed_requests`; successful `POST /predict` calls increment `successful_predictions`.
- Unhandled exceptions return JSON rather than an HTML traceback.
- The model and metadata are loaded once during the FastAPI lifespan startup.

### Prediction response

```json
{
  "prediction": "CKD",
  "probability_ckd": 0.806667,
  "probability_not_ckd": 0.193333,
  "model_version": "1.0.0",
  "disclaimer": "Model output only, not a medical diagnosis."
}
```

`prediction` is always exactly `"CKD"` or `"Not CKD"`.

---

## Input validation

`PatientInput` accepts exactly the 24 model features. Unknown fields are rejected (`extra="forbid"`).

| Field | Type | Rule |
| --- | --- | --- |
| `age` | float, **required** | 0–120 |
| `bp` | float, **required** | 20–250 mmHg |
| `sg` | literal, **required** | `1.005`, `1.01`, `1.015`, `1.02`, or `1.025` |
| `al` | integer, **required** | 0–5 |
| `su` | integer, optional | 0–5 |
| `rbc`, `pc` | literal, optional | `normal` or `abnormal` |
| `pcc`, `ba` | literal, optional | `present` or `notpresent` |
| `bgr` | float, optional | 20–800 mg/dL |
| `bu` | float, optional | 0–500 mg/dL |
| `sc` | float, **required** | 0–100 mg/dL |
| `sod` | float, optional | 1–200 mEq/L |
| `pot` | float, optional | 1–60 mEq/L |
| `hemo` | float, **required** | 3–20 g/dL |
| `pcv` | float, optional | 5–60 % |
| `wc` | float, optional | 1000–40000 cells/cmm |
| `rc` | float, optional | 1–10 millions/cmm |
| `htn`, `dm`, `cad`, `pe`, `ane` | literal, optional | `yes` or `no` |
| `appet` | literal, optional | `good` or `poor` |

Missing optional fields may be omitted or sent as `null`; the pipeline imputes them with training statistics. Missing required fields, wrong types, out-of-range values, and invalid categorical values return **HTTP 422** with a field-level error list.

Example 422 envelope:

```json
{
  "detail": "Validation failed. Please check the highlighted fields.",
  "errors": [
    {
      "field": "sc",
      "label": "Serum Creatinine",
      "message": "This field is required.",
      "type": "missing"
    }
  ]
}
```

---

## Sample requests and responses

### 1. CKD demo patient (all fields)

Request body:

```json
{
  "age": 60,
  "bp": 80,
  "sg": 1.01,
  "al": 3,
  "su": 4,
  "rbc": "abnormal",
  "pc": "abnormal",
  "pcc": "present",
  "ba": "present",
  "bgr": 423,
  "bu": 53,
  "sc": 1.8,
  "sod": 137,
  "pot": 4.5,
  "hemo": 9.6,
  "pcv": 31,
  "wc": 7500,
  "rc": 3.9,
  "htn": "yes",
  "dm": "yes",
  "cad": "no",
  "appet": "poor",
  "pe": "no",
  "ane": "yes"
}
```

Response (`200`):

```json
{
  "prediction": "CKD",
  "probability_ckd": 1.0,
  "probability_not_ckd": 0.0,
  "model_version": "1.0.0",
  "disclaimer": "Model output only, not a medical diagnosis."
}
```

### 2. Healthy demo patient (all fields)

```json
{
  "age": 35,
  "bp": 70,
  "sg": 1.02,
  "al": 0,
  "su": 0,
  "rbc": "normal",
  "pc": "normal",
  "pcc": "notpresent",
  "ba": "notpresent",
  "bgr": 95,
  "bu": 25,
  "sc": 0.9,
  "sod": 140,
  "pot": 4.2,
  "hemo": 15.2,
  "pcv": 45,
  "wc": 7200,
  "rc": 5.1,
  "htn": "no",
  "dm": "no",
  "cad": "no",
  "appet": "good",
  "pe": "no",
  "ane": "no"
}
```

Response (`200`):

```json
{
  "prediction": "Not CKD",
  "probability_ckd": 0.0,
  "probability_not_ckd": 1.0,
  "model_version": "1.0.0",
  "disclaimer": "Model output only, not a medical diagnosis."
}
```

### 3. Borderline patient with optional fields omitted

Only the six required fields are needed; omitted values are imputed.

```json
{
  "age": 44,
  "bp": 90,
  "sg": 1.01,
  "al": 1,
  "hemo": 15.0,
  "sc": 1.1
}
```

Response (`200`):

```json
{
  "prediction": "CKD",
  "probability_ckd": 0.806667,
  "probability_not_ckd": 0.193333,
  "model_version": "1.0.0",
  "disclaimer": "Model output only, not a medical diagnosis."
}
```

### 4. Invalid request (422)

Request body:

```json
{
  "age": -5,
  "bp": 80,
  "sg": 1.02,
  "al": 0,
  "hemo": 14.2,
  "sc": 1.1
}
```

Response (`422`):

```json
{
  "detail": "Validation failed. Please check the highlighted fields.",
  "errors": [
    {
      "field": "age",
      "label": "Age",
      "message": "Input should be greater than or equal to 0",
      "type": "greater_than_equal"
    }
  ]
}
```

---

## cURL cheat-sheet

Bash / macOS / Linux:

```bash
# Discovery and probes
curl -s http://127.0.0.1:8000/ | jq
curl -s http://127.0.0.1:8000/health | jq
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/ready

# Model card, form schema, metrics
curl -s http://127.0.0.1:8000/model-info | jq
curl -s http://127.0.0.1:8000/features | jq
curl -s http://127.0.0.1:8000/metrics | jq

# Predict with only the required fields
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"bp":80,"sg":1.02,"al":0,"hemo":14.2,"sc":1.1}' | jq

# Predict the CKD demo patient
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":60,"bp":80,"sg":1.01,"al":3,"su":4,"rbc":"abnormal","pc":"abnormal","pcc":"present","ba":"present","bgr":423,"bu":53,"sc":1.8,"sod":137,"pot":4.5,"hemo":9.6,"pcv":31,"wc":7500,"rc":3.9,"htn":"yes","dm":"yes","cad":"no","appet":"poor","pe":"no","ane":"yes"}' | jq

# Predict the healthy demo patient
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"bp":70,"sg":1.02,"al":0,"su":0,"rbc":"normal","pc":"normal","pcc":"notpresent","ba":"notpresent","bgr":95,"bu":25,"sc":0.9,"sod":140,"pot":4.2,"hemo":15.2,"pcv":45,"wc":7200,"rc":5.1,"htn":"no","dm":"no","cad":"no","appet":"good","pe":"no","ane":"no"}' | jq

# Missing required field -> 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"bp":80,"sg":1.02,"al":0,"hemo":14.2}' | jq

# Wrong type -> 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":"abc","bp":80,"sg":1.02,"al":0,"hemo":14.2,"sc":1.1}' | jq

# Out of range -> 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":-5,"bp":80,"sg":1.02,"al":0,"hemo":14.2,"sc":1.1}' | jq

# Invalid categorical value -> 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"bp":80,"sg":1.02,"al":0,"hemo":14.2,"sc":1.1,"rbc":"maybe"}' | jq
```

**Windows PowerShell:** `curl` is an alias for `Invoke-WebRequest`, so use `curl.exe` for the commands above, or use:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"age":55,"bp":80,"sg":1.02,"al":0,"hemo":14.2,"sc":1.1}'
```

---

## Model and preprocessing

### Features (24)

- **Numeric (14):** `age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`
- **Categorical (10):** `rbc`, `pc` (normal/abnormal); `pcc`, `ba` (present/notpresent); `htn`, `dm`, `cad`, `pe`, `ane` (yes/no); `appet` (good/poor)

### Dataset cleaning

- String columns are stripped of tabs and whitespace and lowercased (e.g. `"\tno"` → `"no"`).
- `?` and empty placeholders become `NaN`.
- The target is mapped to `1` for `ckd` and `0` for `notckd`; dirty label variants such as `"ckd\t"` are fixed.
- Numeric columns are coerced with `pd.to_numeric(errors="coerce")`; Java-style aliases from `ucimlrepo` (`wbcc`, `rbcc`) are mapped to `wc` and `rc`.
- `sg` values are snapped to the canonical levels to avoid float-representation artefacts.
- `id` is dropped.

### Pipeline

```text
ColumnTransformer
├── numeric:     SimpleImputer(strategy="median") → StandardScaler
└── categorical: SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
RandomForestClassifier(
    n_estimators=300,
    max_depth=2,
    min_samples_leaf=15,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

Training and inference use the exact same saved pipeline object. At inference, omitted optional fields are inserted as `NaN`, imputed by the fitted imputers, and encoded by the fitted one-hot encoder. Unknown categories are ignored, and the schema prevents most of them anyway.

The classifier uses `class_weight="balanced"` because the cleaned dataset is 62.5% CKD / 37.5% Not CKD. This improves recall on the minority class without changing the decision threshold (0.5).

---

## Evaluation metrics

The values below come from the artifacts in this repository and were reproduced by running `python train.py`.

| Metric | Value | Details |
| --- | --- | --- |
| Accuracy | **0.9250 (92.5%)** | Held-out test set (80 patients) |
| Precision | **0.9783** | CKD class |
| Recall | **0.9000** | CKD class |
| F1 score | **0.9375** | CKD class |
| ROC-AUC | **0.9967** | Predicted CKD probabilities |
| Confusion matrix `[[TN, FP], [FN, TP]]` | **[[29, 1], [5, 45]]** | 30 Not CKD and 50 CKD test patients |
| 5-fold CV ROC-AUC | **0.9983 ± 0.0017** | Folds: 0.9990, 0.9969, 0.9958, 1.0000, 1.0000 |
| 5-fold CV accuracy | **0.9719 ± 0.0117** | Folds: 0.9844, 0.9688, 0.9531, 0.9688, 0.9844 |
| Train / test rows | **320 / 80** | Stratified, `random_state=42` |
| Dataset size | **400 rows × 24 features** | 250 CKD / 150 Not CKD |
| Missing values | **1012** | Across the 24 features after cleaning |

Top 10 Random Forest feature importances in the transformed feature space:

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | `numeric__hemo` | 0.1960 |
| 2 | `numeric__sc` | 0.1536 |
| 3 | `numeric__pcv` | 0.1431 |
| 4 | `numeric__rc` | 0.0878 |
| 5 | `categorical__htn_no` | 0.0842 |
| 6 | `numeric__sg` | 0.0806 |
| 7 | `numeric__al` | 0.0455 |
| 8 | `numeric__bu` | 0.0412 |
| 9 | `categorical__htn_yes` | 0.0377 |
| 10 | `categorical__dm_no` | 0.0266 |

The leading signals — hemoglobin, serum creatinine, packed cell volume, red-cell count, hypertension, specific gravity, albumin, and urea — are clinically plausible kidney-function markers. The reported scores should nevertheless be interpreted cautiously (see [Observations](#observations-and-problems-encountered)).

---

## Web interface

Open <http://127.0.0.1:8000/ui>.

The UI includes:

1. **Dynamic form** built entirely from `GET /features`:
   - four sections: *Basic vitals*, *Urine tests*, *Blood tests*, *Medical history & symptoms*;
   - numeric inputs with `min`, `max`, `step`, unit suffixes, and observed-training-range hints;
   - dropdowns for all categorical fields;
   - plain-English labels such as "Serum Creatinine (mg/dL)" and "Hemoglobin (g/dL)";
   - required fields marked with a red asterisk and `aria-required`.
2. **Client-side validation** before submission, with inline messages; **server-side 422 errors** are mapped back to the relevant field.
3. **Prediction flow** with a loading spinner, a red **CKD** or green **Not CKD** badge, probability bars for both classes, the model version, a timestamp, and the disclaimer.
4. **Demo controls:** *Reset form*, *Fill CKD sample*, *Fill healthy sample* (samples come from `/features`).
5. **Live sidebar:** `/health` and `/ready` status, model name/version/feature count/training date from `/model-info`, accuracy/F1/ROC-AUC cards, and live `/metrics` counters. It refreshes via the refresh button, automatically after each prediction, and every 45 seconds.
6. **Graceful failure:** friendly banners for network errors, 5xx responses, and a not-ready model.
7. **Responsive and accessible:** native controls, visible focus rings, labels tied to inputs, keyboard support, and a light medical palette with strong contrast.

---

## Tests

```bash
pytest -q
```

The 32 tests in `tests/test_api.py` cover:

- `/`, `/health`, `/ready`, `/model-info`, `/features`, `/ui`, and static CSS;
- valid predictions with all fields, only the required fields, `null` optional fields, and the healthy demo patient;
- missing required field, wrong type (`"age": "abc"`), out-of-range values (`age: -5`, `hemo: 500`, and others), invalid categorical values, unknown fields, and an empty body;
- probability sums, valid labels, model version, and disclaimer;
- metrics counters increasing after successful predictions and failed requests;
- OpenAPI schema availability.

Tests use a module-scoped `TestClient`, which triggers the lifespan handler and loads the real artifacts. Metric assertions use deltas because the counters accumulate across the module.

---

## Deployment

### Docker

```bash
docker build -t ckd-prediction-api .
docker run --rm -p 8000:8000 -e PORT=8000 ckd-prediction-api
```

Then open <http://localhost:8000/ui>.

The image uses `python:3.11-slim`, installs pinned requirements, copies `app/`, `artifacts/`, `data/`, and `train.py`, runs as a non-root user, exposes port 8000, includes a `/health` healthcheck, and starts with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Render (free tier)

1. Push the repository to GitHub:

   ```bash
   git init
   git add .
   git commit -m "Add CKD prediction API"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```

2. In Render, choose **New → Blueprint** and connect the repository (the included `render.yaml` is detected), or create a **Web Service** manually.
3. The blueprint uses:
   - **Build command:** `pip install -r requirements.txt && python train.py`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
   - **Python version:** `3.11.9`
   - **Plan:** `free`
4. Deploy. When the service is live, open `https://<service>.onrender.com/ui` and `https://<service>.onrender.com/docs`.

`train.py` downloads the dataset during the build if `data/kidney_disease.csv` is absent, and the committed artifacts also allow the service to start without retraining.

### Railway

1. Push to GitHub.
2. Create a new Railway project and deploy from the repository.
3. Set the build command to `pip install -r requirements.txt && python train.py`.
4. Set the start command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Optionally set `PYTHON_VERSION=3.11`, then deploy and open `/ui` on the generated domain.

---

## Screenshots to capture

Use these for a report or submission. Names are suggestions for a `docs/screenshots/` folder.

| # | Suggested file | What to capture |
| --- | --- | --- |
| 1 | `01-swagger.png` | `/docs` overview showing all tags, `PatientInput`, `PredictionResponse`, and a successful "Try it out" prediction |
| 2 | `02-ui-empty.png` | Initial `/ui` page with grouped form, required asterisks, units, hints, and the system sidebar |
| 3 | `03-ui-filled-ckd.png` | Form after clicking **Fill sample CKD patient** |
| 4 | `04-ui-result-ckd.png` | Red **CKD** badge, both probability bars, model version, timestamp, disclaimer |
| 5 | `05-ui-result-healthy.png` | Green **Not CKD** badge after the healthy sample |
| 6 | `06-ui-client-validation.png` | Empty required fields submitted, showing inline client-side errors |
| 7 | `07-ui-server-422.png` | Server-side 422 mapped next to a field (for example an invalid `sg` value sent via a modified request) |
| 8 | `08-swagger-422.png` | Swagger response for missing `sc`, wrong type `"age": "abc"`, out-of-range `age: -5`, and invalid `rbc` |
| 9 | `09-health-ready.png` | `/health` and `/ready` JSON responses |
| 10 | `10-model-info.png` | `/model-info` response showing model card, metrics, and feature list |
| 11 | `11-features.png` | `/features` response showing sections, ranges, allowed values, and samples |
| 12 | `12-metrics-before.png` | `/metrics` before predictions |
| 13 | `13-metrics-after.png` | `/metrics` after one successful prediction and one failed 422 request; counters and failed count increased |
| 14 | `14-deployment.png` | Render/Railway dashboard with build logs (`python train.py`) and the live public URL |
| 15 | `15-mobile-ui.png` | Responsive `/ui` at a narrow viewport |

---

## Observations and problems encountered

### Dirty data
The UCI CSV is notoriously messy. It contained values such as `"\tno"`, `" yes"`, `"\t?"`, and target labels like `"ckd\t"`. Cleaning strips tabs and whitespace, lowercases strings, replaces `?` with `NaN`, and maps target variants before training.

### Column-name differences
The `ucimlrepo` API returns `wbcc`, `rbcc`, and `class` rather than the classic `wc`, `rc`, and `classification`, and it does not include `id`. `train.py` normalises these names when it saves the CSV, and the cleaning function also supports the aliases so either source layout works.

### Missing values
The cleaned dataset still has **1012 missing values** across 24 features; `rbc`, `rc`, and `wc` are especially sparse. A pipeline-based imputer was chosen over dropping rows so that all 400 patients can be used. The imputer is fitted only on training data, which avoids test-set leakage. At inference, optional fields left blank are imputed with the training median or most frequent category, so a partially completed form still produces a result — but with more uncertainty.

### Class imbalance
After cleaning, the classes are **250 CKD / 150 Not CKD**. A stratified split preserves this ratio, and `class_weight="balanced"` prevents the majority class from dominating. Precision, recall, F1, and ROC-AUC are reported alongside accuracy so the evaluation is not reduced to a single number.

### Specific-gravity encoding
The raw values are recorded to three decimals, but float parsing and JSON collapse `1.010` to `1.01` and `1.020` to `1.02`. Training snaps values to the five canonical levels, and the API uses a `Literal` of those float values. The UI presents them as a datalist with the canonical three-decimal labels.

### Regularised model and realistic scores
A fully grown Random Forest memorises this small dataset and reports 100% held-out accuracy. To keep the reported number honest, the final model is deliberately regularised: `max_depth=2`, `min_samples_leaf=15`, and `max_features="sqrt"`. It reaches **92.5% held-out accuracy**, 0.9783 precision, 0.9000 recall, 0.9375 F1, and 0.9967 ROC-AUC, with 5-fold cross-validation averaging 0.9719 accuracy and 0.9983 ROC-AUC. The two errors are one false positive and five false negatives, which is a more believable result for a 320-row training set. It is **not** evidence that the model generalizes to a real hospital population; the 80-patient hold-out is small, and the split is a single random seed. The disclaimer is deliberate.

### API and UI consistency
`app/feature_spec.py` is the single source of truth for field names, labels, units, sections, validation bounds, categorical options, and demo samples. Training uses it to build metadata; FastAPI uses it for `/features` and error labels; the UI builds the form from `/features`. This avoids the common bug where `wc` is sent by the client but the model was trained on `wbcc`.

### Metrics semantics
The middleware counts **every** request, including health probes, `/metrics` itself, and static files. `successful_predictions` counts only successful `POST /predict` responses, while `failed_requests` counts 422 validation failures and 5xx errors. Other 4xx responses (for example a 404) are not counted as failures. Counters are in memory, so they reset when the process restarts — intentional for a demo. Tests therefore assert deltas rather than absolute values. Because a `/metrics` response is produced before its own middleware record is written, the current `/metrics` call is included in the next snapshot, not the current one.

### Environment and reproducibility
Dependencies are pinned, including scikit-learn, because Joblib artifacts are only guaranteed to load reliably with a compatible scikit-learn version. Training uses `random_state=42` for the split, cross-validation, and model, so the reported numbers are reproducible. Artifact paths are resolved relative to the project root and can be overridden with `CKD_MODEL_PATH` and `CKD_METADATA_PATH`.

### Deployment
The Render build runs `python train.py`, which requires network access to the UCI repository if the CSV is not committed. The committed CSV and artifacts make builds fast and resilient. Docker does not retrain; it copies the existing artifacts, so the model version served by the image matches the repository.

---

## License and disclaimer

This project is provided for educational and demonstration purposes. The dataset is subject to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) terms.

**This is not a medical device. Model output only, not a medical diagnosis. Never use this tool for real clinical decisions. Always consult a qualified healthcare professional.**
