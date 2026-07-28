# Crypto Candlestick Up/Down Classifier

An end-to-end ML pipeline that predicts short-term price direction ("Up" or
"Down") from a candlestick chart image, covering data acquisition,
preprocessing, training, evaluation, a retrainable model served via a
FastAPI backend, and a Streamlit dashboard for prediction, data insights,
bulk upload, and retraining.

- **Video Demo:** [YOUTUBE LINK PLACEHOLDER]
- **Deployed API URL:** [DEPLOYED URL PLACEHOLDER]
- **Deployed UI URL:** [DEPLOYED URL PLACEHOLDER]

## 1. Project Overview

**Dataset:** [Candlestick Image Data](https://www.kaggle.com/datasets/raimiazeezbabatunde/candle-image-data)
by Raimi Azeez Babatunde (Apache 2.0). Each image renders a 10-candle
formation; the label is `Up` or `Down` depending on whether the cumulative
return over the following 5 days was positive or negative. 1,433 training
images (809 Up / 624 Down) and 351 test images (194 Up / 157 Down).

**Use case:** a finance-domain image classification problem -- given a
recent candlestick formation, predict whether price moves up or down over
the next 5 days. Predicting price direction from chart shape alone is a
genuinely hard problem (markets are close to informationally efficient), so
results are reported honestly rather than tuned toward an inflated number
(see the notebook's Evaluation section).

## 2. Architecture

```
Browser
  │
  ▼
Streamlit UI (ui/app.py, :8501) ──HTTP──► nginx gateway (:8000) ──► FastAPI backend (src/api.py, :8000)
                                                  │                          │
                                     round-robins across scaled             ├─► src/prediction.py  (inference)
                                     `api` replicas for the                 ├─► src/model.py        (train/retrain)
                                     Locust flood test                      ├─► src/database.py     (SQLite upload log)
                                                                             └─► models/candlestick_model.h5
```

- **`nginx`** is a fixed public gateway (host port 8000) that re-resolves the
  `api` service name every 5s, so it load-balances across however many `api`
  containers are running -- this is what makes the "different container
  counts" Locust comparison below actually work with plain `docker-compose
  --scale`.
- **Retraining** runs synchronously inside the `/retrain` endpoint (FastAPI
  runs sync route handlers in a threadpool, so `/health` stays responsive
  while it runs). Deliberately not a background job queue -- fine-tuning on
  this dataset size finishes in well under a minute, so that complexity
  isn't worth it here.
- **Auto-retrain trigger:** in addition to the UI's manual "Trigger Model
  Retraining" button, `GET /insights` and the response of `POST
  /upload-data` include `auto_retrain_recommended`, which flips `true` once
  20+ uploaded samples are pending (see `src/database.AUTO_RETRAIN_THRESHOLD`).

## 3. Repository Structure

```
crypto_ml_pipeline/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── nginx/nginx.conf
├── notebook/crypto_candlestick_classification.ipynb
├── src/
│   ├── preprocessing.py   # loading, augmentation, 3 feature visualizations
│   ├── model.py           # build/train/evaluate/save/load/retrain
│   ├── prediction.py      # single-image inference
│   ├── database.py        # SQLite upload log + auto-retrain trigger
│   └── api.py              # FastAPI app
├── data/{train,test}/{Up,Down}/
├── models/candlestick_model.h5
└── ui/app.py               # Streamlit dashboard
```

## 4. Setup Instructions

### Local (no Docker)

```bash
cd crypto_ml_pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Train the initial model (or open the notebook and run all cells)
jupyter notebook notebook/crypto_candlestick_classification.ipynb

# Run the API (from the crypto_ml_pipeline/ directory)
uvicorn src.api:app --reload --port 8000

# In a second terminal, run the UI
set API_URL=http://localhost:8000        # Windows
# export API_URL=http://localhost:8000   # macOS/Linux
streamlit run ui/app.py
```

Visit `http://localhost:8501` for the dashboard and `http://localhost:8000/docs`
for interactive API documentation (Swagger UI).

### Docker

```bash
cd crypto_ml_pipeline
docker-compose up --build
```

- UI: `http://localhost:8501`
- API (through the nginx gateway): `http://localhost:8000`

## 5. API Documentation

| Method | Endpoint       | Description                                                                 |
|--------|----------------|-------------------------------------------------------------------------------|
| GET    | `/health`      | Status, whether the model file is loaded, whether a retrain is in progress |
| GET    | `/uptime`      | Process uptime, start time, model version, last retrain timestamp          |
| POST   | `/predict`     | `multipart/form-data`, field `file` (PNG/JPG) → `{prediction, confidence, raw_probability}` |
| POST   | `/upload-data` | `multipart/form-data`, field `label` (`Up`/`Down`) + `files` (multiple) → saves to `data/train/<label>/`, logs to SQLite |
| POST   | `/retrain`     | Fine-tunes the current model on `data/train/`, evaluates before/after, overwrites the `.h5` file |
| GET    | `/insights`    | Train/test class counts + upload stats (for the dashboard's Data Insights tab) |

Full interactive docs at `/docs` once the API is running.

## 6. Locust Flood Simulation

`locustfile.py` sends a weighted mix of `POST /predict` (using real sample
images from `data/test/`) and `GET /health` requests.

**Container-count comparison** (run from the project root, one scale value
at a time):

```bash
docker-compose up -d --build --scale api=1
docker-compose --profile loadtest run --rm locust \
    -f locustfile.py --host http://nginx:80 --headless -u 100 -r 10 -t 60s
docker-compose down

docker-compose up -d --build --scale api=2
docker-compose --profile loadtest run --rm locust \
    -f locustfile.py --host http://nginx:80 --headless -u 100 -r 10 -t 60s
docker-compose down

docker-compose up -d --build --scale api=4
docker-compose --profile loadtest run --rm locust \
    -f locustfile.py --host http://nginx:80 --headless -u 100 -r 10 -t 60s
docker-compose down
```

Or with the interactive Locust web UI, from the project root against a
locally running API: `locust -f locustfile.py --host http://localhost:8000`,
then open `http://localhost:8089`.

### Results

*[PLACEHOLDER — fill in after running the comparison above]*

| API containers | Requests/s | Median latency (ms) | p95 latency (ms) | Failure rate |
|-----------------|-----------|----------------------|-------------------|--------------|
| 1               |           |                      |                   |              |
| 2               |           |                      |                   |              |
| 4               |           |                      |                   |              |

## 7. Notebook

`notebook/crypto_candlestick_classification.ipynb` contains: data loading &
preprocessing, the 3 feature-interpretation visualizations, model training
(MobileNetV2 transfer learning, Dropout, Adam, EarlyStopping), full
evaluation (Accuracy, Loss, Precision, Recall, F1, Confusion Matrix, ROC
curve), model persistence, and a single-datapoint inference check.

## 8. Deployment

*[PLACEHOLDER — cloud platform, deploy steps, and any production evaluation notes go here]*
