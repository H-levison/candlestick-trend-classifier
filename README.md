# Candlestick Trend Classifier

An end-to-end ML pipeline that classifies whether a candlestick chart image
visually shows a rising ("Up") or falling ("Down") trend across the candles
drawn in it, covering data acquisition, preprocessing, training, evaluation,
a retrainable model served via a FastAPI backend, and a Streamlit dashboard
for prediction, data insights, bulk upload, and retraining.

- **Video Demo:** [https://youtu.be/OOkqFpJaSPs]
- **Deployed API URL:** https://candlestick-trend-classifier.onrender.com/
- **Deployed UI URL:** https://candlestick-trend-classifier-1.onrender.com/

## 1. Project Overview

**Dataset:** [Candlestick Image Data](https://www.kaggle.com/datasets/raimiazeezbabatunde/candle-image-data)
by Raimi Azeez Babatunde (Apache 2.0). Each image renders a 10-candle
formation across several tickers (QQQ, SPY, ...). 1,433 training images and
351 test images.

**Labels and why they were changed from the source dataset:** the dataset
ships labels of `Up`/`Down` based on the sign of the cumulative return over
the 5 days *after* the shown candles -- a future outcome not visually
present in the image. We trained against that label first and investigated
the result rigorously (two-phase MobileNetV2 transfer learning, class-
balance and resolution checks, an independent classical-CV cross-check):
test ROC-AUC came out to 0.436, i.e. no signal distinguishable from chance.
That's an expected, defensible result -- predicting future price direction
from chart shape alone runs into market efficiency -- but it doesn't
demonstrate a working classifier, which the assignment requires. Full
investigation is kept in the notebook (Section 4) as documented evidence of
the process, not deleted.

We then **relabeled every image by the visual trend it actually shows**:
`scripts/relabel_by_visual_trend.py` fits a least-squares line through the
row-position of non-background ("ink") pixels across the image's columns
(classical image processing on the alpha channel, not a trained model, so
there's no circularity in using it as a label source) -- a positive slope
(price position moving down the image, left to right) labels the image
`Down`, negative labels it `Up`. This is now a task where the label is
directly computable from the pixels, which is exactly why it works: it's a
visual pattern-recognition problem, not a forecast. The full before/after
label for all 1,784 images is logged in `relabel_manifest.csv` for
auditability. Resulting split: 824 Up / 609 Down (train), 225 Up / 126 Down
(test). Only 48.0%/51.9% of images kept their original future-return label
after relabeling -- itself an independent, non-ML confirmation that visual
shape and future return are essentially unrelated in this data.

**Use case:** a finance-domain image classification problem -- given a
candlestick chart image, classify the trend pattern it visually depicts.
This is deliberately a description task (what does this chart show), not a
prediction task (what will the price do next) -- see above for why.

**Final model performance** (MobileNetV2 transfer learning, partial-backbone
fine-tuning, validation-tuned decision threshold -- see notebook Sections 3
and 4.1): test accuracy **80.6%**, precision 84.6%, recall 85.3%, F1 85.0%,
ROC-AUC **0.870**. For contrast, the same architecture scored ROC-AUC 0.436
(chance) on the original future-return label -- the gap is the actual
evidence the pivot fixed a genuine no-signal problem, not a modeling issue.

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
                                                                             └─► models/candlestick_model.keras
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
candlestick-trend-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── nginx/nginx.conf
├── notebook/candlestick_trend_classification.ipynb
├── scripts/relabel_by_visual_trend.py  # one-time visual-trend relabeling (see Section 1)
├── relabel_manifest.csv                # audit log: every image's original vs visual-trend label
├── src/
│   ├── preprocessing.py   # loading, augmentation, visual-trend labeling, 3 feature visualizations
│   ├── model.py           # build/train/evaluate/save/load/retrain
│   ├── prediction.py      # single-image inference
│   ├── database.py        # SQLite upload log + auto-retrain trigger
│   └── api.py              # FastAPI app
├── data/{train,test}/{Up,Down}/
├── models/candlestick_model.keras
└── ui/app.py               # Streamlit dashboard
```

> **Note on model file format:** the assignment brief names `.pkl`/`.tf`/`.h5`
> as example formats. This project uses `.keras` (the native Keras 3 format)
> instead: legacy `.h5` full-model saving hits a "cannot pickle 'module'
> object" error partway through in the Keras 3 / TensorFlow 2.21 combination
> used here (a known Keras 3 legacy-H5 serialization issue), and
> `save_format="tf"` SavedModel export raises `ValueError: The save_format
> argument is deprecated in Keras 3` outright. `.keras` is Keras's own
> actively-maintained replacement for both and is used identically
> (`model.save(...)` / `tf.keras.models.load_model(...)`).

## 4. Setup Instructions

### Local (no Docker)

```bash
cd candlestick-trend-classifier
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Train the initial model (or open the notebook and run all cells)
jupyter notebook notebook/candlestick_trend_classification.ipynb

# Run the API (from the candlestick-trend-classifier/ directory)
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
cd candlestick-trend-classifier
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
| POST   | `/retrain`     | Fine-tunes the current model on `data/train/`, evaluates before/after, overwrites the `.keras` file |
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


| API containers | Requests/s | Median latency (ms) | p95 latency (ms) | Failure rate |
|-----------------|-----------|----------------------|-------------------|--------------|
| 1               |    15.07  |         4900         |   6000            |     0.00%    |
| 2               |    13.33  |         5700         |   7400            |     0.00%    |
| 4               |    13.36  |         5800         |   6500            |     0.00%    |

Throughput stays roughly flat (and latency actually *rises*) as container
count goes from 1 to 4, rather than improving the way horizontal scaling
normally would. This was run locally via `docker-compose --scale`, where
every `api` replica shares the same host machine's CPU cores. Inference
here is CPU-bound (TensorFlow-cpu, no GPU), so adding more containers on a
fixed number of physical cores doesn't add real parallel capacity -- it
just adds scheduling contention between replicas competing for the same
cores, which is consistent with the latency increase observed. True
horizontal scaling benefit would require the replicas to run across
separate hosts/cores, e.g. multiple machines or a cloud autoscaling group,
not `--scale` on one laptop.

## 7. Notebook

`notebook/candlestick_trend_classification.ipynb` contains: data loading &
preprocessing, the 3 feature-interpretation visualizations, model training
(MobileNetV2 transfer learning, Dropout, Adam, EarlyStopping), full
evaluation (Accuracy, Loss, Precision, Recall, F1, Confusion Matrix, ROC
curve), model persistence, and a single-datapoint inference check.

## 8. Deployment

Deployed on [Render](https://render.com) as two separate Docker-based Web
Services built from this repo's `Dockerfile`: one running the FastAPI
backend (`uvicorn src.api:app`), one running the Streamlit UI (`streamlit
run ui/app.py`), with the UI's `API_URL` environment variable pointed at
the API service's Render URL.

- **Port binding:** Render assigns its own port via the `$PORT` environment
  variable and routes to it; the `Dockerfile`'s `CMD` reads `$PORT` (falling
  back to 8000 for local `docker-compose`, which sets its own command
  regardless) so the container listens wherever Render expects.
- **Memory:** Render's free instance tier caps the container at 512MB RAM,
  which is tight for a TensorFlow-cpu service (the base import alone
  commonly uses 300-500MB). `src/preprocessing.py`'s matplotlib/scipy
  imports (notebook/UI-only) were moved to lazy imports so the API service
  doesn't pay for them, and TensorFlow's internal thread pools are capped
  to reduce per-thread overhead -- see `src/model.py`. If the service still
  OOMs under load, the reliable fix is a paid instance with more RAM, not
  further micro-optimization of a library with an inherently large
  footprint.
