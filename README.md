# Boston Rideshare Price API

Serves the notebook's `RandomForestRegressor` (n_estimators=200, max_depth=20,
min_samples_split=45, min_samples_leaf=15 — test MAPE ≈ 7.85%) behind a Flask
API, containerized with Docker.

```
rideshare_api/
├── train.py             # reproduces the notebook's pipeline, saves artifacts/
├── app.py                # Flask API that loads artifacts/ and serves predictions
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── artifacts/             # created by train.py — model, scaler, encoder, config
```

The `app.py` logic has been tested against a synthetic dataset matching the
real schema (health check, schema endpoint, single/batch predict, missing-field
validation, and unseen-category handling all verified). Docker build/run
itself needs to be tested in your own environment, since it wasn't available
here — see the note at the bottom.

## 1. Generate the artifacts (run this once)

This sandbox has no internet access, so I couldn't download the Kaggle
dataset to train the real model. Run this step yourself, wherever you have
the CSV — e.g. in the same Colab notebook, or locally:

```bash
# if running where opendatasets already pulled the data (as in your notebook):
python train.py --csv uber-and-lyft-dataset-boston-ma/rideshare_kaggle.csv

# or point it at wherever your CSV actually lives:
python train.py --csv /path/to/rideshare_kaggle.csv
```

This prints train/val/test MAPE (should match the notebook: ~7.3% / ~7.9% /
~7.85%) and writes four files into `artifacts/`:

- `model.joblib`
- `scaler.joblib`
- `encoder.joblib`
- `feature_config.json`

If you're running this in Colab, download the `artifacts/` folder afterward
and place it inside this project directory before building the Docker image.

**Version note:** the packages used to train the model must be able to
unpickle correctly inside the container. `requirements.txt` pins
`scikit-learn==1.6.1`. If your training environment uses a different
scikit-learn version, either update `requirements.txt` to match, or re-train
with 1.6.1 installed, to avoid `joblib.load` compatibility issues.

## 2. Run locally without Docker (fastest way to test)

```bash
pip install -r requirements.txt
python app.py
```

Then in another terminal:

```bash
curl http://localhost:5000/health

curl http://localhost:5000/schema

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 18, "day": 15, "month": 12,
    "source": "Back Bay", "destination": "Fenway",
    "cab_type": "Uber", "name": "UberX",
    "distance": 2.5, "surge_multiplier": 1.0,
    "temperature": 35.0, "temperatureMax": 40.0, "temperatureMin": 30.0,
    "precipIntensity": 0.0, "precipProbability": 0.1, "humidity": 0.6,
    "windSpeed": 5.0, "windGust": 8.0, "visibility": 10.0,
    "dewPoint": 28.0, "pressure": 1015.0, "cloudCover": 0.5,
    "uvIndex": 0, "ozone": 300.0, "precipIntensityMax": 0.0
  }'
# -> {"predicted_price": <number>}
```

`GET /schema` always tells you the exact required fields and valid category
values, pulled straight from `feature_config.json` — use it instead of
hardcoding the field list, in case you retrain on updated data later.

Batch prediction: POST a JSON **list** of ride objects instead of a single
object, and you'll get back `{"predictions": [...]}` in the same order.

## 3. Containerize

```bash
docker build -t rideshare-api .
docker run -p 5000:5000 rideshare-api
```

Same endpoints as above, now served by `gunicorn` (2 workers) instead of
Flask's dev server. Adjust `--workers` in the `Dockerfile`'s `CMD` based on
your host's CPU count and expected traffic.

Sanity check the image size and that it starts cleanly:

```bash
docker images rideshare-api
docker logs $(docker ps -lq)
```

## 4. Deploy

A few common options, roughly in order of "least setup" to "most control":

- **Render / Railway / Fly.io** — point them at this repo (or push the built
  image), they build from the `Dockerfile` and give you a public URL with
  minimal config. Good default if you just want it live quickly.
- **Google Cloud Run** — `gcloud run deploy` from this directory; serverless,
  scales to zero, pay-per-request. Good fit for a model API with uneven
  traffic.
- **AWS ECS/Fargate** — push the image to ECR, run as a Fargate service behind
  an ALB. More setup, more control over networking/scaling.
- **A plain VM/EC2/Droplet** — `docker run` directly, put nginx in front for
  TLS. Most manual, cheapest for steady low traffic.

Tell me which one you want to use and I can walk through the exact commands
— they differ enough (auth setup, CLI tools, config files) that it's worth
tailoring rather than guessing.

## Known limitations / things to double check on your end

- **Docker itself wasn't tested here** — no `docker` binary in this sandbox.
  The `Dockerfile` follows a standard, well-tested pattern (slim Python base,
  pinned deps, gunicorn), but run `docker build` yourself to confirm it works
  end to end before deploying.
- **`OneHotEncoder(handle_unknown='ignore')`** means a ride from a
  neighborhood outside the original 12 Boston locations, or a `name` outside
  the original 12 vehicle tiers, won't error — it'll just get an all-zero
  encoding for that field, which quietly degrades that prediction rather than
  failing loudly. Check `GET /schema` if you want to validate inputs against
  known categories before sending them.
- **No auth on `/predict`** — add an API key check or put it behind a gateway
  before exposing it publicly, if that matters for your use case.
