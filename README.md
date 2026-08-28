# Cats vs Dogs, end to end MLOps pipeline

MLOps (S1-25_AIMLCZG523) Assignment 2.

Binary image classifier for a pet adoption platform, wired into a full pipeline:
data versioning with DVC, experiment tracking with MLflow, a FastAPI inference
service, a Docker image published to GitHub Container Registry by CI, automatic
deployment to Kubernetes by CD, and request metrics on the running service.

**Name:** Mohammad Saqib Koti  
**Roll number:** 2024AD05127  

## What is in here

```
.
├── src/                  training and preprocessing code
│   ├── config.py         paths, hyperparameters, class order
│   ├── data_prep.py      raw images to a clean 224x224 80/10/10 split
│   ├── dataset.py        torch loaders and the augmentation pipeline
│   ├── model.py          the baseline CNN, save and load helpers
│   ├── train.py          training loop with MLflow logging
│   └── predict.py        inference helpers used by the API and the tests
├── api/main.py           FastAPI service: /health /predict /metrics /stats
├── tests/                40 unit tests, run by CI on every push
├── scripts/
│   ├── smoke_test.py     post deploy check, fails the pipeline if broken
│   └── monitor_batch.py  scores the deployed model on held out images
├── k8s/                  Deployment, Service and the kind cluster config
├── monitoring/           Prometheus scrape config and a Grafana dashboard
├── .github/workflows/    ci.yml and cd.yml
├── dvc.yaml              the prepare and train stages
├── Dockerfile            two stage build, CPU torch, non root user
└── docker-compose.yml    api plus Prometheus plus Grafana for local runs
```

## Setup

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt
```

For GPU training install the CUDA build instead:

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

Every version in `requirements.txt` and `requirements-api.txt` is pinned, so a
rebuild months from now resolves to the same packages.

## M1: data, model and experiment tracking

### Data versioning

The dataset is the Kaggle cats and dogs set. Drop the extracted archive into
`data/raw/`. Both common layouts work, either `train/cats` and `train/dogs`
folders or flat `cat.0.jpg` style filenames, because the label is read out of
the path.

```bash
python -m dvc add data/raw
python -m src.data_prep
python -m dvc add data/processed
python -m dvc push
git add data/raw.dvc data/processed.dvc .gitignore
```

`data/raw` and `data/processed` never enter git, only the small `.dvc` pointer
files do. The remote is a local directory (`../dvcstore-catdog`) configured in
`.dvc/config`. Point it at S3, GDrive or any other supported backend by editing
that one line.

The two steps are also wired as a DVC pipeline, so `dvc repro` reruns only what
changed:

```bash
python -m dvc repro
```

### Preprocessing

`src/data_prep.py` converts everything to RGB (some files in the archive are
greyscale or have an alpha channel), resizes to 224x224, and writes a
stratified 80/10/10 split. Stratified means each split keeps the same cat to
dog ratio, and the seed is fixed so the split is reproducible. Truncated files
in the archive are counted and skipped rather than crashing the run.

Augmentation is applied to the training split only: random resized crop,
horizontal flip, small rotation and colour jitter. No vertical flip, an upside
down pet is not something the model will ever be asked about.

### Model

`SimpleCNN` is four conv blocks (each two 3x3 convs with batch norm, then max
pool), global average pooling, and a two layer head with dropout. Global
pooling instead of a flatten keeps the parameter count low enough to train on a
4 GB laptop GPU.

```bash
python -m src.train --epochs 12 --batch-size 32
```

The checkpoint goes to `models/cats_dogs_cnn.pt`. It stores the weights along
with the class order and the image size, so the API cannot load a model and
then mislabel its outputs.

### Experiment tracking

Every run logs to MLflow: all hyperparameters, per epoch train and validation
loss and accuracy, the learning rate, and the final test accuracy, precision,
recall, F1 and ROC AUC. The confusion matrix, the loss and accuracy curves,
`metrics.json` and the checkpoint itself are logged as artifacts.

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Then open http://localhost:5000.

## M2: packaging and containerization

### The service

```bash
uvicorn api.main:app --reload --port 8000
```

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | 200 when the model is loaded, 503 when it is not |
| POST | `/predict` | multipart image upload, returns label, confidence and both probabilities |
| GET | `/metrics` | Prometheus exposition format |
| GET | `/stats` | the same counters as readable JSON |
| GET | `/docs` | interactive Swagger UI |

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict -F "file=@data/processed/test/dog/dog_00001.jpg"
```

```json
{
  "label": "dog",
  "confidence": 0.9412,
  "probabilities": { "cat": 0.0588, "dog": 0.9412 },
  "filename": "dog_00001.jpg"
}
```

If the checkpoint is missing the container still starts, and `/health` returns
503 with the reason. That is deliberate. It lets Kubernetes mark the pod
unready instead of restart looping, and the failure shows up in one place.

### The image

Two stage build. The first stage compiles wheels, the second copies only the
installed packages, so no build toolchain ships in the final image. torch is
pulled from the CPU wheel index rather than PyPI, because the default wheel is
the CUDA build and serving never touches the GPU. The container runs as uid
10001, not root, and has a Docker healthcheck on `/health`.

The image lands at about 1.5 GB. Most of that is torch itself, roughly 1 GB
unpacked for libtorch and the maths kernels, and there is no way around it
while the model is served through torch. Exporting to ONNX and serving with
onnxruntime would cut it to around 300 MB, which is the obvious next step if
image size ever became a real constraint.

```bash
docker build -t catdog-api:local .
docker run --rm -p 8000:8000 catdog-api:local
```

## M3: CI

`.github/workflows/ci.yml` runs on every push and pull request.

1. `test` installs the pinned dependencies and runs the full pytest suite. The
   JUnit report is uploaded as a build artifact.
2. `build-and-push` only starts if the tests pass. It builds the image with
   Buildx, tags it `latest`, `sha-<short>` and the branch name, and pushes to
   `ghcr.io/saqib8/2024ad05127-mlops-assignment2`. Layer caching is on, so a
   code only change rebuilds in well under a minute.
3. The freshly built image is started and hit with the smoke test before the
   job is allowed to go green.

Pull requests build the image but do not publish it.

### The tests

40 tests across three files.

- `tests/test_data_prep.py` covers the preprocessing side: label extraction from
  both dataset layouts, the RGB conversion, and the stratified split including
  that it preserves class balance, that nothing is dropped or duplicated, and
  that the seed makes it reproducible.
- `tests/test_inference.py` covers the model utilities and the inference path:
  output shape, softmax, the save and load round trip, decoding uploaded bytes,
  and that a prediction is well formed and deterministic.
- `tests/test_api.py` drives the real app through FastAPI's test client and
  checks health, prediction, the error cases, and that the counters move.

The model tests build an untrained network on the fly, so CI needs neither the
dataset nor a trained checkpoint.

```bash
python -m pytest
```

## M4: CD

`.github/workflows/cd.yml` starts automatically when CI finishes successfully
on `main`.

1. Creates a kind cluster on the runner using `k8s/kind-cluster.yaml`.
2. Pulls the exact image CI just published, matched by commit sha, and
   sideloads it into the cluster.
3. Applies `k8s/deployment.yaml` and `k8s/service.yaml`, then pins the
   deployment to that image and waits for the rollout.
4. Runs `scripts/smoke_test.py` against the deployed Service. The script waits
   for `/health`, sends a real prediction, validates the response shape and the
   probabilities, and checks that `/metrics` is exposing counters. Any failure
   exits non zero and fails the pipeline.
5. On failure it runs `kubectl rollout undo` and dumps the pod logs.

The Deployment runs two replicas with `maxUnavailable: 0`, so a bad image never
takes the service down. Readiness and liveness both point at `/health`, which
means traffic only reaches a pod once the model is actually loaded.

To deploy locally instead:

```bash
kind create cluster --name catdog --config k8s/kind-cluster.yaml
kind load docker-image catdog-api:local --name catdog
kubectl create namespace catdog
kubectl -n catdog apply -f k8s/
kubectl -n catdog rollout status deployment/catdog-api
python scripts/smoke_test.py --base-url http://localhost:30080
```

## M5: monitoring and logging

### Logs

Middleware logs every request with a short request id, method, path, status and
latency in milliseconds. Predictions log the filename, the byte count, the
predicted label and the confidence. The uploaded image itself is never written
to the log, only its size.

### Metrics

`prometheus_client` exposes:

- `api_requests_total` by endpoint, method and status
- `api_request_latency_seconds` as a histogram, so p50, p95 and p99 are
  available from the bucket data
- `predictions_total` by predicted label, which is how prediction drift shows up
- `model_loaded` as a gauge

`/stats` returns the same numbers as JSON for a quick look without Prometheus.

```bash
docker compose up -d --build
```

That brings up the API on 8000, Prometheus on 9090 and Grafana on 3000
(admin/admin). The dashboard in `monitoring/grafana/provisioning/dashboards/`
is provisioned automatically, no manual import.

### Post deployment performance

`scripts/monitor_batch.py` sends a balanced batch of held out test images to
the running service, compares the answers against the true labels, and writes
`reports/post_deployment_report.json` with live accuracy, the latency
distribution and a confusion breakdown.

```bash
python scripts/monitor_batch.py --base-url http://localhost:8000 --limit 100
```

Passing `--min-accuracy 0.80` makes it exit non zero when live accuracy drops
below the threshold, which is what you would wire into an alert.

## Notes on a couple of choices

**Why the checkpoint is in git and the dataset is not.** The checkpoint is a few
MB and the Docker build needs it, so committing it keeps CI self contained. The
dataset is around 25000 images, which is exactly what DVC is for.

**Why kind inside CI rather than a local cluster.** GitHub's hosted runners
cannot reach a laptop, so deploying to a local minikube would need a self hosted
runner. Building the cluster inside the job keeps the whole thing reproducible
and leaves the full deployment visible in the run logs.
