# Shortcuts for the common steps. On Windows run these through Git Bash, or
# just copy the command out of the recipe.

IMAGE ?= catdog-api:local
BASE_URL ?= http://localhost:8000

.PHONY: install prepare train test api docker-build docker-run compose-up compose-down mlflow k8s-deploy k8s-delete smoke monitor clean

install:
	pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
	pip install -r requirements.txt

prepare:
	python -m src.data_prep

train:
	python -m src.train

test:
	python -m pytest

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --backend-store-uri file:./mlruns --port 5000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --name catdog-api $(IMAGE)

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down -v

k8s-deploy:
	kubectl create namespace catdog --dry-run=client -o yaml | kubectl apply -f -
	kubectl -n catdog apply -f k8s/deployment.yaml
	kubectl -n catdog apply -f k8s/service.yaml
	kubectl -n catdog rollout status deployment/catdog-api --timeout=300s

k8s-delete:
	kubectl delete namespace catdog --ignore-not-found

smoke:
	python scripts/smoke_test.py --base-url $(BASE_URL)

monitor:
	python scripts/monitor_batch.py --base-url $(BASE_URL) --limit 100

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ test-results.xml
