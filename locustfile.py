"""Flood-simulation load test against POST /predict (and light /health polling).

Run against the local stack, e.g.:
    locust -f locustfile.py --host http://localhost:8000

Or headless, for the container-count comparison described in the README:
    locust -f locustfile.py --host http://nginx:80 --headless -u 100 -r 10 -t 60s \
        --csv results/2containers
"""

import glob
import os
import random

from locust import HttpUser, task, between

SAMPLE_IMAGES = glob.glob("data/test/Up/*.png") + glob.glob("data/test/Down/*.png")


class CandlestickUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        if not SAMPLE_IMAGES:
            raise RuntimeError(
                "No sample images found under data/test/. Run locust from the "
                "project root (crypto_ml_pipeline/) so relative paths resolve."
            )

    @task(5)
    def predict(self):
        image_path = random.choice(SAMPLE_IMAGES)
        with open(image_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(image_path), f, "image/png")},
                name="/predict",
            )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
