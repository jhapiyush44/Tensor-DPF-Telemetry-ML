import json
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("logs/predictions.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)


def log_prediction(value: float):
    record = {
        "timestamp": str(datetime.now()),
        "prediction": float(value)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
