"""Append-only JSONL telemetry supplied exclusively by actual training events."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone


class Telemetry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def emit(self, event: str, fields: dict):
        if not isinstance(event, str) or not event or not isinstance(fields, dict):
            raise ValueError("telemetry requires event name and fields")
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, "fields": fields}
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
