"""Remote sync — push unsynced triage cases to CouchDB when connectivity returns."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.sync.store import LocalStore

logger = logging.getLogger(__name__)


class RemoteSync:
    """Push offline triage cases to CouchDB when connectivity is available."""

    def __init__(self, store: LocalStore, remote_url: str, db_name: str = "cairn_triage"):
        self.store = store
        self.remote_url = remote_url.rstrip("/")
        self.db_name = db_name
        self._client = httpx.Client(timeout=30.0)

    def is_online(self) -> bool:
        """Check if the remote CouchDB is reachable."""
        try:
            resp = self._client.get(f"{self.remote_url}/")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def ensure_db_exists(self) -> bool:
        """Create the remote database if it doesn't exist."""
        try:
            resp = self._client.put(f"{self.remote_url}/{self.db_name}")
            return resp.status_code in (201, 412)  # 412 = already exists
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def sync(self) -> dict[str, Any]:
        """Push all unsynced cases to remote. Returns sync summary."""
        if not self.is_online():
            return {"status": "offline", "synced": 0, "failed": 0}

        self.ensure_db_exists()
        unsynced = self.store.get_unsynced_cases()

        if not unsynced:
            return {"status": "online", "synced": 0, "failed": 0, "message": "Nothing to sync"}

        synced = 0
        failed = 0

        for case in unsynced:
            try:
                doc = {
                    "_id": case["case_id"],
                    "type": "triage_encounter",
                    **{k: v for k, v in case.items() if k != "case_id"},
                }
                resp = self._client.put(
                    f"{self.remote_url}/{self.db_name}/{case['case_id']}",
                    json=doc,
                )
                if resp.status_code in (201, 409):  # 409 = already exists
                    self.store.mark_synced(case["case_id"])
                    synced += 1
                else:
                    logger.warning(f"Sync failed for {case['case_id']}: {resp.status_code}")
                    failed += 1
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Connection lost during sync: {e}")
                failed += 1
                break  # Stop sync if connection drops

        return {
            "status": "online",
            "synced": synced,
            "failed": failed,
            "remaining": len(unsynced) - synced - failed,
        }

    def close(self):
        self._client.close()
