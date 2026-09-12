from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_ticket_automation_lock_root(tmp_path, monkeypatch):
    monkeypatch.setenv("_TICKET_AUTOMATION_LOCK_ROOT", str(tmp_path / "ta-locks"))
