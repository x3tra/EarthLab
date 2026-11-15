from __future__ import annotations

import os

import pytest

from nasa_agri_intelligence.collection.task_manager import TaskManager


def test_live_appeears_task_submission_requires_explicit_opt_in() -> None:
    if os.getenv("APPEEARS_CONFIRM_REAL_TASK_SUBMISSION") != "YES":
        pytest.skip(
            "Set APPEEARS_CONFIRM_REAL_TASK_SUBMISSION=YES to submit one live AppEEARS task."
        )

    manifest = TaskManager().submit(
        "MOD13Q1.061",
        task_name="nasa-agri-live-integration-test",
    )

    assert manifest["task_id"]
    assert manifest["submission_status"] == "submitted"
