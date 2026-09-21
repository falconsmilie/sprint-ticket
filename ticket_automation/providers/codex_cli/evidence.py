"""Provider-native diagnostic artifacts for Codex CLI executions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...application.agent_execution import ArtifactReference

PROMPT_ARTIFACT = "prompt.md"
EVENTS_ARTIFACT = "events.jsonl"
STDERR_ARTIFACT = "stderr.log"
EXECUTION_ARTIFACT = "execution.json"
RESULT_ARTIFACT = "codex-result.json"


@dataclass(frozen=True)
class CodexArtifactPaths:
    directory: Path
    prompt: Path
    events: Path
    stderr: Path
    execution: Path
    result: Path

    @classmethod
    def create(cls, directory: Path) -> CodexArtifactPaths:
        directory = Path(directory)
        return cls(
            directory=directory,
            prompt=directory / PROMPT_ARTIFACT,
            events=directory / EVENTS_ARTIFACT,
            stderr=directory / STDERR_ARTIFACT,
            execution=directory / EXECUTION_ARTIFACT,
            result=directory / RESULT_ARTIFACT,
        )

    def references(self) -> tuple[ArtifactReference, ...]:
        return (
            ArtifactReference("prompt", self.prompt, "text/markdown"),
            ArtifactReference("events", self.events, "application/x-ndjson"),
            ArtifactReference("standard-error", self.stderr, "text/plain"),
            ArtifactReference("execution-details", self.execution, "application/json"),
            ArtifactReference("structured-result", self.result, "application/json"),
        )


def prepare(paths: CodexArtifactPaths, prompt: str) -> None:
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.result.unlink(missing_ok=True)
    paths.prompt.write_text(prompt, encoding="utf-8", newline="\n")
    paths.events.write_text("", encoding="utf-8", newline="\n")
    paths.stderr.write_text("", encoding="utf-8", newline="\n")


def write_process_output(
    paths: CodexArtifactPaths,
    *,
    stdout: str,
    stderr: str,
) -> None:
    paths.events.write_text(stdout, encoding="utf-8", newline="\n")
    paths.stderr.write_text(stderr, encoding="utf-8", newline="\n")


def write_execution(paths: CodexArtifactPaths, record: dict[str, Any]) -> None:
    _atomic_write_json(paths.execution, record)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            descriptor = -1
            output.write(payload)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        if descriptor != -1:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EVENTS_ARTIFACT",
    "EXECUTION_ARTIFACT",
    "PROMPT_ARTIFACT",
    "RESULT_ARTIFACT",
    "STDERR_ARTIFACT",
    "CodexArtifactPaths",
    "prepare",
    "write_execution",
    "write_process_output",
]
