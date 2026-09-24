"""Provider-native diagnostic artifacts for Codex CLI executions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...application.agent_execution import (
    ArtifactReference,
    ArtifactRole,
    AttemptArtifactLayout,
)
from ...persistence import JsonValue, atomic_write_json

PROMPT_ARTIFACT = "prompt.md"
EVENTS_ARTIFACT = "events.jsonl"
STDERR_ARTIFACT = "stderr.log"
EXECUTION_ARTIFACT = "codex-execution.json"
RESULT_ARTIFACT = "codex-result.json"


@dataclass(frozen=True)
class CodexArtifactPaths:
    layout: AttemptArtifactLayout
    directory: Path
    prompt: Path
    events: Path
    stderr: Path
    execution: Path
    result: Path

    @classmethod
    def create(cls, layout: AttemptArtifactLayout) -> CodexArtifactPaths:
        if not isinstance(layout, AttemptArtifactLayout):
            raise TypeError("layout must be an AttemptArtifactLayout.")
        directory = layout.attempt_root
        return cls(
            layout=layout,
            directory=directory,
            prompt=directory / PROMPT_ARTIFACT,
            events=directory / EVENTS_ARTIFACT,
            stderr=directory / STDERR_ARTIFACT,
            execution=directory / EXECUTION_ARTIFACT,
            result=directory / RESULT_ARTIFACT,
        )

    def references(self) -> tuple[ArtifactReference, ...]:
        references = [
            self.layout.reference(ArtifactRole.PROMPT, self.prompt, "text/markdown"),
            self.layout.reference(
                ArtifactRole.PROVIDER_EVENTS,
                self.events,
                "application/x-ndjson",
            ),
            self.layout.reference(
                ArtifactRole.STANDARD_ERROR, self.stderr, "text/plain"
            ),
            self.layout.reference(
                ArtifactRole.PROVIDER_EXECUTION_DETAILS,
                self.execution,
                "application/json",
            ),
        ]
        if self.result.is_file():
            references.append(
                self.layout.reference(
                    ArtifactRole.TYPED_RESULT, self.result, "application/json"
                )
            )
        return tuple(references)


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


def write_execution(paths: CodexArtifactPaths, record: Mapping[str, JsonValue]) -> None:
    atomic_write_json(paths.execution, record)


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
