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
from ...persistence import JsonValue, atomic_write_json, atomic_write_text

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
        self.revalidate()
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

    def revalidate(self) -> None:
        """Reject cached provider paths after their owning run is retargeted."""

        self.layout.revalidate()
        expected = {
            "prompt": self.layout.path(PROMPT_ARTIFACT),
            "events": self.layout.path(EVENTS_ARTIFACT),
            "stderr": self.layout.path(STDERR_ARTIFACT),
            "execution": self.layout.path(EXECUTION_ARTIFACT),
            "result": self.layout.path(RESULT_ARTIFACT),
        }
        for name, path in expected.items():
            if getattr(self, name) != path:
                raise RuntimeError(f"Cached Codex artifact path changed: {name}.")


def prepare(paths: CodexArtifactPaths, prompt: str) -> None:
    paths.revalidate()
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
    paths.revalidate()
    paths.events.write_text(stdout, encoding="utf-8", newline="\n")
    paths.stderr.write_text(stderr, encoding="utf-8", newline="\n")


def write_execution(paths: CodexArtifactPaths, record: Mapping[str, JsonValue]) -> None:
    paths.revalidate()
    atomic_write_json(paths.execution, record)


def write_result(paths: CodexArtifactPaths, result: str) -> None:
    paths.revalidate()
    atomic_write_text(paths.result, result)


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
    "write_result",
]
