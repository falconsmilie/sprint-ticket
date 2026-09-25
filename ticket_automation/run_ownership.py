"""Physical ownership token for one run directory."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class RunOwnershipError(RuntimeError):
    """Raised when a run is no longer the owned child selected by the caller."""


def validate_run_id(value: object) -> None:
    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Run ID must be one non-empty path component containing only letters, "
            "numbers, dots, underscores, and hyphens."
        )


@dataclass(frozen=True)
class RunOwnership:
    """Stable runs-root and run identity used to revalidate persistence paths."""

    runs_root: Path
    run_id: str
    _runs_root_identity: tuple[int, int] = field(init=False, repr=False)
    _run_identity: tuple[int, int] = field(init=False, repr=False)

    @classmethod
    def acquire(cls, runs_root: Path | str, run_id: str) -> RunOwnership:
        validate_run_id(run_id)
        try:
            root = Path(runs_root).resolve(strict=True)
        except FileNotFoundError as error:
            raise RunOwnershipError(
                f"Runs directory does not exist: {Path(runs_root)}"
            ) from error
        except (OSError, RuntimeError) as error:
            raise RunOwnershipError(
                f"Could not resolve the runs directory safely: {error}"
            ) from error
        ownership = cls(root, run_id)
        ownership.validate()
        return ownership

    @classmethod
    def reserve(cls, runs_root: Path | str, run_id: str) -> RunOwnership:
        """Create and bind one direct run directory as a single reservation step."""

        validate_run_id(run_id)
        try:
            root = Path(runs_root).resolve(strict=True)
        except FileNotFoundError as error:
            raise RunOwnershipError(
                f"Runs directory does not exist: {Path(runs_root)}"
            ) from error
        except (OSError, RuntimeError) as error:
            raise RunOwnershipError(
                f"Could not resolve the runs directory safely: {error}"
            ) from error
        candidate = root / run_id
        candidate.mkdir()
        try:
            ownership = cls(root, run_id)
            ownership.validate()
            return ownership
        except BaseException:
            try:
                candidate.rmdir()
            except OSError:
                pass
            raise

    def __post_init__(self) -> None:
        if not isinstance(self.runs_root, Path) or not self.runs_root.is_absolute():
            raise TypeError("runs_root must be an absolute Path.")
        validate_run_id(self.run_id)
        try:
            root_details = self.runs_root.stat()
            run_details = self.run_dir.stat()
        except OSError as error:
            raise RunOwnershipError(
                f"Could not bind the selected run directory safely: {error}"
            ) from error
        object.__setattr__(
            self,
            "_runs_root_identity",
            (root_details.st_dev, root_details.st_ino),
        )
        object.__setattr__(
            self,
            "_run_identity",
            (run_details.st_dev, run_details.st_ino),
        )

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    def validate(self) -> Path:
        """Return the owned run path only while it remains a direct, unlinked child."""

        candidate = self.run_dir
        try:
            resolved_root = self.runs_root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise RunOwnershipError(
                f"Run directory ownership was lost; the selected run is missing: "
                f"{candidate}"
            ) from error
        except (OSError, RuntimeError) as error:
            raise RunOwnershipError(
                f"Run directory ownership could not be revalidated safely: {error}"
            ) from error
        if resolved_root != self.runs_root:
            raise RunOwnershipError(
                "Run directory ownership was lost; the configured runs directory "
                f"was retargeted: {self.runs_root}"
            )
        if resolved != candidate or resolved.parent != self.runs_root:
            raise RunOwnershipError(
                "Run directory ownership was lost; the selected path must remain "
                f"a direct, non-linked child of the configured runs directory: "
                f"{candidate}"
            )
        if not resolved.is_dir():
            raise RunOwnershipError(
                f"Run directory ownership was lost; the selected path is not a "
                f"directory: {candidate}"
            )
        try:
            root_details = resolved_root.stat()
            run_details = resolved.stat()
        except OSError as error:
            raise RunOwnershipError(
                f"Run directory ownership could not be revalidated safely: {error}"
            ) from error
        if (root_details.st_dev, root_details.st_ino) != self._runs_root_identity:
            raise RunOwnershipError(
                "Run directory ownership was lost; the configured runs directory "
                "was replaced."
            )
        if (run_details.st_dev, run_details.st_ino) != self._run_identity:
            raise RunOwnershipError(
                "Run directory ownership was lost; the selected run directory "
                "was replaced."
            )
        return resolved

    def validate_run_path(self, path: Path | str) -> Path:
        """Validate that ``path`` is exactly this token's still-owned run."""

        owned = self.validate()
        requested = Path(os.path.abspath(Path(path)))
        if requested != owned:
            raise RunOwnershipError(
                "Run persistence path does not match its configured ownership "
                f"token: {requested}"
            )
        return owned

    def validate_descendant(self, path: Path | str) -> Path:
        """Validate a current path below the still-owned run without following escape links."""

        owned = self.validate()
        requested = Path(os.path.abspath(Path(path)))
        try:
            relative = requested.relative_to(owned)
        except ValueError as error:
            raise RunOwnershipError(
                f"Run-owned path escapes the selected run directory: {requested}"
            ) from error
        if not relative.parts:
            return owned
        try:
            resolved_parent = requested.parent.resolve(strict=True)
            resolved_parent.relative_to(owned)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise RunOwnershipError(
                f"Run-owned path could not be confined safely: {requested}: {error}"
            ) from error
        try:
            requested.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RunOwnershipError(
                f"Run-owned path could not be inspected safely: {requested}"
            ) from error
        else:
            try:
                requested.resolve(strict=True).relative_to(owned)
            except (OSError, RuntimeError, ValueError) as error:
                raise RunOwnershipError(
                    f"Run-owned path was retargeted outside its owner: {requested}"
                ) from error
        return requested


def validate_unlinked_run_directory(path: Path | str) -> Path:
    """Reject a run path whose current physical target differs from its path."""

    try:
        requested = Path(os.path.abspath(Path(path)))
        resolved = requested.resolve(strict=True)
    except FileNotFoundError as error:
        raise RunOwnershipError(
            f"Run directory does not exist: {Path(path)}"
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise RunOwnershipError(
            f"Could not revalidate the run directory safely: {error}"
        ) from error
    if resolved != requested:
        raise RunOwnershipError(
            "Run directory ownership was lost; the run path must not be linked: "
            f"{requested}"
        )
    if not resolved.is_dir():
        raise RunOwnershipError(f"Run directory does not exist: {requested}")
    return resolved


__all__ = [
    "RunOwnership",
    "RunOwnershipError",
    "validate_run_id",
    "validate_unlinked_run_directory",
]
