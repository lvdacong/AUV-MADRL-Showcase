from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def runtime_information(device: str | torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "device": str(torch.device(device)),
    }


def repository_state(
    repository_root: Path,
    *,
    excluded_paths: Sequence[Path] = (),
) -> dict[str, object]:
    root = repository_root.resolve()
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        encoding="utf-8",
    ).strip()
    status = subprocess.check_output(
        (
            "git",
            "-c",
            "core.quotepath=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        cwd=root,
        text=True,
        encoding="utf-8",
    )
    excluded = []
    for path in excluded_paths:
        try:
            excluded.append(path.resolve().relative_to(root).as_posix())
        except ValueError:
            continue
    dirty_paths = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        value = line[3:].strip('"').replace("\\", "/")
        candidates = [item.strip('"') for item in value.split(" -> ")]
        if any(
            candidate == prefix or candidate.startswith(prefix + "/")
            for candidate in candidates
            for prefix in excluded
        ):
            continue
        dirty_paths.extend(candidates)
    unique_paths = sorted(set(dirty_paths))
    return {
        "repository_commit": commit,
        "repository_dirty": bool(unique_paths),
        "repository_dirty_paths": unique_paths,
    }


def create_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one complete JSON file without replacing any existing target."""
    data=(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    identity=os.fstat(descriptor)
    def same_file(candidate):
        stat=candidate.lstat()
        return (stat.st_dev,stat.st_ino)==(identity.st_dev,identity.st_ino)
    def diagnostic(error):
        try:print(f'JSON publication {path}: {type(error).__name__}: {error}',file=sys.stderr)
        except Exception:pass
    try:
        with os.fdopen(descriptor,'wb') as stream:
            if stream.write(data)!=len(data):raise OSError('Incomplete JSON temporary write')
            stream.flush()
            os.fsync(stream.fileno())
        if not same_file(temporary):raise RuntimeError('JSON temporary file identity changed')
        try:
            os.link(temporary,path)
        except OSError as error:
            # A filesystem call may succeed before its response is lost.
            try:published=same_file(path) and path.read_bytes()==data
            except OSError:published=False
            if not published:raise
            diagnostic(error)
    finally:
        try:
            if same_file(temporary):temporary.unlink()
            else:raise RuntimeError('JSON temporary file replaced; retained foreign file')
        except OSError as error:
            diagnostic(error)
        except RuntimeError as error:
            diagnostic(error)


def write_json_atomic(path: Path, payload: Mapping[str, object], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=None if compact else 2,
                   separators=(',', ':') if compact else None, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Windows readers can briefly deny replacement, as with checkpoint files.
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def update_json_atomic(path: Path, changes: Mapping[str, object], *, compact: bool = False) -> None:
    payload: dict[str, object] = {}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            payload = value
    payload.update(changes)
    write_json_atomic(path, payload, compact=compact)
