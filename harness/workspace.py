"""Narrow fixture-workspace operations; confinement is not a code sandbox.

Generated Python is executed only by an injected sandbox executor. The default
workspace never invokes Python or a shell on the developer's host.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

CommandExecutor = Callable[[list[str], Path], dict[str, Any]]
RecallReader = Callable[[str, int, int, int], str]
MAX_FILE_BYTES = 512_000
MAX_OUTPUT_CHARS = 64_000
BLOCKED_PARTS = {".git", ".ssh", ".aws", ".codex", ".venv", "__pycache__"}


def encode_result(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class Workspace:
    def __init__(
        self,
        root: str | Path,
        *,
        executor: CommandExecutor | None = None,
        recall_reader: RecallReader | None = None,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace root must be a directory")
        self.executor = executor
        self.recall_reader = recall_reader

    def resolve(self, path: str, *, allow_directory: bool = False) -> Path:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Use a relative path within the task workspace")
        if any(part in BLOCKED_PARTS or part.startswith(".env") for part in relative.parts):
            raise ValueError("Reserved workspace path")
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlinks are not allowed in workspace operations")
        target = current.resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Path escapes the workspace")
        if target.exists() and target.is_dir() and not allow_directory:
            raise ValueError("Expected a file path")
        return target

    @staticmethod
    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _read_bytes(self, target: Path) -> bytes:
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("File exceeds the fixture size limit")
        return target.read_bytes()

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        if offset < 0 or not 1 <= limit <= 2000:
            raise ValueError("offset must be nonnegative; limit must be 1..2000 lines")
        if path.startswith("/context/"):
            chunk_id = path.removeprefix("/context/").removesuffix(".txt")
            if not chunk_id or "/" in chunk_id or ".." in chunk_id:
                raise ValueError("Invalid archived context path")
            return self.recall(chunk_id, offset, min(limit, 200))
        target = self.resolve(path)
        data = self._read_bytes(target)
        lines = data.decode("utf-8").splitlines(keepends=True)
        return encode_result({
            "path": target.relative_to(self.root).as_posix(),
            "content": "".join(lines[offset:offset + limit]),
            "sha256": self._digest(data),
            "offset": offset,
            "limit": limit,
            "total_lines": len(lines),
            "full_read": offset == 0 and limit >= len(lines),
        })

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        data = content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the fixture size limit")
        previous = self._digest(self._read_bytes(target)) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return encode_result({"path": path, "before_sha256": previous,
                              "sha256": self._digest(data), "written_bytes": len(data)})

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        target = self.resolve(path)
        content = self._read_bytes(target).decode("utf-8")
        if not old_string or content.count(old_string) != 1:
            raise ValueError("old_string must match exactly once; supply more context")
        return self.write_file(path, content.replace(old_string, new_string, 1))

    def files(self) -> list[str]:
        files: list[str] = []
        for candidate in sorted(self.root.rglob("*")):
            relative = candidate.relative_to(self.root).as_posix()
            try:
                resolved = self.resolve(relative, allow_directory=True)
                if resolved.is_file():
                    files.append(relative)
            except ValueError:
                continue
        return files

    def revision(self) -> str:
        """Content fingerprint of visible fixture files, not an invented Git revision."""
        digest = hashlib.sha256()
        for name in self.files():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(self._read_bytes(self.resolve(name)))
        return digest.hexdigest()

    def search(self, query: str, path: str = ".") -> str:
        """Literal search via rg, with Python fallback when rg is unavailable."""
        if not query or len(query) > 1000:
            raise ValueError("Search query must contain 1..1000 characters")
        if path != ".":
            target = self.resolve(path, allow_directory=True)
        else:
            target = self.root
        names = [name for name in self.files()
                 if self.resolve(name).is_relative_to(target) or self.resolve(name) == target]
        if not names:
            return encode_result({"query": query, "matches": "", "truncated": False})
        try:
            result = subprocess.run(
                ["rg", "--no-config", "--line-number", "--fixed-strings", "--", query, *names],
                cwd=self.root, env={"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"},
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode not in (0, 1):
                raise ValueError("Literal search failed")
            output = result.stdout
        except FileNotFoundError:
            hits = []
            for name in names:
                text = self._read_bytes(self.resolve(name)).decode("utf-8", errors="replace")
                hits.extend(f"{name}:{i}:{line}" for i, line in enumerate(text.splitlines(), 1)
                            if query in line)
            output = "\n".join(hits)
        return encode_result({"query": query, "matches": output[:MAX_OUTPUT_CHARS],
                              "truncated": len(output) > MAX_OUTPUT_CHARS})

    def run_tests(self, selection: str = "tests") -> str:
        """Run one pytest file/directory in the runner's isolated executor."""
        if self.executor is None:
            raise RuntimeError("Test execution requires an explicitly configured sandbox executor")
        # No flags, node expressions or arbitrary command text are accepted.
        target = self.resolve(selection, allow_directory=True)
        if not target.exists() or selection.startswith("-") or "::" in selection:
            raise ValueError("Select an existing test file or directory")
        revision = self.revision()
        result = self.executor(["python", "-m", "pytest", "-q", "--", selection], self.root)
        output = str(result.get("stdout", "")) + str(result.get("stderr", ""))
        return encode_result({"test_selection": selection, "exit_code": result["exit_code"],
                              "revision": revision, "output": output[:MAX_OUTPUT_CHARS],
                              "truncated": len(output) > MAX_OUTPUT_CHARS})

    def shell(self, operation: str) -> str:
        """A deliberately narrow compatibility tool; never invokes a system shell."""
        if operation == "list_files":
            return encode_result({"files": self.files()})
        if operation == "workspace_status":
            return encode_result({"revision": self.revision(), "files": len(self.files())})
        raise ValueError("Allowed shell operations: list_files, workspace_status")

    def recall(self, chunk_id: str, offset: int = 0, limit: int = 80, char_offset: int = 0) -> str:
        if not chunk_id or "/" in chunk_id or ".." in chunk_id:
            raise ValueError("Invalid chunk ID")
        if offset < 0 or char_offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Invalid recall line range")
        if self.recall_reader is None:
            return "Nothing hidden in this arm. Original outputs remain in conversation history."
        return self.recall_reader(chunk_id, offset, limit, char_offset)
