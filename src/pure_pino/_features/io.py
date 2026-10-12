from __future__ import annotations
import hashlib, json, os, tempfile, time
from pathlib import Path
from typing import Any

class ContractError(ValueError):
    """Missing or inconsistent scientific/data contract; never silently impute."""

class BlockedError(RuntimeError):
    """An integration or evidence gate has not passed."""

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def hash_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path: Path) -> Any:
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)

def write_json(path: Path, value: Any) -> None:
    """Atomic replacement in the NEW output tree only; caller controls path."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def append_jsonl(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

def confined(root: Path, relative: str) -> Path:
    root=Path(root).resolve(); p=(root/relative).resolve()
    try: p.relative_to(root)
    except ValueError as exc: raise ContractError(f'Path escapes dataset: {relative}') from exc
    return p

def require_new_dir(path: Path) -> Path:
    path=Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise BlockedError(f'Refusing to overwrite nonempty run directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def assert_output_separate(output: Path, protected: list[Path]) -> None:
    out=Path(output).resolve()
    for p in protected:
        p=Path(p).resolve()
        if out==p or p in out.parents or out in p.parents:
            raise BlockedError(f'Output and protected path overlap: {out} / {p}')

def code_digest() -> str:
    root=Path(__file__).parent
    return hash_object({str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))})

class RunLock:
    """Exclusive run lock. Existing lock is never auto-deleted or stolen."""
    def __init__(self, root: Path): self.path=Path(root)/'RUN.lock'
    def __enter__(self):
        try: fd=os.open(self.path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as e: raise BlockedError(f'Existing lock: {self.path}') from e
        with os.fdopen(fd,'w') as f: json.dump({'pid':os.getpid(),'created_utc_s':time.time()},f)
        return self
    def __exit__(self,*args):
        if self.path.exists(): self.path.unlink()
