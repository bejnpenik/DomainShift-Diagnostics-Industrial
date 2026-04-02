"""
Collection dataset downloader.

Reads the 'download' section from a collection YAML and fetches all dataset files.

Supported formats:
  - 'individual': one HTTP GET per file ID, saved flat into dirname/
  - 'zip'/'rar':  one archive per unique subdirectory, extracted directly into dirname/

Files/directories that already exist are skipped.
Downloads stream in 1 MB chunks and retry up to 3 times with exponential backoff.

RAR extraction requires the 'rarfile' package and the system 'unrar' binary:
  pip install rarfile
  apt install unrar   # or brew install rar
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def download_collection(yaml_path: str | Path) -> None:
    """Download all dataset files listed in a collection YAML.

    Args:
        yaml_path: Path to the collection YAML file.

    Raises:
        KeyError: If the YAML has no 'download' section.
        ValueError: If the download format is unknown.
    """
    import yaml

    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if 'download' not in cfg:
        raise KeyError(
            f"Collection YAML '{yaml_path}' has no 'download' section. "
            "Add base_url, format, and filename_template to enable downloading."
        )

    dl = cfg['download']
    base_url: str = dl['base_url'].rstrip('/')
    fmt: str = dl['format']
    filename_template: str = dl['filename_template']
    dirname = Path(cfg['dirname'])
    filetype: str = cfg.get('filetype', 'mat')

    dirname.mkdir(parents=True, exist_ok=True)

    skip = {str(s) for s in dl.get('skip', [])}

    if fmt == 'individual':
        _download_individual(cfg['files'], base_url, filename_template, dirname, filetype, skip)
    elif fmt in ('zip', 'rar'):
        _download_archive(cfg['files'], base_url, filename_template, dirname)
    else:
        raise ValueError(f"Unknown download format '{fmt}'. Use 'individual', 'zip', or 'rar'.")


def _download_individual(
    files: dict,
    base_url: str,
    filename_template: str,
    dirname: Path,
    filetype: str,
    skip: set[str],
) -> None:
    """Download one file per entry in the files dict, skipping listed IDs."""
    total = len(files)
    for i, file_id in enumerate(files, 1):
        if str(file_id) in skip:
            print(f"[{i}/{total}] {file_id} in skip list, skipping.")
            continue
        filename = filename_template.format(file_id=file_id, filetype=filetype)
        dest = dirname / filename
        if dest.exists():
            print(f"[{i}/{total}] {filename} already exists, skipping.")
            continue
        url = f"{base_url}/{filename}"
        print(f"[{i}/{total}] Downloading {url} ...")
        _fetch(url, dest)
        print(f"[{i}/{total}] Saved to {dest}")


def _download_archive(
    files: dict[str, Any],
    base_url: str,
    filename_template: str,
    dirname: Path,
) -> None:
    """Download one archive per unique subdirectory, extract directly into dirname/.

    Supports .zip and .rar (RAR requires 'rarfile' package + system 'unrar' binary).
    Multiple file entries can share a subdirectory; each is downloaded only once.
    """
    # Collect unique subdirectories in order of first appearance
    seen: dict[str, None] = {}
    for entry in files.values():
        subdir_name = entry.get('subdirectory') if isinstance(entry, dict) else str(entry)
        if subdir_name not in seen:
            seen[subdir_name] = None

    total = len(seen)
    for i, subdir_name in enumerate(seen, 1):
        subdir = dirname / subdir_name
        if subdir.exists() and any(subdir.iterdir()):
            print(f"[{i}/{total}] {subdir_name}/ already populated, skipping.")
            continue
        filename = filename_template.format(subdirectory=subdir_name)
        url = f"{base_url}/{filename}"
        archive = dirname / filename
        print(f"[{i}/{total}] Downloading {url} ...")
        _fetch(url, archive)
        print(f"[{i}/{total}] Extracting to {dirname} ...")
        _extract(archive, dirname)
        archive.unlink()
        print(f"[{i}/{total}] Done.")


def _extract(archive: Path, dest: Path) -> None:
    """Extract archive to dest. Supports .zip and .rar."""
    suffix = archive.suffix.lower()
    if suffix == '.zip':
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif suffix == '.rar':
        try:
            import rarfile
        except ImportError:
            raise ImportError(
                "RAR extraction requires the 'rarfile' package and the 'unrar' system binary.\n"
                "  pip install rarfile\n"
                "  apt install unrar   # or: brew install rar"
            )
        with rarfile.RarFile(archive) as rf:
            rf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format '{suffix}'. Use .zip or .rar.")


def _fetch(url: str, dest: Path, retries: int = 3, chunk_size: int = 1024 * 1024) -> None:
    """Download url to dest, streaming in 1 MB chunks with retry on failure."""
    import time

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url) as response:
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Attempt {attempt} failed ({exc}), retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise
