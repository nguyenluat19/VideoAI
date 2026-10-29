"""Safe mapping between public /media URLs and files under STORAGE_DIR."""
from pathlib import Path

from .config import settings


def url_for(path: Path) -> str:
    return "/media/" + path.resolve().relative_to(settings.storage_dir).as_posix()


def resolve(url: str) -> Path | None:
    """Return the local file for a /media URL, or None if it escapes storage or is missing."""
    if not url or not url.startswith("/media/"):
        return None
    p = (settings.storage_dir / url[len("/media/"):]).resolve()
    # Only the two public roots are addressable; a bare storage-dir check would let
    # "/media/uploads/../app.db" reach the database.
    for root in (settings.uploads_dir.resolve(), settings.outputs_dir.resolve()):
        try:
            p.relative_to(root)
        except ValueError:
            continue
        return p if p.is_file() else None
    return None
