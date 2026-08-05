from __future__ import annotations

import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version
from platformdirs import user_cache_path


CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day


def fetch_latest_version(project_name: str, timeout: float = 5.0) -> Version | None:
    """Fetch the latest version of a project from PyPI."""
    url = f"https://pypi.org/pypi/{project_name}/json"
    headers = {
        "Accept": "application/json",
        "User-Agent": f"{project_name} update-check",
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            latest_version_str = data["info"]["version"]
            return Version(latest_version_str)
    except (HTTPError, URLError, json.JSONDecodeError, KeyError, InvalidVersion):
        return None

def read_cache(cache_file: Path) -> tuple[Version, float] | None:
    """Read the cached version and timestamp from a cache file."""
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        latest_version = Version(data["latest_version"])
        checked_at = data["checked_at"]
        return latest_version, checked_at
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        InvalidVersion,
    ):
        # Missing, corrupted, or invalid cache file
        return None

def write_cache(cache_file: Path, latest_version: Version) -> None:
    """Write the latest version and current timestamp to a cache file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "latest_version": str(latest_version),
            "checked_at": time.time(),
        }
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        # Ignore errors when writing the cache
        pass

def check_for_update(
    project_name: str = "pysideband",
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    timeout: float = 5.0,
) -> tuple[str, str] | None:
    """Check for updates to the specified project.

    Returns the latest version if an update is available, otherwise None.
    """
    try:
        current_version = Version(version(project_name))
    except (PackageNotFoundError, InvalidVersion):
        return None
    
    cache_file = (
        user_cache_path(project_name, appauthor=False)
        / "update_check_cache.json"
    )
    cached = read_cache(cache_file)
    latest_version: Version | None = None
    
    if cached is not None:
        cached_version, checked_at = cached
        cache_age = time.time() - checked_at
        if 0 <= cache_age < cache_ttl_seconds:
            latest_version = cached_version
    
    if latest_version is None:
        try:
            latest_version = fetch_latest_version(project_name, timeout=timeout)
            if latest_version is not None:
                write_cache(cache_file, latest_version)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            InvalidVersion,
        ):
            # optionally use a cached version if PyPI is unreachable
            if cached is not None:
                latest_version = cached[0]
            else:
                return None
    
    if latest_version > current_version:
        return str(latest_version), str(current_version)
    
    return None
