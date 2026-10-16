"""Fetch a large public file without trusting the connection to stay up.

The Severson batches are two to three gigabytes each and the publisher's
content delivery network serves a single connection at well under a megabyte a
second from some places. So this downloads in byte ranges, several at once,
each written straight into its place in one preallocated ``.partial`` file, and
renames that into place only once every range has completed and the size
matches what the server reported. A finished range leaves a marker, so an
interrupted run resumes with the ranges it does not have. Peak disk use is the
size of the file, which matters on a CI runner.

Nothing here executes or unpickles what it downloads. The files are MATLAB
v7.3 containers, which are HDF5, and they are only ever opened by h5py.
"""

from __future__ import annotations

import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_USER_AGENT = (
    "cobalt-battery-prognostics/1 (+https://github.com/CGainsWorks/cobalt-battery-prognostics)"
)
_CHUNK = 1 << 20


def remote_size(url: str, timeout_s: float = 60.0) -> int:
    """Size in bytes, from a one-byte range request.

    A HEAD request is the obvious choice and the publisher's storage answers it
    with a redirect that drops the length, so ask for one byte and read the
    total from ``Content-Range`` instead.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise OSError(f"{url} did not report a size (Content-Range {content_range!r})")
        return int(content_range.rsplit("/", 1)[1])


def _fetch_range(url: str, start: int, end: int, partial: Path, timeout_s: float) -> None:
    """Fetch ``[start, end]`` into its place in ``partial``."""
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Range": f"bytes={start}-{end}"}
    )
    written = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 206:
            raise OSError(
                f"expected a partial response for bytes {start}-{end}, got {response.status}"
            )
        with partial.open("r+b") as out:
            out.seek(start)
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                written += len(chunk)
    if written != end - start + 1:
        raise OSError(f"bytes {start}-{end}: got {written} of {end - start + 1}")


def download(
    url: str,
    target: Path,
    connections: int = 8,
    timeout_s: float = 120.0,
    retries: int = 5,
    log=print,
) -> Path:
    """Download ``url`` to ``target`` in parallel ranges. Idempotent and resumable."""
    target = Path(target)
    size = remote_size(url)
    if target.exists() and target.stat().st_size == size:
        log(f"{target.name}: already complete ({size:,} bytes)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    markers = target.with_name(target.name + ".ranges")
    markers.mkdir(exist_ok=True)
    if not partial.exists() or partial.stat().st_size != size:
        shutil.rmtree(markers)
        markers.mkdir()
        with partial.open("wb") as out:
            out.truncate(size)

    n_ranges = max(1, connections * 4)
    step = -(-size // n_ranges)
    ranges = [
        (i, i * step, min(size, (i + 1) * step) - 1) for i in range(n_ranges) if i * step < size
    ]
    log(f"{target.name}: {size:,} bytes in {len(ranges)} ranges over {connections} connections")

    def one(item: tuple[int, int, int]) -> str | None:
        index, start, end = item
        marker = markers / f"{index:04d}"
        if marker.exists():
            return None
        try:
            _fetch_range(url, start, end, partial, timeout_s)
        except OSError as error:  # urllib errors are OSError subclasses
            return f"range {index}: {error}"
        marker.touch()
        return None

    for attempt in range(1, retries + 1):
        with ThreadPoolExecutor(max_workers=connections) as pool:
            failures = [failure for failure in pool.map(one, ranges) if failure]
        if not failures:
            break
        log(f"{target.name}: attempt {attempt} left {len(failures)} ranges incomplete, retrying")
    else:
        raise OSError(f"{target.name}: gave up after {retries} attempts")

    if partial.stat().st_size != size:
        raise OSError(f"{target.name}: {partial.stat().st_size} of {size} bytes")
    partial.replace(target)
    shutil.rmtree(markers)
    log(f"{target.name}: complete")
    return target
