"""Is the package the repository actually ships the package it thinks it is?

This file exists because of a real failure. The first commit's `.gitignore`
contained `data/`, meaning to exclude the generated fleet at the repository
root. That pattern matches a directory called `data` at *any* depth, so it also
excluded `src/cobalt/data/` — the schema and the split protocols, the core of
the repository's argument.

Nothing local noticed. The working tree had the files, the tests passed, the
Docker image built from the working tree and passed too. Only a fresh clone was
broken, and the first sign of it was ruff failing to resolve an import on a CI
runner, which is a long way from the cause.

So: walk the package, import everything, and check the declared entry points
resolve. Cheap, and it turns a class of silent packaging errors into an
immediate failure with the missing module named.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import cobalt

#: Every subpackage that must exist. Listed explicitly rather than discovered,
#: because discovery on a broken checkout discovers the breakage as normal.
REQUIRED_SUBPACKAGES = (
    "cobalt.business",
    "cobalt.data",
    "cobalt.evaluate",
    "cobalt.features",
    "cobalt.hal",
    "cobalt.models",
    "cobalt.service",
    "cobalt.sim",
)

REQUIRED_MODULES = (
    "cobalt.check_readme",
    "cobalt.cli",
    "cobalt.config",
    "cobalt.deploy",
)


@pytest.mark.parametrize("name", REQUIRED_SUBPACKAGES + REQUIRED_MODULES)
def test_every_declared_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_every_module_on_disk_imports() -> None:
    """Walk what is actually installed and import all of it."""
    failures: list[str] = []
    for module in pkgutil.walk_packages(cobalt.__path__, prefix="cobalt."):
        try:
            importlib.import_module(module.name)
        except Exception as error:  # noqa: BLE001 - the point is to report any of them
            failures.append(f"{module.name}: {type(error).__name__}: {error}")
    assert not failures, "\n".join(failures)


def test_every_source_directory_is_a_package() -> None:
    """A directory of modules without an __init__.py will not be installed."""
    root = Path(cobalt.__file__).resolve().parent
    missing = [
        str(directory.relative_to(root.parent))
        for directory in root.rglob("*")
        if directory.is_dir()
        and directory.name != "__pycache__"
        and any(directory.glob("*.py"))
        and not (directory / "__init__.py").exists()
    ]
    assert not missing, f"directories with modules but no __init__.py: {missing}"


def test_the_version_is_exposed() -> None:
    assert cobalt.__version__
    assert cobalt.__version__[0].isdigit()


def test_the_console_entry_point_resolves() -> None:
    """`cobalt` on the command line has to reach something callable."""
    from cobalt.cli import main

    assert callable(main)


def test_the_gitignore_does_not_exclude_any_source_file() -> None:
    """The exact failure this file was written for, asserted directly.

    Reads the ignore file rather than shelling out to git, so it works inside
    the Docker image where there is no repository. Only the pattern shape is
    checked: a bare `name/` pattern matches at any depth, and for anything that
    shares a name with a source directory that is a trap.
    """
    from cobalt.config import repository_root

    gitignore = repository_root() / ".gitignore"
    if not gitignore.exists():
        pytest.skip("no .gitignore, probably running from an installed copy")

    package_root = Path(cobalt.__file__).resolve().parent
    source_directories = {
        directory.name
        for directory in package_root.rglob("*")
        if directory.is_dir() and directory.name != "__pycache__"
    }
    source_directories.add(package_root.name)

    offending = []
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#") or pattern.startswith("/"):
            continue
        name = pattern.rstrip("/")
        if "/" in name:
            continue
        if name in source_directories:
            offending.append(pattern)

    assert not offending, (
        f".gitignore patterns {offending} match source directories at any depth. "
        f"Anchor them with a leading slash."
    )
