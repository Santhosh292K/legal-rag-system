"""Every module must import cleanly.

Cheap, but it is the check that catches a rename or a moved constant that
only one call site still references — several modules here are only
reachable through optional branches (ablation flags, the document track,
the REPL's /graph command) and would otherwise fail first in production.
"""
import importlib
import pkgutil
from pathlib import Path

import pytest

RAG_ROOT = Path(__file__).resolve().parent.parent

# Modules whose import has an unavoidable side effect or a heavy optional
# dependency. Everything else must import with nothing running.
SKIP = set()


def _module_names() -> list[str]:
    names = []
    for package in ("pipeline", "data", "evaluation"):
        pkg_path = RAG_ROOT / package
        if not pkg_path.exists():
            continue
        names.append(package)
        for mod in pkgutil.iter_modules([str(pkg_path)]):
            names.append(f"{package}.{mod.name}")
        for sub in pkgutil.iter_modules([str(pkg_path)]):
            sub_path = pkg_path / sub.name
            if sub_path.is_dir():
                for leaf in pkgutil.iter_modules([str(sub_path)]):
                    names.append(f"{package}.{sub.name}.{leaf.name}")
    return sorted(set(names) - SKIP)


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_top_level_modules_import():
    for name in ("config", "main"):
        importlib.import_module(name)


def test_lazy_package_exports_resolve():
    """pipeline/__init__ resolves its names through __getattr__ instead of
    importing everything eagerly; every advertised name must still work."""
    import pipeline
    for name in pipeline.__all__:
        assert getattr(pipeline, name) is not None


def test_unknown_package_attribute_raises():
    import pipeline
    with pytest.raises(AttributeError):
        pipeline.NoSuchThing
