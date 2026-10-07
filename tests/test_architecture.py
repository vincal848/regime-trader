"""The rules in docs/ARCHITECTURE.md, enforced on the source tree.

- Each module belongs to one layer, and imports only from its own layer
  or lower ones.
- `risk` imports nothing but `bars`, so no model can reach a hard limit.
- Layers 0-2 do no I/O.
- hmmlearn's smoothed and Viterbi methods appear only in `calibration`,
  the offline evaluation module. Decisions must use our own forward filter.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "regime_trader"

LAYERS: dict[str, int] = {
    # 0 core: pure, typed, no I/O
    "bars": 0,
    "features": 0,
    "hmm": 0,
    "switching": 0,
    "playbook": 0,
    "sizing": 0,
    "risk": 0,
    "metrics": 0,
    # 1 engine
    "engine": 1,
    # 2 research
    "refit": 2,
    "backtest": 2,
    "calibration": 2,
    # 3 adapters: all I/O
    "store": 3,
    "ibkr": 3,
    "yahoo": 3,
    "alerts": 3,
    "llm": 3,
    # 4 apps
    "live": 4,
    "nightly": 4,
    "dashboard": 4,
    "cli": 4,
}

SMOOTHING_OR_VITERBI = {"predict_proba", "predict", "decode", "score_samples"}
IO_MODULES = {
    "logging",
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "sqlite3",
    "pathlib",
    "time",
}
IO_CALLS = {"open", "print", "input"}


def _source(module: str) -> ast.Module:
    return ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))


def _internal_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("regime_trader."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and node.module == "regime_trader":
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {a.name.split(".")[1] for a in node.names if a.name.startswith("regime_trader.")}
    return found


def _top_level_imports(tree: ast.Module) -> set[str]:
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    return names


def _modules() -> set[str]:
    return {p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__"}


def test_every_module_has_a_layer() -> None:
    assert _modules() <= set(LAYERS), f"unassigned modules: {sorted(_modules() - set(LAYERS))}"


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_imports_never_reach_a_higher_layer(module: str) -> None:
    if module not in _modules():
        pytest.skip(f"{module} not built yet")
    upward = {m for m in _internal_imports(_source(module)) if LAYERS.get(m, -1) > LAYERS[module]}
    assert not upward, f"{module} (layer {LAYERS[module]}) imports higher layers: {sorted(upward)}"


def test_risk_depends_on_nothing_but_bars() -> None:
    if "risk" not in _modules():
        pytest.skip("risk not built yet")
    assert _internal_imports(_source("risk")) <= {"bars"}


@pytest.mark.parametrize("module", sorted(m for m, layer in LAYERS.items() if layer <= 2))
def test_core_engine_and_research_do_no_io(module: str) -> None:
    if module not in _modules():
        pytest.skip(f"{module} not built yet")
    tree = _source(module)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & IO_CALLS, f"{module} calls {sorted(calls & IO_CALLS)}"
    assert not _top_level_imports(tree) & IO_MODULES, (
        f"{module} imports {sorted(_top_level_imports(tree) & IO_MODULES)}"
    )


@pytest.mark.parametrize("module", sorted(m for m in LAYERS if m != "calibration"))
def test_smoothed_or_viterbi_inference_is_confined_to_calibration(module: str) -> None:
    if module not in _modules():
        pytest.skip(f"{module} not built yet")
    tree = _source(module)
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    leaked = attributes & SMOOTHING_OR_VITERBI
    assert not leaked, (
        f"{module} uses look-ahead inference {sorted(leaked)}; decisions must use hmm.forward_filter"
    )
