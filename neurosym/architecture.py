from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


_LAYER_ALLOWED = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "application": {"domain", "ports", "application"},
    "adapters": {"domain", "ports", "application", "adapters"},
    "reporting": {"domain", "ports", "application", "adapters", "reporting"},
}


@dataclass(frozen=True)
class ArchitectureReport:
    dependency_violations: Tuple[str, ...]
    cycles: Tuple[Tuple[str, ...], ...]

    @property
    def passed(self) -> bool:
        return not self.dependency_violations and not self.cycles


def _module_for(root: Path, path: Path) -> str:
    relative = path.relative_to(root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    output: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            output.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            output.add(node.module)
    return output


def _layer(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "neurosym" else None


def analyze_architecture(root: Path) -> ArchitectureReport:
    package_root = Path(root)
    files = sorted(package_root.rglob("*.py"))
    module_paths = {_module_for(package_root, path): path for path in files}
    graph: Dict[str, Set[str]] = {module: set() for module in module_paths}
    violations: List[str] = []
    for module, path in module_paths.items():
        source_layer = _layer(module)
        for imported in _imports(path):
            target = imported
            while target and target not in module_paths:
                target = target.rpartition(".")[0]
            if target:
                graph[module].add(target)
            target_layer = _layer(imported)
            if source_layer in _LAYER_ALLOWED and target_layer:
                if target_layer not in _LAYER_ALLOWED[source_layer]:
                    violations.append(f"{module} -> {imported}")

    cycles: Set[Tuple[str, ...]] = set()
    visiting: List[str] = []
    visited: Set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            index = visiting.index(module)
            cycle = tuple(visiting[index:] + [module])
            rotations = [cycle[index:-1] + cycle[:index] for index in range(len(cycle) - 1)]
            canonical = min(rotations)
            cycles.add(canonical + (canonical[0],))
            return
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return ArchitectureReport(
        dependency_violations=tuple(sorted(set(violations))),
        cycles=tuple(sorted(cycles)),
    )
