from pathlib import Path

from neurosym.architecture import analyze_architecture


def test_canonical_layers_have_no_cycles_or_reverse_dependencies() -> None:
    root = Path(__file__).parent.parent / "neurosym"
    report = analyze_architecture(root)

    assert report.dependency_violations == ()
    assert report.cycles == ()


def test_domain_and_ports_do_not_load_optional_dependencies() -> None:
    import subprocess
    import sys

    code = (
        "import sys; before=set(sys.modules); "
        "import neurosym.domain, neurosym.ports; "
        "loaded=set(sys.modules)-before; "
        "forbidden={'neo4j','openai','sentence_transformers','scallopy','numpy'}; "
        "assert not loaded.intersection(forbidden), loaded.intersection(forbidden)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
