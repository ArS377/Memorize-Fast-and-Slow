from experiments.retrieval_report import evaluate_retrieval_rows, recall_at_k, render_markdown


def test_recall_at_k_uses_relevant_fact_ids() -> None:
    assert recall_at_k(["f1", "f2"], ["f2", "f3"], 1) == 0.0
    assert recall_at_k(["f1", "f2"], ["f2", "f3"], 2) == 0.5


def test_retrieval_report_compares_modes_and_fusion_diagnostics() -> None:
    rows = [
        {
            "mode": "sparse",
            "retrieved_fact_ids": ["f1"],
            "relevant_fact_ids": ["f2"],
            "branch_counts": {"sparse": 1, "dense": 0},
            "sparse_fact_ids": ["f1"],
        },
        {
            "mode": "hybrid",
            "retrieved_fact_ids": ["f2", "f1"],
            "relevant_fact_ids": ["f2"],
            "branch_counts": {"sparse": 1, "dense": 2},
            "sparse_fact_ids": ["f1"],
            "dense_fact_ids": ["f2", "f1"],
            "pre_fusion_fact_ids": ["f1", "f2", "f1"],
        },
    ]

    report = evaluate_retrieval_rows(rows)

    assert report["modes"]["sparse"]["recall_at_1"] == 0.0
    assert report["modes"]["hybrid"]["recall_at_1"] == 1.0
    assert report["modes"]["hybrid"]["branch_overlap_fact_count"] == 1
    assert report["modes"]["hybrid"]["fusion_duplicate_rate"] == 0.333333
    assert "Retrieval Quality Report" in render_markdown(report)

    unlabeled = evaluate_retrieval_rows([
        {"mode": "dense", "retrieved_fact_ids": ["f1"], "relevant_fact_ids": []}
    ])
    assert unlabeled["modes"]["dense"]["recall_at_1"] is None
    assert "n/a" in render_markdown(unlabeled)


def test_retrieval_report_emits_regression_and_degradation_warnings() -> None:
    rows = [
        {
            "cell_id": 5,
            "example_id": "ex1",
            "mode": "sparse",
            "retrieved_fact_ids": ["f1"],
            "relevant_fact_ids": ["f1"],
            "relevance_source": "cited_fact_ids",
        },
        {
            "cell_id": 5,
            "example_id": "ex1",
            "mode": "hybrid",
            "retrieved_fact_ids": ["f2"],
            "relevant_fact_ids": ["f2"],
            "relevance_source": "cited_fact_ids",
            "degraded": True,
        },
    ]

    report = evaluate_retrieval_rows(rows)

    assert report["modes"]["sparse"]["recall_at_1"] == 0.5
    assert report["modes"]["hybrid"]["recall_at_1"] == 0.5
    assert any(warning["code"] == "retrieval_mode_degraded" for warning in report["warnings"])

    rows[1]["retrieved_fact_ids"] = []
    regressed = evaluate_retrieval_rows(rows)
    assert any(
        warning["code"] == "retrieval_recall_regression"
        and warning["mode"] == "hybrid"
        and warning["metric"] == "recall_at_1"
        for warning in regressed["warnings"]
    )
    assert "Regression Warnings" in render_markdown(regressed)
