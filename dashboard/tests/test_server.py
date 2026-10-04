import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from dashboard.server import DashboardConfig, build_dashboard, create_server


class DashboardCollationTests(unittest.TestCase):
    """Verify artifact discovery and conservative run labeling."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository_root = Path(self.temporary_directory.name)
        self.results_root = self.repository_root / "results"
        self.results_root.mkdir()

    def write_json(self, relative_path, payload):
        """Write a JSON fixture beneath the temporary repository."""
        path = self.repository_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_collates_complete_and_partial_runs_with_provenance(self):
        self.write_json(
            "results/complete/manifest.json",
            {
                "run_id": "complete-run",
                "status": "completed",
                "git_sha": "1234567890abcdef",
                "git_dirty": True,
                "timestamp": "2026-08-09T10:00:00+00:00",
                "artifacts": ["predictions.jsonl"],
                "artifact_sha256": {
                    "predictions.jsonl": hashlib.sha256(
                        b'{"prediction":"accept"}\n'
                    ).hexdigest()
                },
            },
        )
        self.write_json(
            "results/complete/metrics.json",
            {"accuracy": 0.75, "nested": {"f1": 0.5}},
        )
        (self.results_root / "complete" / "report.md").write_text(
            "# Report\n", encoding="utf-8"
        )
        (self.results_root / "complete" / "predictions.jsonl").write_text(
            '{"prediction":"accept"}\n', encoding="utf-8"
        )
        (self.results_root / "complete" / "credentials.env").write_text(
            "SECRET=not-public\n", encoding="utf-8"
        )
        self.write_json(
            "results/partial/manifest.json",
            {"run_id": "partial-run", "cell_status": {"1": "complete", "2": "pending"}},
        )

        dashboard = build_dashboard(self.repository_root, self.results_root)

        runs = {run["id"]: run for run in dashboard["runs"]}
        complete = runs["complete"]
        partial = runs["partial"]
        self.assertEqual(complete["status"], "completed")
        self.assertEqual(complete["status_provenance"], "manifest.status")
        self.assertFalse(complete["incomplete"])
        self.assertEqual(complete["primary_metric"]["value"], 0.75)
        self.assertEqual(
            [metric["key"] for metric in complete["score_metrics"]],
            ["accuracy", "nested.f1"],
        )
        self.assertEqual(complete["git_sha"], "1234567890abcdef")
        self.assertTrue(complete["git_dirty"])
        self.assertEqual(
            {artifact["name"] for artifact in complete["artifacts"]},
            {"manifest.json", "metrics.json", "predictions.jsonl", "report.md"},
        )
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["status_provenance"], "manifest.cell_status")
        self.assertTrue(partial["incomplete"])
        self.assertCountEqual(partial["missing_artifacts"], ["metrics.json", "report.md"])
        self.assertEqual(dashboard["summary"]["run_count"], 2)
        self.assertEqual(dashboard["summary"]["scored_run_count"], 1)

    def test_reports_invalid_json_without_dropping_the_run(self):
        run_directory = self.results_root / "broken"
        run_directory.mkdir()
        (run_directory / "metrics.json").write_text("{broken", encoding="utf-8")

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertEqual(run["status"], "invalid")
        self.assertTrue(run["incomplete"])
        self.assertIn("metrics.json", run["parse_errors"])

    def test_extracts_dynamic_model_comparison_and_structured_notes(self):
        self.write_json(
            "results/model-run/metrics.json",
            {
                "models": {
                    "learned_gate": {
                        "test": {
                            "accuracy": 0.8,
                            "by_label": {
                                "accept": {"recall": 0.9},
                                "reject": {"recall": 0.7},
                                "replace": {"recall": "not-numeric"},
                            },
                            "hard_gate_violations": 2,
                        },
                        "training": {"initial_loss": 0.7, "final_loss": 0.1},
                    },
                    "symbolic_reference": {
                        "test": {
                            "accuracy": 0.75,
                            "hard_gate_violations": 0,
                        }
                    },
                    "invalid_model": "not-an-object",
                },
                "interpretation": {
                    "summary": "Measured on the held-out synthetic split.",
                    "limitations": ["No external-domain evaluation."],
                    "sample_count": 100,
                },
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        comparison = run["model_comparison"]
        self.assertEqual(
            [model["name"] for model in comparison["models"]],
            ["learned_gate", "symbolic_reference"],
        )
        learned = comparison["models"][0]
        self.assertEqual(learned["test_accuracy"], 0.8)
        self.assertEqual(
            learned["per_label_recall"],
            [
                {"label": "accept", "recall": 0.9},
                {"label": "reject", "recall": 0.7},
            ],
        )
        self.assertEqual(learned["hard_gate_violations"], 2)
        self.assertEqual(learned["training_loss"], {"initial": 0.7, "final": 0.1})
        self.assertEqual(
            comparison["notes"],
            [
                {
                    "path": "interpretation.summary",
                    "text": "Measured on the held-out synthetic split.",
                },
                {
                    "path": "interpretation.limitations[0]",
                    "text": "No external-domain evaluation.",
                },
            ],
        )

    def test_model_comparison_does_not_invent_interpretation(self):
        self.write_json(
            "results/no-claims/metrics.json",
            {"models": {"model_from_metrics": {"test": {"accuracy": 1.0}}}},
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertEqual(run["model_comparison"]["notes"], [])

    def test_extracts_every_dynamically_named_stress_scenario(self):
        self.write_json(
            "results/stress-run/metrics.json",
            {
                "models": {
                    "model_from_metrics": {
                        "stress": {
                            "custom_shift_beta": {
                                "stddev": 0.125,
                                "repeat_count": 7,
                                "accuracy_mean": 0.82,
                                "accuracy_min": 0.76,
                                "accuracy_max": 0.88,
                                "eligible_accuracy_mean": 0.73,
                                "eligible_accuracy_min": 0.61,
                                "eligible_accuracy_max": 0.81,
                                "hard_gate_violations_max": 3,
                            },
                            "unseen_scenario_alpha": {
                                "stddev": 0.4,
                                "repeat_count": 4,
                                "accuracy_mean": 0.65,
                                "accuracy_min": 0.5,
                                "accuracy_max": 0.75,
                                "hard_gate_violations_max": 1,
                            },
                            "malformed_scenario": "not-an-object",
                        }
                    }
                }
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        model = run["model_comparison"]["models"][0]
        self.assertEqual(
            model["stress_scenarios"],
            [
                {
                    "name": "custom_shift_beta",
                    "stddev": 0.125,
                    "repeat_count": 7,
                    "accuracy_mean": 0.82,
                    "accuracy_min": 0.76,
                    "accuracy_max": 0.88,
                    "eligible_accuracy_mean": 0.73,
                    "eligible_accuracy_min": 0.61,
                    "eligible_accuracy_max": 0.81,
                    "hard_gate_violations_max": 3,
                },
                {
                    "name": "unseen_scenario_alpha",
                    "stddev": 0.4,
                    "repeat_count": 4,
                    "accuracy_mean": 0.65,
                    "accuracy_min": 0.5,
                    "accuracy_max": 0.75,
                    "eligible_accuracy_mean": None,
                    "eligible_accuracy_min": None,
                    "eligible_accuracy_max": None,
                    "hard_gate_violations_max": 1,
                },
            ],
        )

    def test_extracts_dynamic_context_benchmark_with_sourced_interpretation(self):
        score = {
            "answer_accuracy": 0.8,
            "gold_evidence_recall": 0.7,
            "target_thread_precision": 0.6,
            "case_count": 9,
        }
        self.write_json(
            "results/context-run/metrics.json",
            {
                "methods": {"custom_retriever": score},
                "realized_context_tiers": {
                    "unseen_context_tier": {
                        "model_input_tokens_min": 1000,
                        "model_input_tokens_max": 1200,
                        "requested_target_thread_percent": 12.5,
                        "target_thread_percent_mean": 12.1,
                        "gold_position_absolute_error_max": 1.2,
                        "hard_lexical_distractors": 3,
                        "case_count": 9,
                    }
                },
                "window_truncation": {"custom_window": score},
                "by_context_tier": {
                    "unseen_context_tier": {"custom_retriever": score}
                },
                "by_evidence_position": {
                    "37": {"custom_retriever": score}
                },
                "window_truncation_by_context_tier": {
                    "unseen_context_tier": {"custom_window": score}
                },
                "window_truncation_by_evidence_position": {
                    "37": {"custom_window": score}
                },
                "dataset": {
                    "model_parameter_axis_evaluated": False,
                    "model_parameter_tiers": [
                        {
                            "name": "metadata_band",
                            "min_billions": 2.0,
                            "max_billions": 8.0,
                        }
                    ],
                },
                "interpretation": {
                    "limitations": [
                        "Parameter tiers are metadata only.",
                        "No model was evaluated in this fixture.",
                    ]
                },
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        benchmark = run["context_benchmark"]
        self.assertEqual(
            benchmark["methods"],
            [
                {
                    "name": "custom_retriever",
                    "grounded_answer_accuracy": None,
                    "answer_accuracy": 0.8,
                    "evidence_recall": 0.7,
                    "exact_evidence_hit_rate": None,
                    "cross_task_interference_rate": None,
                    "stale_memory_intrusion_rate": None,
                    "retraction_compliance": None,
                    "thread_precision": 0.6,
                    "case_count": 9,
                }
            ],
        )
        self.assertEqual(
            benchmark["realized_context_tiers"],
            [
                {
                    "name": "unseen_context_tier",
                    "token_min": 1000,
                    "token_max": 1200,
                    "requested_target_percent": 12.5,
                    "realized_target_percent": 12.1,
                    "max_position_error": 1.2,
                    "hard_lexical_count": 3,
                    "case_count": 9,
                }
            ],
        )
        self.assertEqual(benchmark["windows"][0]["name"], "custom_window")
        self.assertEqual(
            [breakdown["source"] for breakdown in benchmark["breakdowns"]],
            [
                "by_context_tier",
                "by_evidence_position",
                "window_truncation_by_context_tier",
                "window_truncation_by_evidence_position",
            ],
        )
        self.assertEqual(
            benchmark["parameter_tiers"],
            [{"name": "metadata_band", "min_billions": 2.0, "max_billions": 8.0}],
        )
        self.assertEqual(
            benchmark["interpretation_notes"],
            [
                {
                    "path": "interpretation.limitations[0]",
                    "text": "Parameter tiers are metadata only.",
                },
                {
                    "path": "interpretation.limitations[1]",
                    "text": "No model was evaluated in this fixture.",
                },
            ],
        )

    def test_context_benchmark_does_not_derive_model_evaluation_claim(self):
        self.write_json(
            "results/context-no-claims/metrics.json",
            {
                "methods": {"dynamic_method": {"answer_accuracy": 1.0}},
                "dataset": {
                    "model_parameter_axis_evaluated": False,
                    "model_parameter_tiers": [{"name": "dynamic_tier"}],
                },
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertEqual(run["context_benchmark"]["interpretation_notes"], [])

    def test_extracts_continual_memory_scores_provenance_and_integrity(self):
        run_directory = self.results_root / "continual-memory"
        run_directory.mkdir()
        score = {
            "grounded_answer_accuracy": 0.85,
            "answer_accuracy": 0.95,
            "evidence_recall": 0.9,
            "exact_evidence_hit_rate": 0.85,
            "cross_task_interference_rate": 0.03,
            "stale_memory_intrusion_rate": 0.05,
            "retraction_compliance": 1.0,
            "checkpoint_count": 1080,
        }
        self.write_json(
            "results/continual-memory/metrics.json",
            {
                "methods": {"dense": score},
                "retention_by_interference_tier": {
                    "extreme_interference": {"dense": score}
                },
                "dataset": {"model_parameter_tiers": []},
                "memory_growth": {
                    "checkpoint_count": 1080,
                    "stored_turns_mean": 80.5,
                    "stored_turns_max": 429,
                    "stored_tokens_mean": 4747.7,
                    "stored_tokens_max": 25488,
                    "by_interference_tier": {
                        "extreme_interference": {
                            "checkpoint_count": 270,
                            "stored_turns_mean": 231.0,
                            "stored_turns_max": 429,
                            "stored_tokens_mean": 13604.5,
                            "stored_tokens_max": 25488,
                        }
                    },
                },
                "interpretation": {
                    "summary": "Continual memory without online parameter updates."
                },
                "scallop_reasoning_ablation": {
                    "enabled": True,
                    "engine": "scallopy",
                    "scallopy_version": "0.2.4",
                    "causal_feature": "recursive transitive closure",
                    "methods": {
                        "scallop_recursive": {
                            "accuracy": 1.0,
                            "by_checkpoint_kind": {
                                "private": {"accuracy": 1.0},
                                "private-lineage-positive": {"accuracy": 1.0},
                                "private-lineage": {"accuracy": 1.0},
                            },
                        },
                        "scallop_one_hop": {
                            "accuracy": 2 / 3,
                            "by_checkpoint_kind": {
                                "private": {"accuracy": 1.0},
                                "private-lineage-positive": {"accuracy": 1.0},
                                "private-lineage": {"accuracy": 0.0},
                            },
                        },
                    },
                },
                "scallop_stream_injection_ablation": {
                    "enabled": True,
                    "scallop_query_or_gold_used": False,
                    "capsule_selector": "BM25 over visible query text, one slot per enabled relation",
                    "source_grounded": True,
                    "engine": "scallopy",
                    "scallopy_versions": ["0.2.4"],
                    "rule_version": "preference_stream.v1",
                    "by_window": {
                        "window_128k": {
                            "deep_context_rot": {
                                "preference-change-delayed": {
                                    "matched_count": 30,
                                    "no_injection_grounded_accuracy": 0.0,
                                    "scallop_injected_grounded_accuracy": 1.0,
                                    "reverse_ranked_control_grounded_accuracy": 0.0,
                                    "change_rule_only_grounded_accuracy": 1.0,
                                    "incongruity_rule_only_grounded_accuracy": 0.0,
                                    "paired_delta": 1.0,
                                }
                            }
                        }
                    },
                },
            },
        )
        (run_directory / "report.md").write_text("# Report\n", encoding="utf-8")
        (run_directory / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
        (run_directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        artifact_names = [
            "episodes.jsonl",
            "predictions.jsonl",
            "metrics.json",
            "report.md",
        ]
        self.write_json(
            "results/continual-memory/manifest.json",
            {
                "run_id": "continual-memory",
                "status": "completed",
                "source": {"git_sha": "abcdef1234567890", "git_dirty": True},
                "artifacts": artifact_names,
                "artifact_sha256": {
                    name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest()
                    for name in artifact_names
                },
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertFalse(run["incomplete"])
        self.assertEqual(run["git_sha"], "abcdef1234567890")
        self.assertTrue(run["git_dirty"])
        self.assertEqual(
            run["primary_metric"]["key"],
            "methods.dense.grounded_answer_accuracy",
        )
        self.assertEqual(run["artifact_integrity_errors"], {})
        self.assertEqual(
            run["context_benchmark"]["methods"],
            [
                {
                    "name": "dense",
                    "grounded_answer_accuracy": 0.85,
                    "answer_accuracy": 0.95,
                    "evidence_recall": 0.9,
                    "exact_evidence_hit_rate": 0.85,
                    "cross_task_interference_rate": 0.03,
                    "stale_memory_intrusion_rate": 0.05,
                    "retraction_compliance": 1.0,
                    "thread_precision": None,
                    "case_count": 1080,
                }
            ],
        )
        self.assertEqual(
            [item["source"] for item in run["context_benchmark"]["breakdowns"]],
            ["retention_by_interference_tier"],
        )
        self.assertEqual(
            run["context_benchmark"]["memory_growth"]["stored_turns_max"], 429
        )
        self.assertEqual(
            run["context_benchmark"]["scallop_reasoning_ablation"],
            {
                "engine": "scallopy",
                "version": "0.2.4",
                "causal_feature": "recursive transitive closure",
                "methods": [
                    {
                        "name": "scallop_one_hop",
                        "accuracy": 2 / 3,
                        "direct_accuracy": 1.0,
                        "positive_accuracy": 1.0,
                        "recursive_probe_accuracy": 0.0,
                    },
                    {
                        "name": "scallop_recursive",
                        "accuracy": 1.0,
                        "direct_accuracy": 1.0,
                        "positive_accuracy": 1.0,
                        "recursive_probe_accuracy": 1.0,
                    },
                ],
            },
        )
        self.assertEqual(
            run["context_benchmark"]["scallop_stream_injection_ablation"],
            {
                "scallop_query_or_gold_used": False,
                "capsule_selector": "BM25 over visible query text, one slot per enabled relation",
                "source_grounded": True,
                "engine": "scallopy",
                "scallopy_versions": ["0.2.4"],
                "rule_version": "preference_stream.v1",
                "rows": [
                    {
                        "window": "window_128k",
                        "tier": "deep_context_rot",
                        "checkpoint": "preference-change-delayed",
                        "matched_count": 30,
                        "raw_grounded_accuracy": 0.0,
                        "scallop_grounded_accuracy": 1.0,
                        "reverse_ranked_grounded_accuracy": 0.0,
                        "change_only_grounded_accuracy": 1.0,
                        "incongruity_only_grounded_accuracy": 0.0,
                        "paired_delta": 1.0,
                    }
                ],
            },
        )

    def test_declared_missing_or_hash_mismatched_artifacts_are_incomplete(self):
        run_directory = self.results_root / "integrity-failure"
        run_directory.mkdir()
        self.write_json("results/integrity-failure/metrics.json", {"accuracy": 1.0})
        (run_directory / "report.md").write_text("changed\n", encoding="utf-8")
        self.write_json(
            "results/integrity-failure/manifest.json",
            {
                "status": "completed",
                "artifacts": ["metrics.json", "report.md", "predictions.jsonl"],
                "artifact_sha256": {"report.md": "0" * 64},
            },
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertTrue(run["incomplete"])
        self.assertIn("predictions.jsonl", run["missing_artifacts"])
        self.assertIn("report.md", run["artifact_integrity_errors"])
        self.assertIn("metrics.json", run["artifact_integrity_errors"])

    def test_extracts_interleaved_horizon_loss_without_method_table(self):
        run_directory = self.results_root / "interleaved"
        run_directory.mkdir()
        self.write_json(
            "results/interleaved/metrics.json",
            {
                "horizons": [
                    {
                        "horizon_accounts": 1024,
                        "stream_token_count": 3098845,
                        "suffix_contract_loss_rate": {
                            "4096": 0.9983,
                            "1048576": 0.6621,
                        },
                    }
                ],
                "contradiction_ledger": {
                    "engine": "scallopy",
                    "pair_count": 1024,
                    "resolved_pair_count": 1024,
                },
                "online_checkpoint_evaluation": {
                    "answer_evaluator": "deterministic oracle resolver; not LLM accuracy",
                    "methods": {
                        "full_structured_memory": {
                            "checkpoint_count": 4096,
                            "oracle_resolver_answer_accuracy": 1.0,
                            "grounded_answer_accuracy": 1.0,
                            "complete_provenance_rate": 1.0,
                            "evidence_recall": 1.0,
                            "stale_memory_intrusion_rate": 0.0,
                        }
                    },
                },
                "context_horizon_contract": {
                    "declared_context_limit": 1048576,
                    "required_multiplier": 2.0,
                    "stream_token_count": 2733929,
                    "realized_multiplier": 2.607,
                },
                "interpretation": {"summary": "Peer-task finite-context loss."},
            },
        )
        self.write_json(
            "results/interleaved/manifest.json", {"status": "completed"}
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]
        benchmark = run["context_benchmark"]

        self.assertEqual(benchmark["methods"][0]["answer_accuracy"], 1.0)
        self.assertEqual(benchmark["methods"][0]["exact_evidence_hit_rate"], 1.0)
        self.assertEqual(benchmark["methods"][0]["case_count"], 4096)
        self.assertEqual(
            benchmark["interleaved_horizons"],
            [
                {
                    "horizon_accounts": 1024,
                    "stream_token_count": 3098845,
                    "loss_by_window": [
                        {"window_tokens": 4096, "loss_rate": 0.9983},
                        {"window_tokens": 1048576, "loss_rate": 0.6621},
                    ],
                }
            ],
        )
        self.assertEqual(
            benchmark["contradiction_ledger"]["engine"], "scallopy"
        )
        self.assertEqual(
            benchmark["context_horizon_contract"]["stream_token_count"], 2733929
        )

    def test_malformed_interleaved_window_key_is_skipped(self):
        run_directory = self.results_root / "malformed-interleaved"
        run_directory.mkdir()
        self.write_json(
            "results/malformed-interleaved/metrics.json",
            {
                "horizons": [
                    {
                        "horizon_accounts": 8,
                        "stream_token_count": 100,
                        "suffix_contract_loss_rate": {"bad": 0.5, "4096": 0.25},
                    }
                ]
            },
        )
        self.write_json(
            "results/malformed-interleaved/manifest.json", {"status": "completed"}
        )

        run = build_dashboard(self.repository_root, self.results_root)["runs"][0]

        self.assertEqual(
            run["context_benchmark"]["interleaved_horizons"][0]["loss_by_window"],
            [{"window_tokens": 4096, "loss_rate": 0.25}],
        )


class DashboardServerTests(unittest.TestCase):
    """Verify HTTP behavior using a real ephemeral local server."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository_root = Path(self.temporary_directory.name)
        self.results_root = self.repository_root / "results"
        self.results_root.mkdir()
        run_directory = self.results_root / "live-run"
        run_directory.mkdir()
        (run_directory / "manifest.json").write_text(
            json.dumps({"status": "running"}), encoding="utf-8"
        )
        static_root = self.repository_root / "static"
        static_root.mkdir()
        (static_root / "index.html").write_text("dashboard shell", encoding="utf-8")
        code_path = self.repository_root / "neurosym" / "architecture.py"
        code_path.parent.mkdir()
        code_path.write_text("ARCHITECTURE = True\n", encoding="utf-8")
        (self.repository_root / "secret.txt").write_text("not public\n", encoding="utf-8")
        (run_directory / "credentials.env").write_text("SECRET=not-public\n", encoding="utf-8")
        config = DashboardConfig(
            repository_root=self.repository_root,
            results_root=self.results_root,
            static_root=static_root,
        )
        self.server = create_server("127.0.0.1", 0, config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def get(self, path):
        """Fetch an endpoint and return its response body and headers."""
        with urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return response.read(), response.headers

    def test_serves_shell_health_dynamic_data_and_artifact(self):
        body, headers = self.get("/")
        self.assertEqual(body, b"dashboard shell")
        self.assertIn("text/html", headers["Content-Type"])

        health_body, _ = self.get("/api/health")
        self.assertEqual(json.loads(health_body), {"status": "ok"})

        first_body, first_headers = self.get("/api/dashboard")
        first_payload = json.loads(first_body)
        self.assertEqual(first_payload["summary"]["scored_run_count"], 0)
        self.assertEqual(first_headers["Cache-Control"], "no-store")

        metrics_path = self.results_root / "live-run" / "metrics.json"
        metrics_path.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
        second_body, _ = self.get("/api/dashboard")
        self.assertEqual(json.loads(second_body)["summary"]["scored_run_count"], 1)

        artifact_body, _ = self.get("/files/results/live-run/metrics.json")
        self.assertEqual(json.loads(artifact_body), {"accuracy": 0.9})

        code_body, _ = self.get("/files/code/neurosym/architecture.py")
        self.assertEqual(code_body, b"ARCHITECTURE = True\n")

    def test_rejects_path_traversal(self):
        with self.assertRaises(HTTPError) as error:
            self.get("/files/results/%2e%2e/static/index.html")
        self.assertEqual(error.exception.code, 404)

    def test_code_route_only_serves_declared_references(self):
        with self.assertRaises(HTTPError) as error:
            self.get("/files/code/secret.txt")
        self.assertEqual(error.exception.code, 404)

    def test_result_route_only_serves_canonical_or_manifest_declared_artifacts(self):
        with self.assertRaises(HTTPError) as error:
            self.get("/files/results/live-run/credentials.env")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
