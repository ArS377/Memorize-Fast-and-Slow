# NeuroSym Results Dashboard

Local, dependency-free dashboard for the live experiment evidence under `results/**`.
It scans `metrics.json`, `manifest.json`, and `report.md` on every API request, so new and updated runs appear without a build step or server restart.

## Launch

From the repository root, choose an available local port:

```bash
python -m dashboard.server --port 8080
```

Open `http://127.0.0.1:8080/`.
The host defaults to `127.0.0.1`.
The port must be supplied explicitly or through `DASHBOARD_PORT`, preventing a hidden fixed-port dependency.

Environment-based launch:

```bash
DASHBOARD_PORT=8080 python -m dashboard.server
```

Use `--host` to expose the server on another interface and `--results` to scan a different results directory:

```bash
python -m dashboard.server --host 0.0.0.0 --port 8080 --results results
```

## Endpoints

- `/` serves the responsive dashboard.
- `/api/health` reports server health.
- `/api/dashboard` rescans and returns current run, metric, artifact, status, and code-reference data.
- `/files/results/...` serves discovered local artifacts.
- `/files/code/...` serves linked implementation references from the checkout.

The browser refreshes `/api/dashboard` every 15 seconds and also provides a manual refresh button.
Run status is taken only from `manifest.status` or `manifest.cell_status`.
Missing files, invalid JSON, unknown state, and differing metric paths remain explicit.
Runs whose metrics contain a top-level `models` object also receive a responsive model comparison with held-out accuracy, per-label recall, hard-gate violations, and available training loss.
Every dynamically named model stress scenario is shown with its reported distribution summary and maximum hard-gate violations.
Interpretation text is shown only when supplied under a structured top-level `interpretation` or `limitations` metrics field.
Runs with top-level `methods` receive a context benchmark section covering aggregate methods, realized context tiers, window truncation, dynamic tier and evidence-position slices, and parameter-tier ranges.
Model-evaluation and parameter-tier interpretation statements are copied only from the metrics `interpretation` field.

## Test

```bash
python -m unittest discover -s dashboard/tests -v
```

The editable architecture source is `dashboard/diagrams/neurosym-dashboard.drawio`; the dashboard renders the dependency-free SVG at `dashboard/static/architecture.svg`.
