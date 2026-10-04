const state = { data: null };

const byId = (id) => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatNumber(value) {
  if (value === null || value === undefined) return "-";
  if (Number.isInteger(value)) return value.toLocaleString();
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatRate(value) {
  if (value === null || value === undefined) return "Not reported";
  return Number(value).toLocaleString(undefined, {
    style: "percent",
    maximumFractionDigits: 2,
  });
}

function formatLoss(value) {
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 0.0001) return number.toExponential(2);
  return number.toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function statusBadge(run) {
  const badge = element("span", `status-badge ${run.status.replace(/[^a-z0-9_-]/g, "")}`, run.status);
  return badge;
}

function renderKpis(summary) {
  const definitions = [
    [summary.run_count, "Runs discovered"],
    [summary.scored_run_count, "Runs with scores"],
    [summary.complete_run_count, "Complete evidence"],
    [summary.artifact_count, "Artifacts linked"],
  ];
  const grid = byId("kpi-grid");
  grid.replaceChildren();
  definitions.forEach(([value, label], index) => {
    const card = element("article", "kpi");
    card.append(element("span", "kpi-index", `0${index + 1}`));
    card.append(element("strong", "kpi-value", formatNumber(value)));
    card.append(element("span", "kpi-label", label));
    grid.append(card);
  });
}

function renderCurrentState(data) {
  const summary = data.summary;
  const percentage = summary.run_count
    ? Math.round((summary.complete_run_count / summary.run_count) * 100)
    : 0;
  byId("completion-percentage").textContent = `${percentage}%`;
  byId("completion-bar").style.width = `${percentage}%`;

  const statusList = byId("status-list");
  statusList.replaceChildren();
  Object.entries(summary.status_counts)
    .sort((left, right) => right[1] - left[1])
    .forEach(([status, count]) => {
      const row = element("div", "status-line");
      row.append(element("span", "", status));
      row.append(element("strong", "", String(count)));
      statusList.append(row);
    });

  const activeRuns = byId("active-runs");
  activeRuns.replaceChildren();
  const incomplete = data.runs.filter((run) => run.incomplete).slice(0, 7);
  if (!incomplete.length) {
    activeRuns.append(element("p", "empty-state", "No incomplete evidence sets."));
    return;
  }
  incomplete.forEach((run) => {
    const item = element("div", "active-item");
    item.append(element("strong", "", run.name));
    const integrityFailures = Object.keys(run.artifact_integrity_errors || {});
    const missing = run.missing_artifacts.length
      ? `Missing ${run.missing_artifacts.join(", ")}`
      : integrityFailures.length
        ? `Integrity failure: ${integrityFailures.join(", ")}`
      : `State: ${run.status}`;
    item.append(element("span", "", missing));
    activeRuns.append(item);
  });
}

function artifactLinks(run) {
  const wrapper = element("div", "artifact-links");
  run.artifacts.forEach((artifact) => {
    const link = element("a", "", artifact.name.replace(/\.(json|md)$/i, ""));
    link.href = artifact.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.title = `${artifact.path} (${artifact.size_bytes.toLocaleString()} bytes)`;
    wrapper.append(link);
  });
  return wrapper;
}

function filteredRuns() {
  if (!state.data) return [];
  const query = byId("run-search").value.trim().toLowerCase();
  const status = byId("status-filter").value;
  return state.data.runs.filter((run) => {
    const matchesQuery = !query || `${run.name} ${run.id}`.toLowerCase().includes(query);
    const matchesStatus = status === "all" || run.status === status;
    return matchesQuery && matchesStatus;
  });
}

function renderTable() {
  const runs = filteredRuns();
  const body = byId("run-table-body");
  body.replaceChildren();
  byId("table-empty").hidden = runs.length > 0;
  runs.forEach((run) => {
    const row = document.createElement("tr");

    const nameCell = document.createElement("td");
    nameCell.append(element("strong", "run-name", run.name));
    nameCell.append(element("span", "run-path", run.id));
    row.append(nameCell);

    const statusCell = document.createElement("td");
    statusCell.append(statusBadge(run));
    if (run.incomplete) statusCell.append(element("span", "incomplete-label", "Incomplete evidence"));
    row.append(statusCell);

    const metricCell = document.createElement("td");
    if (run.primary_metric) {
      metricCell.append(element("strong", "metric-value", formatNumber(run.primary_metric.value)));
      metricCell.append(element("span", "metric-path", run.primary_metric.key));
    } else {
      metricCell.textContent = "Not reported";
    }
    row.append(metricCell);

    const cellsCell = document.createElement("td");
    const cellStates = Object.entries(run.cell_status_counts);
    cellsCell.textContent = cellStates.length
      ? cellStates.map(([key, count]) => `${key}: ${count}`).join(" / ")
      : "Not declared";
    row.append(cellsCell);

    const artifactsCell = document.createElement("td");
    artifactsCell.append(artifactLinks(run));
    if (run.missing_artifacts.length) {
      artifactsCell.append(element("span", "missing-text", `Missing: ${run.missing_artifacts.join(", ")}`));
    }
    Object.entries(run.artifact_integrity_errors || {}).forEach(([name, message]) => {
      artifactsCell.append(element("span", "error-text", `${name}: ${message}`));
    });
    row.append(artifactsCell);

    const provenanceCell = document.createElement("td");
    provenanceCell.append(element("span", "", run.status_provenance));
    if (run.git_sha) provenanceCell.append(element("code", "provenance-text", run.git_sha.slice(0, 12)));
    if (run.git_dirty === true) provenanceCell.append(element("span", "incomplete-label", "dirty source"));
    row.append(provenanceCell);

    row.append(element("td", "", formatDate(run.timestamp)));
    body.append(row);
  });
}

function renderStatusFilter(runs) {
  const filter = byId("status-filter");
  const selected = filter.value;
  filter.replaceChildren();
  const all = element("option", "", "All states");
  all.value = "all";
  filter.append(all);
  [...new Set(runs.map((run) => run.status))].sort().forEach((status) => {
    const option = element("option", "", status);
    option.value = status;
    filter.append(option);
  });
  filter.value = [...filter.options].some((option) => option.value === selected) ? selected : "all";
}

function renderCodeReferences(references) {
  const container = byId("code-references");
  container.replaceChildren();
  references.forEach((reference) => {
    const link = element("a", "code-reference");
    link.href = reference.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.append(element("span", "", reference.label));
    link.append(element("code", "", reference.path));
    container.append(link);
  });
}

function renderModelComparisons(runs) {
  const section = byId("model-comparison-section");
  const container = byId("model-comparisons");
  const comparisonRuns = runs.filter((run) => run.model_comparison);
  section.hidden = comparisonRuns.length === 0;
  container.replaceChildren();

  comparisonRuns.forEach((run) => {
    const comparison = element("article", "model-comparison-run");
    const heading = element("div", "model-run-heading");
    const identity = element("div", "");
    identity.append(element("h3", "", run.name));
    identity.append(element("code", "run-path", run.id));
    heading.append(identity);
    heading.append(statusBadge(run));
    comparison.append(heading);

    if (run.model_comparison.notes.length) {
      const notes = element("aside", "model-notes");
      notes.append(element("div", "panel-label", "Reported interpretation / limitations"));
      run.model_comparison.notes.forEach((note) => {
        const noteRow = element("div", "model-note");
        noteRow.append(element("code", "", note.path));
        noteRow.append(element("p", "", note.text));
        notes.append(noteRow);
      });
      comparison.append(notes);
    }

    const grid = element("div", "model-grid");
    run.model_comparison.models.forEach((model, index) => {
      const card = element("article", "model-card");
      const cardHeading = element("div", "model-card-heading");
      cardHeading.append(element("span", "model-index", String(index + 1).padStart(2, "0")));
      cardHeading.append(element("h4", "", model.name));
      card.append(cardHeading);

      const accuracy = element("div", "model-accuracy");
      accuracy.append(element("span", "model-metric-label", "Test accuracy"));
      accuracy.append(element("strong", "", formatRate(model.test_accuracy)));
      card.append(accuracy);

      const violations = element("div", "model-stat-line");
      violations.append(element("span", "", "Hard-gate violations"));
      violations.append(
        element(
          "strong",
          "",
          model.hard_gate_violations === null
            ? "Not reported"
            : formatNumber(model.hard_gate_violations),
        ),
      );
      card.append(violations);

      if (model.per_label_recall.length) {
        const recalls = element("div", "recall-list");
        recalls.append(element("span", "model-metric-label", "Per-label recall"));
        model.per_label_recall.forEach((item) => {
          const recall = element("div", "recall-line");
          recall.append(element("span", "", item.label));
          recall.append(element("strong", "", formatRate(item.recall)));
          recalls.append(recall);
        });
        card.append(recalls);
      }

      if (model.stress_scenarios.length) {
        const stress = element("div", "stress-list");
        stress.append(element("span", "model-metric-label", "Stress scenarios"));
        model.stress_scenarios.forEach((scenario) => {
          const scenarioRow = element("div", "stress-scenario");
          const scenarioHeading = element("div", "stress-heading");
          scenarioHeading.append(element("code", "", scenario.name));
          if (scenario.stddev !== null) {
            scenarioHeading.append(element("span", "", `stddev ${formatNumber(scenario.stddev)}`));
          }
          scenarioRow.append(scenarioHeading);

          const metrics = element("div", "stress-metrics");
          [
            ["Eligible mean", formatRate(scenario.eligible_accuracy_mean)],
            ["Eligible min", formatRate(scenario.eligible_accuracy_min)],
            ["Eligible max", formatRate(scenario.eligible_accuracy_max)],
            ["Overall mean", formatRate(scenario.accuracy_mean)],
            ["Repeats", scenario.repeat_count === null ? "Not reported" : formatNumber(scenario.repeat_count)],
            [
              "Gate violations max",
              scenario.hard_gate_violations_max === null
                ? "Not reported"
                : formatNumber(scenario.hard_gate_violations_max),
            ],
          ].forEach(([label, value]) => {
            const metric = element("div", "stress-metric");
            metric.append(element("span", "", label));
            metric.append(element("strong", "", value));
            metrics.append(metric);
          });
          scenarioRow.append(metrics);
          stress.append(scenarioRow);
        });
        card.append(stress);
      }

      if (model.training_loss) {
        const training = element("div", "training-loss");
        training.append(element("span", "model-metric-label", "Training loss"));
        const values = [];
        if (model.training_loss.initial !== undefined) {
          values.push(`initial ${formatLoss(model.training_loss.initial)}`);
        }
        if (model.training_loss.final !== undefined) {
          values.push(`final ${formatLoss(model.training_loss.final)}`);
        }
        training.append(element("strong", "", values.join(" / ")));
        card.append(training);
      }
      grid.append(card);
    });
    comparison.append(grid);
    container.append(comparison);
  });
}

function contextScoreTable(records) {
  const scroll = element("div", "context-table-scroll");
  const table = element("table", "context-table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Method / window", "Grounded accuracy", "Answer accuracy", "Evidence recall", "Exact evidence", "Causal interference", "Stale intrusion", "Retraction compliance", "Thread precision", "Cases"].forEach((label) => {
    headerRow.append(element("th", "", label));
  });
  head.append(headerRow);
  table.append(head);

  const body = document.createElement("tbody");
  records.forEach((record) => {
    const row = document.createElement("tr");
    row.append(element("td", "context-record-name", record.name));
    row.append(element("td", "", formatRate(record.answer_accuracy)));
    row.insertBefore(
      element("td", "", formatRate(record.grounded_answer_accuracy)),
      row.children[1],
    );
    row.append(element("td", "", formatRate(record.evidence_recall)));
    row.append(element("td", "", formatRate(record.exact_evidence_hit_rate)));
    row.append(element("td", "", formatRate(record.cross_task_interference_rate)));
    row.append(element("td", "", formatRate(record.stale_memory_intrusion_rate)));
    row.append(element("td", "", formatRate(record.retraction_compliance)));
    row.append(element("td", "", formatRate(record.thread_precision)));
    row.append(
      element("td", "", record.case_count === null ? "Not reported" : formatNumber(record.case_count)),
    );
    body.append(row);
  });
  table.append(body);
  scroll.append(table);
  return scroll;
}

function contextBlock(title, content) {
  const block = element("section", "context-block");
  block.append(element("h4", "", title));
  block.append(content);
  return block;
}

function memoryGrowthGrid(growth) {
  const grid = element("div", "context-tier-grid");
  const records = [{ name: "Overall", ...growth }, ...growth.by_interference_tier];
  records.forEach((record) => {
    const card = element("article", "context-tier-card");
    card.append(element("h5", "", record.name));
    [
      ["Checkpoints", formatNumber(record.checkpoint_count)],
      ["Mean stored turns", formatNumber(record.stored_turns_mean)],
      ["Maximum stored turns", formatNumber(record.stored_turns_max)],
      ["Mean stored tokens", formatNumber(record.stored_tokens_mean)],
      ["Maximum stored tokens", formatNumber(record.stored_tokens_max)],
    ].forEach(([label, value]) => {
      const metric = element("div", "context-tier-metric");
      metric.append(element("span", "", label));
      metric.append(element("strong", "", value));
      card.append(metric);
    });
    grid.append(card);
  });
  return grid;
}

function scallopAblationTable(ablation) {
  const records = ablation.methods.map((method) => ({
    name: method.name,
    grounded_answer_accuracy: method.accuracy,
    answer_accuracy: method.recursive_probe_accuracy,
    evidence_recall: method.direct_accuracy,
    exact_evidence_hit_rate: method.positive_accuracy,
    cross_task_interference_rate: null,
    stale_memory_intrusion_rate: null,
    retraction_compliance: null,
    thread_precision: null,
    case_count: null,
  }));
  return contextScoreTable(records);
}

function scallopInjectionTable(ablation) {
  const table = element("table", "context-score-table");
  const header = element("tr", "");
  ["Window", "Tier", "Checkpoint", "Raw", "Scallop", "Reverse ranked", "Change only", "Incongruity only", "Delta", "N"].forEach((label) => {
    header.append(element("th", "", label));
  });
  table.append(header);
  ablation.rows.forEach((record) => {
    const row = element("tr", "");
    row.append(element("td", "", record.window));
    row.append(element("td", "", record.tier));
    row.append(element("td", "", record.checkpoint));
    row.append(element("td", "", formatRate(record.raw_grounded_accuracy)));
    row.append(element("td", "", formatRate(record.scallop_grounded_accuracy)));
    row.append(element("td", "", formatRate(record.reverse_ranked_grounded_accuracy)));
    row.append(element("td", "", formatRate(record.change_only_grounded_accuracy)));
    row.append(element("td", "", formatRate(record.incongruity_only_grounded_accuracy)));
    row.append(element("td", "", formatRate(record.paired_delta)));
    row.append(element("td", "", formatNumber(record.matched_count)));
    table.append(row);
  });
  return table;
}

function interleavedHorizonTable(horizons) {
  const windows = [...new Set(horizons.flatMap((horizon) => horizon.loss_by_window.map((item) => item.window_tokens)))];
  const table = element("table", "context-score-table");
  const header = element("tr", "");
  ["Accounts", "Stream tokens", ...windows.map((window) => `${formatNumber(window)} loss`)].forEach((label) => {
    header.append(element("th", "", label));
  });
  table.append(header);
  horizons.forEach((horizon) => {
    const row = element("tr", "");
    row.append(element("td", "", formatNumber(horizon.horizon_accounts)));
    row.append(element("td", "", formatNumber(horizon.stream_token_count)));
    windows.forEach((window) => {
      const measurement = horizon.loss_by_window.find((item) => item.window_tokens === window);
      row.append(element("td", "", formatRate(measurement?.loss_rate ?? null)));
    });
    table.append(row);
  });
  return table;
}

function contextHorizonGrid(contract) {
  const grid = element("div", "context-tier-grid");
  [
    ["Declared context", formatNumber(contract.declared_context_limit)],
    ["Required multiple", `${formatNumber(contract.required_multiplier)}x`],
    ["Stream tokens", formatNumber(contract.stream_token_count)],
    ["Realized multiple", `${Number(contract.realized_multiplier).toFixed(3)}x`],
  ].forEach(([label, value]) => {
    const card = element("article", "context-tier-card");
    card.append(element("h5", "", label));
    card.append(element("strong", "", value));
    grid.append(card);
  });
  return grid;
}

function renderContextBenchmarks(runs) {
  const section = byId("context-benchmark-section");
  const container = byId("context-benchmarks");
  const benchmarkRuns = runs.filter((run) => run.context_benchmark);
  section.hidden = benchmarkRuns.length === 0;
  container.replaceChildren();

  benchmarkRuns.forEach((run) => {
    const benchmark = run.context_benchmark;
    const article = element("article", "context-benchmark-run");
    const heading = element("div", "model-run-heading");
    const identity = element("div", "");
    identity.append(element("h3", "", run.name));
    identity.append(element("code", "run-path", run.id));
    heading.append(identity);
    heading.append(statusBadge(run));
    article.append(heading);

    if (benchmark.interpretation_notes.length) {
      const notes = element("aside", "model-notes context-notes");
      notes.append(element("div", "panel-label", "Reported interpretation / limitations"));
      benchmark.interpretation_notes.forEach((note) => {
        const noteRow = element("div", "model-note");
        noteRow.append(element("code", "", note.path));
        noteRow.append(element("p", "", note.text));
        notes.append(noteRow);
      });
      article.append(notes);
    }

    if (benchmark.methods.length) {
      const title = benchmark.answer_evaluator
        ? "Online checkpoint context availability"
        : "Method outcomes";
      article.append(contextBlock(title, contextScoreTable(benchmark.methods)));
    }

    if (benchmark.context_horizon_contract) {
      article.append(contextBlock(
        "Declared finite-context horizon",
        contextHorizonGrid(benchmark.context_horizon_contract),
      ));
    }

    if (benchmark.interleaved_horizons.length) {
      article.append(contextBlock(
        "Finite-context loss as the task horizon grows",
        interleavedHorizonTable(benchmark.interleaved_horizons),
      ));
    }

    if (benchmark.scallop_reasoning_ablation) {
      const ablation = benchmark.scallop_reasoning_ablation;
      const title = `Scallop reasoning ablation: ${ablation.causal_feature || "recursive closure"}`;
      article.append(contextBlock(title, scallopAblationTable(ablation)));
    }

    if (benchmark.scallop_stream_injection_ablation) {
      article.append(contextBlock(
        "Scallop source-grounded stream injection",
        scallopInjectionTable(benchmark.scallop_stream_injection_ablation),
      ));
    }

    if (benchmark.realized_context_tiers.length) {
      const tiers = element("div", "context-tier-grid");
      benchmark.realized_context_tiers.forEach((tier) => {
        const card = element("article", "context-tier-card");
        card.append(element("h5", "", tier.name));
        const tokenRange = tier.token_min === null || tier.token_max === null
          ? "Not reported"
          : `${formatNumber(tier.token_min)} to ${formatNumber(tier.token_max)}`;
        [
          ["Realized tokens", tokenRange],
          ["Requested target", tier.requested_target_percent === null ? "Not reported" : `${formatNumber(tier.requested_target_percent)}%`],
          ["Realized target", tier.realized_target_percent === null ? "Not reported" : `${formatNumber(tier.realized_target_percent)}%`],
          ["Max position error", tier.max_position_error === null ? "Not reported" : formatNumber(tier.max_position_error)],
          ["Hard lexical count", tier.hard_lexical_count === null ? "Not reported" : formatNumber(tier.hard_lexical_count)],
        ].forEach(([label, value]) => {
          const metric = element("div", "context-tier-metric");
          metric.append(element("span", "", label));
          metric.append(element("strong", "", value));
          card.append(metric);
        });
        tiers.append(card);
      });
      article.append(contextBlock("Realized context tiers", tiers));
    }

    if (benchmark.windows.length) {
      article.append(contextBlock("Window truncation", contextScoreTable(benchmark.windows)));
    }

    if (benchmark.parameter_tiers.length) {
      const metadata = element("div", "parameter-tier-grid");
      benchmark.parameter_tiers.forEach((tier) => {
        const item = element("div", "parameter-tier");
        item.append(element("strong", "", tier.name));
        let range = "Range not reported";
        if (tier.min_billions !== null && tier.max_billions !== null) {
          range = `${formatNumber(tier.min_billions)} to ${formatNumber(tier.max_billions)} billion`;
        } else if (tier.min_billions !== null) {
          range = `${formatNumber(tier.min_billions)}+ billion`;
        }
        item.append(element("span", "", range));
        metadata.append(item);
      });
      article.append(contextBlock("Parameter tiers", metadata));
    }

    if (benchmark.memory_growth) {
      article.append(contextBlock("Memory growth", memoryGrowthGrid(benchmark.memory_growth)));
    }

    if (benchmark.breakdowns.length) {
      const breakdownList = element("div", "context-breakdowns");
      benchmark.breakdowns.forEach((breakdown) => {
        const details = element("details", "context-breakdown");
        details.append(element("summary", "", breakdown.source.replaceAll("_", " ")));
        breakdown.groups.forEach((group) => {
          const groupBlock = element("div", "context-breakdown-group");
          groupBlock.append(element("h5", "", group.name));
          groupBlock.append(contextScoreTable(group.scores));
          details.append(groupBlock);
        });
        breakdownList.append(details);
      });
      article.append(contextBlock("Detailed slices", breakdownList));
    }
    container.append(article);
  });
}

function renderRunCards() {
  const container = byId("run-cards");
  container.replaceChildren();
  filteredRuns().forEach((run) => {
    const card = element("article", "run-card");
    const top = element("div", "run-card-top");
    top.append(element("h3", "", run.name));
    top.append(statusBadge(run));
    card.append(top);
    card.append(element("code", "run-path", run.id));

    const metrics = element("div", "metric-list");
    run.score_metrics.slice(0, 5).forEach((metric) => {
      const line = element("div", "metric-line");
      line.append(element("code", "", metric.key));
      line.append(element("strong", "", formatNumber(metric.value)));
      metrics.append(line);
    });
    if (!run.score_metrics.length) metrics.append(element("p", "empty-state", "No score-like metric reported."));
    card.append(metrics);

    const footer = element("div", "run-card-footer");
    footer.append(artifactLinks(run));
    if (run.missing_artifacts.length) {
      footer.append(element("p", "missing-text", `Missing: ${run.missing_artifacts.join(", ")}`));
    }
    Object.entries(run.parse_errors).forEach(([name, message]) => {
      footer.append(element("p", "error-text", `${name}: ${message}`));
    });
    Object.entries(run.artifact_integrity_errors || {}).forEach(([name, message]) => {
      footer.append(element("p", "error-text", `${name}: ${message}`));
    });
    card.append(footer);
    container.append(card);
  });
}

function render(data) {
  state.data = data;
  byId("connection-label").textContent = "Live";
  byId("updated-at").textContent = `Updated ${formatDate(data.generated_at)}`;
  byId("results-root").textContent = data.results_root;
  document.querySelector(".live-state").classList.remove("offline");
  renderKpis(data.summary);
  renderCurrentState(data);
  renderStatusFilter(data.runs);
  renderCodeReferences(data.code_references);
  renderModelComparisons(data.runs);
  renderContextBenchmarks(data.runs);
  renderTable();
  renderRunCards();
}

async function refresh() {
  const button = byId("refresh-button");
  button.disabled = true;
  button.textContent = "Reading artifacts";
  try {
    const response = await fetch("/api/dashboard", { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard endpoint returned ${response.status}`);
    render(await response.json());
  } catch (error) {
    byId("connection-label").textContent = "Unavailable";
    byId("updated-at").textContent = error.message;
    document.querySelector(".live-state").classList.add("offline");
  } finally {
    button.disabled = false;
    button.textContent = "Refresh evidence";
  }
}

byId("refresh-button").addEventListener("click", refresh);
byId("run-search").addEventListener("input", () => {
  renderTable();
  renderRunCards();
});
byId("status-filter").addEventListener("change", () => {
  renderTable();
  renderRunCards();
});

refresh();
setInterval(refresh, 15000);
