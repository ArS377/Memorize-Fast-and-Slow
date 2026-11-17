# LongMemEval-S Shared-Facts Comparison, All 500 Histories - November 17, 2026

### Results
- Added `results/longmemeval_s_shared_facts_500_20261117/`: NeuroSym versus Graphiti on all 500 LongMemEval-S histories with shared extracted facts (50.2% versus 48.2%; +2.0 points, 95% CI [-1.4, +5.6]; no difference detected), with a per-history label table, a per-history group table for the sensitivity analyses, and a standard-library analysis script.
- Kept `results/longmemeval_s_shared_facts_250_20261114/` as the earlier 250-history snapshot; its per-history labels are included unchanged in the 500-history table.
- Linked the result from `README.md` and `docs/recent-results.md`.

# LongMemEval-S Shared-Facts Comparison - November 14, 2026

### Results
- Added `results/longmemeval_s_shared_facts_250_20261114/`: NeuroSym versus Graphiti on 250 LongMemEval-S histories with shared extracted facts (53.6% versus 52.4%; +1.2 points, 95% CI [-3.2, +5.6]; no difference detected), with a per-history label table and a standard-library analysis script.
- Linked the result from `README.md` and `docs/recent-results.md`.

# Project Consolidation - April 17, 2026

## Latest Changes

### README Rewrite (v2)
- ✅ Repositioned as foundation for Scallop-based symbolic constraints
- ✅ Added thesis statement and research positioning
- ✅ Condensed to essential quick start information
- ✅ Added architecture roadmap (Current → Next → Future)
- ✅ Added comparison table vs. related work (RAG, RLMs, KG-RAG)
- ✅ Removed verbose sections (troubleshooting, citations, acknowledgments)

## Initial Changes

### File Organization
- ✅ Created `tests/` directory for organized test suite
- ✅ Moved all test files to `tests/` folder
- ✅ Renamed `TEST_REPORT.md` → `TESTING.md`
- ✅ Updated test imports to reference parent directory

### Documentation
- ✅ Created comprehensive `README.md` with:
  - Step-by-step instructions for downloading data
  - Step-by-step instructions for running pipeline
  - Output format documentation
  - Troubleshooting guide
  - Future work section (Neo4j integration)
- ✅ Created `.gitignore` to exclude data files and outputs

### Cleanup
- ✅ Deleted temporary output files (`smoke_output.jsonl`, `verified_facts_test.jsonl`)
- ✅ Organized project structure for production use

### Testing
- ✅ All tests pass with new structure:
  - `tests/test_pipeline_smoke.py` - PASSED
  - `tests/test_validation.py` - PASSED (5/5 tests)
  - `tests/test_neo4j_format.py` - PASSED (3/3 tests)

## Final Project Structure

```
NeuroSym/
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
├── TESTING.md                   # Test documentation
├── CHANGELOG.md                 # This file
├── requirements.txt             # Python dependencies
├── longbench_kg_pipeline.py    # Main extraction pipeline
├── download_longbench.py       # Data download utility
└── tests/                       # Test suite
    ├── test_pipeline_smoke.py  # End-to-end smoke test
    ├── test_validation.py      # Logic validation tests
    └── test_neo4j_format.py    # Neo4j formatting tests
```

## Ready for Repository Push

The project is now organized and ready to be pushed to a Git repository with:
- Clear documentation for users
- Organized test suite
- Proper gitignore configuration
- Production-ready structure
