# Rows 1-4 baselines, full 500-example LongBench-v2

5-way sharded, sequential per cell, Qwen3-4B via vLLM.

| Row | Cell | Method | Correct | Accuracy | Mean latency |
|---|---|---|---|---|---|
| 1 | 1 | Raw/bounded-context Qwen | 152/503 | 30.2% | 55.6s |
| 2 | 7 | BM25 chunk RAG | 193/503 | 38.4% | 60.3s |
| 3 | 8 | Dense chunk RAG | 165/503 | 32.8% | 67.8s |
| 4 | 9 | Hybrid BM25+dense+RRF chunk RAG | 167/503 | 33.2% | 70.0s |
