"""GroundTruth evaluation (Phase 3) - start reading here.

Two modes behind one orchestrator and one result schema:

  Mode 1  reference_free/  LLM-as-judge; needs only question + contexts + answer
  Mode 2  ragas_mode/      Ragas metrics, reference-free and reference-based
          synthetic/       Ragas synthetic dataset generation + human review

File map - each file has one job:

  config.py        what an evaluation is configured to do (configs/evaluation.yaml)
  schemas.py       data shapes: EvalSample, MetricResult, SampleResult, EvaluationResult
  inputs.py        where samples come from: a JSON/JSONL file, or a RAG run directory
  prompts.py       every Mode 1 judge prompt, and nothing else
  judge.py         ask the judge LLM, get validated JSON back (one repair retry)
  aggregate.py     how scores are combined - the honest-scoring rules
  store.py         evals/<id>/ persistence and history
  orchestrator.py  runs an evaluation end to end, for either mode

  reference_free/  base.py (Metric contract), one file per metric, evaluator.py
  ragas_mode/      metrics.py (name -> Ragas class), adapters.py, evaluator.py
  synthetic/       generator.py, spans.py, datasets.py

Entry points: `groundtruth.cli eval ...` and the Streamlit app (`groundtruth.cli ui`).
Ragas (and through it LangChain) is imported only inside ragas_mode/ and synthetic/,
so Mode 1 and the core RAG system never load it.
"""
