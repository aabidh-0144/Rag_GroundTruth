"""Mode 2 - Ragas-based evaluation.

  metrics.py    our metric names -> Ragas classes, and which need a ground truth
  adapters.py   our Ollama/Groq judge + local embeddings in the shapes Ragas expects
  evaluator.py  RagasEvaluator, plugged into orchestrator.run_evaluation

Reference-free: faithfulness, response_relevancy, context_precision
Reference-based (need ground_truth): context_precision_with_reference,
  context_recall, answer_correctness, answer_similarity

Importing this package does not import Ragas; that happens only when an evaluator
actually builds metrics.
"""
