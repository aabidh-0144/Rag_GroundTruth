"""Synthetic evaluation datasets, so a golden dataset never has to be hand-written.

  generator.py  Ragas TestsetGenerator over our own chunks -> DatasetSamples
  spans.py      anchor each reference context to (doc_id, start_char, end_char)
  datasets.py   datasets/<id>/ storage, human review (verify / reject / edit),
                and joining a dataset with a RAG run for evaluation

Every generated sample starts as origin="synthetic", verification_status=
"unverified". Only a human review changes that.
"""
