Experiment configs.

All three differ ONLY in the `retrieval` block. `chunking` and `embedding` are
identical across every file on purpose: they determine the index, and all experiments
must share one index for the comparison to isolate a single variable (invariant #3).

If you change `chunking` or `embedding`, you must change it in ALL configs and re-run
`chunk` + `index`; the pipeline refuses to run against a mismatched index.
