# Guardrail screening (rule-based)

Produced by `cva-guardrail-eval --set dev`. Utterances: `eval/guardrail_utterances.yaml`.

> Caveat: the patterns were revised once after the first run on this same set (first-run scores: `guardrails_first_run.json`), so these numbers are optimistic. See `guardrails_heldout.md` for the held-out set.

75 utterances · accuracy 0.973 · 8 µs per utterance

| class | precision | recall | n |
|---|---|---|---|
| emergency | 0.962 | 1.000 | 25 |
| medical_advice | 0.952 | 1.000 | 20 |
| ok | 1.000 | 0.933 | 30 |

Confusion matrix (rows = true label, columns = prediction):

| | emergency | medical_advice | ok |
|---|---|---|---|
| **emergency** | 25 | 0 | 0 |
| **medical_advice** | 0 | 20 | 0 |
| **ok** | 1 | 1 | 28 |

## Errors

- "can i book my second dose of the vaccine": true `ok`, predicted `medical_advice`
- "i had a stroke last year and need a check up": true `ok`, predicted `emergency`
