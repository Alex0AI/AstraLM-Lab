# Results ledger

This page records positive, negative, and inconclusive results under the same
standard. It is deliberately separate from the project pitch.

## Pico public-text smoke run — inconclusive

Run: [GitHub Actions 32290940854](https://github.com/Alex0AI/AstraLM-Lab/actions/runs/32290940854)  
Commit: `9d62a6c8d6105246ff51c5fa9780b5cc0d53cc15`  
Corpus: 750,000 bytes of Tiny Shakespeare pinned to `karpathy/char-rnn@6f9487a`  
Corpus SHA-256: `ebeb1464910873540a8608640cb2cbfcc638d18e543a781dfda78c378424a25a`

| Mode | Parameters | Seeds | Mean best validation loss | Std. dev. |
|---|---:|---:|---:|---:|
| Standard | 110,914 | 3 | 4.816673 | 0.053550 |
| Attention Bridge | 110,914 | 3 | 4.817367 | 0.054357 |

Attention Bridge minus standard is `+0.000694` loss (lower is better), only about
0.014% of the baseline mean and far smaller than seed-to-seed variation. This
run therefore provides **no evidence of an improvement or meaningful regression**.
It validates the experimental pipeline, not the architecture hypothesis.

Protocol: 40 steps per run, batch 8, sequence length 64, 20,480 training tokens
per seed, seeds 17/42/73, pico preset, CPU GitHub-hosted runner. The complete
checkpoints, JSONL metrics, JSON/Markdown/HTML reports, and provenance metadata
are retained as the run artifact for its configured retention period.

## Promotion rule

A result should move from smoke evidence to a headline table only after a fixed
token budget, more seeds, paired uncertainty, recorded environment, and at least
one second corpus. A larger run should replace uncertainty with evidence—not
replace honest wording with stronger marketing.
