# Error analysis — where the best configs still fail

A systematic look at what our top NewsBench configurations miss (drop) or add
(spurious / duplicated / gold-omitted), token by token, on the 15 scored pages
(~54k gold tokens). Produced by aligning each model's page output against the
volunteer gold with a token-level diff, then classifying every contiguous span.

## Error budget by type (tokens, n = 15)

| model (DocLayout harness) | overall | recognition | true-miss | duplication | hallucination | reorder |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.6-luna | 0.975 | 4,701 | 146 | 247 | 210 | 19 |
| mistral-small-2603 | 0.958 | 3,810 | 115 | 360 | 267 | 384 |
| deepseek-v4.1-flash | 0.957 | 4,827 | 154 | 497 | 419 | 193 |

- **recognition** — character-level misreads within a correctly matched region.
- **true-miss** — gold text the model dropped entirely.
- **duplication** — text emitted more than once (present in gold, over-counted).
- **hallucination** — output text with no match in gold.
- **reorder** — gold text present but in a different position.

## Findings

**1. Recognition error dominates (~7–9% of tokens).** The real accuracy ceiling is
reading degraded 19th-century type, not layout. Everything else is a rounding error
by comparison.

**2. "Misses" are mostly the *detector*, not the recognizer.** The same spans are
dropped across models that share the DocLayout-YOLO detector — a poem stanza on
`mss3413201628-41` (confirmed present in gold), a signature line, union-label and
small-ad blocks, some mastheads. Only ~0.3% of tokens, but it's a fixed set of
regions the detector never proposes. **Improving detection (or the residual pass)
is the lever to push past 0.97.** A secondary, recognizer-specific miss source
exists too: GLM-OCR, the most conservative reader, additionally drops faint regions
it declines to transcribe (25 missed spans vs gpt-5.6-luna's 12).

**3. Most "added" text is real — the gold is incomplete.** On gpt-5.6-luna, 15 of
20 added spans (≥6 tokens) were located on the scan by an independent OCR pass —
i.e. the model transcribed real text the volunteer gold omitted (page numbers,
datelines, ad lines). These are gold gaps, not model errors, and they slightly
*understate* the model's true score.

**4. Genuine hallucinations cluster on a few degraded regions**, plus VLM refusal
boilerplate leaking into the text (e.g. *"I'm sorry, but I can't read the text in
this image."*, *"The image has been rotated 180 degrees."*). This is
model-dependent: deepseek is the worst offender (419 hallucination tokens vs
gpt-5.6-luna's 210), largely from verbose refusal strings. A short refusal-phrase
filter in the recognizers would recover most of that precision.

## Two error personalities

Reached from opposite directions to nearly the same score:

- **GLM-OCR (0.970)** — *conservative*: misses more (25 spans), adds less (10). When
  a region is faint it drops it rather than guess. Fewer fabrications.
- **gpt-5.6-luna (0.975)** — *completionist*: misses less (12), adds more (20,
  mostly real gold-gap text). Reads everything, at the cost of the occasional
  degraded-region hallucination.

For a historian this is a real editorial choice: silence vs completeness.

## Visual review

Every missed/added span, cropped from the scan for eyeball judgment (clear error
vs. reasonable):

- gpt-5.6-luna + DocLayout: https://claude.ai/code/artifact/6cf6a9e9-afdd-43e3-9b06-1875baa7293d
- GLM-OCR + DocLayout: https://claude.ai/code/artifact/0194a624-a37c-45ad-acbf-110cf2b1062a
