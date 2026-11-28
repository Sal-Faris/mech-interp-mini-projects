# Mini-Project 1: Where Does GPT-2 Small "Know" Country Capitals?

- **Date:** 2026-11-28
- **Model:** `gpt2` (GPT-2 Small, via TransformerLens 4.0 `TransformerBridge`)

## 1. Question
Where in GPT-2 Small does country-to-capital retrieval happen, and is it driven primarily by MLPs or attention heads?

## 2. Step 1: Dataset & Prompt Findings
- **Zero-shot failure:** `"The capital of {country} is"` got 0/29 at Top-1 and Top-5 (only 2/29 at Top-10) because base GPT-2 predicts sentence continuations (`" the"`, `" located"`, `" a"`).
- **2-shot prompt:** `"The capital of Japan is Tokyo. The capital of Germany is Berlin. The capital of {country} is"` (excluding Japan & Germany from test set).
- **Accuracy:** 29/33 pairs had single-token capitals; **25/29 (86.2%)** were strict Top-1 (`k=1`).
- **Notable failure pattern (4/4):** Every mistake was the country's largest / most famous non-capital city:
  - China -> `' Shanghai'` (not Beijing)
  - Turkey -> `' Istanbul'` (not Ankara)
  - Canada -> `' Toronto'` (not Ottawa)
  - Australia -> `' Sydney'` (not Canberra)

## 3. Step 2 & 3: Logit Lens & DLA
- *(pending)*

## 4. Step 4: Red-Teaming
- Compare **zero-shot vs. 2-shot** DLA (using `logit(correct) - mean(logit(other capitals))`) to separate pure factual recall from in-context formatting heads.
- Test prompt rephrasings and check specific-capital vs. generic-city boost.
