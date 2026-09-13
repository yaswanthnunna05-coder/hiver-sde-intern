# Hiver AI Support Agent — AppleSupport

An AI support-agent prototype for the Hiver SDE Intern take-home. Given an incoming customer message, the system predicts a support intent, retrieves similar historical AppleSupport conversations, drafts a reply grounded in those examples, and decides whether to auto-handle or escalate.

## Problem framing

The goal is not to build a fully autonomous customer-service system. The prototype tests whether historical support behavior can provide useful evidence for intent classification, reply drafting, and a conservative first-pass automation decision.

**Brand:** AppleSupport  
**Historical conversations:** 106,646 reconstructed customer→AppleSupport pairs  
**Golden evaluation set:** 150 examples

## Architecture

```text
Customer message
      |
      v
Intent classifier (TF-IDF + Logistic Regression)
      |
      +----------------------+
      |                      |
      v                      v
Historical retrieval     Intent-aware reply generation
(TF-IDF cosine)              |
      |                      |
      +----------+-----------+
                 v
       Auto-handle / Escalate
                 |
                 v
       Top historical evidence
```

## Intent taxonomy

Nine compact intents were used:

1. `ios_update` — iOS update/upgrade/downgrade/update problems
2. `device_issue` — iPhone/iPad/Mac hardware or general device behavior
3. `app_issue` — applications crashing, freezing, or malfunctioning
4. `battery_charging` — battery drain, health, or charging
5. `account_icloud` — Apple ID, iCloud, password, login, verification
6. `connectivity` — Wi-Fi, Bluetooth, internet, or connection problems
7. `music_media` — Apple Music, iTunes, music, playlists, songs
8. `order_purchase` — purchases, orders, delivery, shipping, refunds
9. `other_support` — general, unclear, or miscellaneous support

The main problem was labelled rather than every keyword. For example, an iOS update causing battery drain is treated as `battery_charging`.

## Data preparation

The project uses the Kaggle **Customer Support on Twitter** dataset from Thought Vector. Only AppleSupport company responses and their corresponding inbound customer tweets were reconstructed into customer→support pairs.

The original dataset is large; the project works on the AppleSupport slice rather than requiring the entire dataset during normal execution.

## Evaluation integrity

The 150-example golden set is excluded from both classifier training and historical retrieval during evaluation. This prevents the evaluated customer messages from being returned as their own evidence.

The golden set was human-reviewed for evidence support. 134 of 150 examples were judged to have supported historical evidence (89.33%). This is an evidence-support metric, **not** overall reply correctness.

## Results

### Intent classification

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 48.00% | 0.0721 |
| TF-IDF retrieval + weak-rule labels | 34.67% | 0.3368 |
| Final V3 classifier | **60.00%** | **0.5307** |

The final classifier improves accuracy by 12 percentage points over the majority baseline and 25.33 percentage points over the retrieval baseline.

### End-to-end reply pipeline

| Metric | Result |
|---|---:|
| Intent accuracy | 60.00% |
| Non-empty replies | 100.00% |
| Evidence retrieved | 100.00% |
| Auto-handle rate | 24.00% |
| Escalation rate | 76.00% |
| Average top similarity | 0.396 |
| Human evidence support | 89.33% (134/150) |

### LLM-as-judge status

An LLM-as-judge harness was implemented using the OpenAI Responses API with relevance, helpfulness, grounding, safety, and overall scores. A 5-example calibration completed successfully and produced meaningful scores.

A larger 40-example run was attempted but could not be completed because the organization-level TPM rate limit was exhausted. **No 40-example aggregate LLM score is reported.** This is intentionally disclosed rather than replacing the missing score with an estimate.

## What is misleading about my headline number?

The 60% intent accuracy is useful but incomplete. It is measured on a small 150-example golden set and does not mean that 60% of real customer cases would be safely resolved end-to-end. Reply quality, retrieval relevance, escalation safety, and distribution shift are separate concerns. In particular, a high retrieval similarity does not guarantee that the retrieved support response is a safe resolution.

## Top failure modes

1. **Device issues predicted as `other_support` (22 cases):** short/noisy device complaints often lack explicit device keywords.
2. **Update problems predicted as `device_issue` (5 cases):** words such as “iPhone” can dominate the actual update intent.
3. **App issues predicted as `other_support` (3 cases):** indirect application complaints may not contain explicit app/crash wording.
4. **App issues predicted as `device_issue` (3 cases):** platform words such as Mac/iPhone can dominate app-level symptoms.
5. **App issues predicted as `ios_update` (3 cases):** “update” is ambiguous between updating software and updating applications.

A related end-to-end weakness is that some customer messages are acknowledgements rather than support requests. The current taxonomy assumes the incoming message is a support case.

## What was not built

- No production deployment or real customer-data integration
- No CRM/ticketing system integration
- No authentication or account-level actions
- No transactional purchase/refund execution
- No autonomous high-risk actions
- No claim of production-grade safety

## Decision log

See [`decision_log.md`](decision_log.md) for the non-obvious implementation and evaluation decisions.

## Reproduce locally

Create/activate a Python virtual environment and install the dependencies in `requirements.txt`.

The interactive agent can be run with:

```powershell
.\.venv\Scripts\python.exe src\pipeline.py
```

Enter a customer message when prompted. The pipeline prints the predicted intent, draft reply, escalation decision, reason, and top historical evidence.

To reproduce the main evaluation:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_replies.py
.\.venv\Scripts\python.exe evaluation\evaluate_baselines.py
.\.venv\Scripts\python.exe evaluation\analyze_failures.py
```

## Example

```text
Customer message: My iPhone battery is draining very quickly after the latest update

Intent: battery_charging

Decision: AUTO-HANDLE

Top evidence includes historical AppleSupport cases about rapidly draining batteries.
```

The example also illustrates a known limitation: the current escalation policy uses retrieval similarity as a signal, so a strong match can still produce an overly aggressive auto-handle decision when historical responses mainly request a DM or additional information.

## Dataset attribution

Dataset: Thought Vector, **Customer Support on Twitter**, Kaggle. The dataset is licensed CC BY-NC-SA 4.0 according to the Kaggle dataset page. The dataset should be used in accordance with its license and attribution requirements.
