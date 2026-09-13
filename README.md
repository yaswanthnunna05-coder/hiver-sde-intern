# AI Support Agent — Hiver SDE Intern Take-Home

An AI support agent prototype that classifies customer support requests, retrieves similar historical support interactions, drafts a response grounded in historical AppleSupport behavior, and decides whether the request should be auto-handled or escalated to a human.

> **Headline result:** 61.33% intent classification accuracy on a 150-example human-verified golden set.

---

## 1. Problem Framing

The goal is to build a lightweight AI support agent that can assist a customer-support team with three decisions:

1. What is the customer's main intent?
2. What response should be drafted based on how similar issues were historically handled?
3. Should the request be auto-handled or escalated to a human?

For this prototype, the system focuses on the `AppleSupport` brand from the Kaggle Customer Support on Twitter dataset.

The system is intentionally designed as a support-assistance prototype rather than a fully autonomous customer-service system.

---

## 2. What the System Does

Given a customer message:

    My iPhone battery is draining very quickly after the latest update

the pipeline performs:

    Customer Message
           |
           v
    Intent Classification
           |
           v
    Historical Similarity Search
           |
           v
    Draft Response Generation
           |
           v
    Escalation Decision
           |
           v
    Support Recommendation

Example output:

    Intent: battery_charging

    Draft Reply:
    Sorry you're having trouble with this. We'd be happy to look into it
    further. Please send us a DM so we can get more details and help you.

    Decision: AUTO-HANDLE

    Reason:
    Strong historical evidence was found.

---

## 3. Architecture

The system consists of four main components.

### Intent Classifier

A TF-IDF + Logistic Regression classifier predicts one of nine support intents.

### Historical Retriever

A TF-IDF similarity index searches historical AppleSupport customer messages and retrieves similar customer/support pairs.

### Reply Generator

The response generator uses retrieved historical support behavior to construct a short draft reply.

### Escalation Policy

The system considers:

- predicted intent
- historical similarity
- account-specific issues
- purchase/order issues

to determine whether the request should be auto-handled or escalated.

---

## 4. Intent Taxonomy

The prototype uses nine intents.

| Intent | Description |
|---|---|
| `ios_update` | iOS/software update, upgrade, downgrade, or update failures |
| `device_issue` | General iPhone, iPad, Mac, hardware, or device behavior |
| `app_issue` | Applications crashing, freezing, failing, or malfunctioning |
| `battery_charging` | Battery drain, battery health, charging, or charger problems |
| `account_icloud` | Apple ID, iCloud, passwords, login, and verification |
| `connectivity` | Wi-Fi, Bluetooth, internet, network, and connection problems |
| `music_media` | Apple Music, iTunes, songs, playlists, and media |
| `order_purchase` | Purchases, orders, delivery, refunds, and shipping |
| `other_support` | General, unclear, or miscellaneous support requests |

### Labeling Principle

The classifier attempts to identify the customer's main problem rather than simply matching every keyword.

For example:

    "iOS update made my battery drain"

is classified as:

    battery_charging

rather than:

    ios_update

because the customer's actual problem is battery drain.

---

## 5. Dataset

The project uses the:

**Customer Support on Twitter (TWCS)** dataset.

Dataset source:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The dataset contains customer-support conversations involving multiple brands.

Relevant fields include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The project filters the dataset to AppleSupport responses and reconstructs customer → support interactions.

### Dataset Preparation

The original dataset contains more than 3 million tweets/replies.

The project extracts:

    AppleSupport Responses
            |
            v
    Find Corresponding Customer Tweet
            |
            v
    Customer → AppleSupport Pair

Result:

    AppleSupport historical conversations: 106,646

The reconstructed conversations are stored locally as:

    data/apple_conversations.csv

Raw and derived datasets are excluded from Git using `.gitignore`.

### Dataset Attribution

Customer Support on Twitter is provided by Thought Vector.

The dataset is licensed under:

**CC BY-NC-SA 4.0**

The dataset should therefore be used in accordance with its license and attribution requirements.

The complete raw dataset is intentionally not included in this repository.

---

## 6. Golden Evaluation Set

The evaluation set contains:

**150 unique customer messages**

The examples were manually reviewed and assigned one of the nine predefined intent labels.

### Sampling

The golden set was constructed in two stages:

- 100 examples were sampled from the AppleSupport reconstructed conversation set using random seed `42`.
- 50 additional examples were sampled from the full dataset while excluding the first 100 examples.

The final 150 examples were manually reviewed.

### Human Labeling

The intent labels were manually evaluated for all 150 examples.

Final intent distribution:

| Intent | Examples |
|---|---:|
| `device_issue` | 73 |
| `battery_charging` | 16 |
| `other_support` | 14 |
| `ios_update` | 13 |
| `app_issue` | 12 |
| `music_media` | 9 |
| `connectivity` | 6 |
| `order_purchase` | 4 |
| `account_icloud` | 3 |
| **Total** | **150** |

### Leakage Prevention

All 150 golden examples are excluded from:

1. classifier training
2. historical retrieval evaluation

This prevents the evaluation messages from appearing directly in the training or retrieval corpus.

---

## 7. Classification Model

The final classifier uses:

    TF-IDF
        +
    Logistic Regression

TF-IDF configuration includes:

- unigram + bigram features
- minimum document frequency filtering
- sublinear TF scaling
- up to 50,000 features
- balanced class weighting

The classifier is trained using automatically assigned historical labels generated from the project's intent rules.

The 150 human-verified golden examples are excluded from classifier training.

---

## 8. Model Selection

Several rule-based labeling strategies were tested.

### V1

Initial keyword rules produced:

    Accuracy: 31.33%

The major problem was that general update/device keywords caused many device issues to be mislabeled as app or update issues.

### V2

A more contextual rule set was tested but caused the `ios_update` class to collapse to a very small training class.

This version was rejected.

### V3

The final selected rule set introduced:

- intent-specific keyword groups
- contextual update detection
- explicit app detection
- device fallback handling
- battery/charging detection
- connectivity detection
- account/order detection
- music/media detection

Final V3 performance:

    Accuracy: 61.33%
    Macro F1: 0.5531

### V4

An additional targeted rule set was tested.

It performed worse:

    Accuracy: 38.00%
    Macro F1: 0.4674

Therefore V3 was retained.

V4 was not used for the final system.

---

## 9. Historical Retrieval

The system uses TF-IDF cosine similarity to retrieve historical customer-support interactions.

For every incoming message:

    Customer Message
          |
          v
    TF-IDF Vector
          |
          v
    Cosine Similarity Against Historical Messages
          |
          v
    Top-k Historical Conversations

Each retrieved example contains:

- customer message
- historical AppleSupport response
- similarity score

The top historical interactions are used as evidence for drafting the response.

---

## 10. Reply Generation

The reply generator is intentionally conservative.

Rather than attempting to generate arbitrary troubleshooting instructions, it looks at historical AppleSupport responses and extracts common support behaviors.

Examples of historical behaviors include:

- asking the customer to send a DM
- suggesting an update
- suggesting a restart
- recommending further Apple Support investigation

This reduces the risk of generating unsupported troubleshooting instructions.

---

## 11. Escalation Policy

The prototype uses a simple rule-based escalation policy.

### Always Escalate

#### Account/iCloud Issues

    account_icloud

These may require account-specific verification.

#### Purchase/Order Issues

    order_purchase

These may require transaction-specific information.

### Weak Historical Evidence

If the best historical similarity is below the configured threshold, the system escalates.

### Strong Historical Evidence

High similarity can allow:

    AUTO-HANDLE

Moderate similarity results in:

    ESCALATE

The policy is intentionally conservative for account and transaction-related requests.

---

## 12. Evaluation

Evaluation was performed on the 150-example golden set.

### Final Results

| Metric | Result |
|---|---:|
| Intent accuracy | **61.33%** |
| Macro F1 | **0.5531** |
| Non-empty replies | **100%** |
| Evidence found | **100%** |
| Auto-handle rate | **24%** |
| Escalation rate | **76%** |
| Average top similarity | **0.396** |

---

## 13. Baseline Comparison

Two simple baselines were evaluated.

### Majority-Class Baseline

The majority-class baseline predicts:

    device_issue

for every example.

Result:

    Accuracy: 48.67%
    Macro F1: 0.0727

### TF-IDF Retrieval Baseline

The retrieval baseline predicts the intent associated with the most similar historical customer message.

Result:

    Accuracy: 34.00%
    Macro F1: 0.3285
    Average similarity: 0.400

### Comparison

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 48.67% | 0.0727 |
| TF-IDF retrieval | 34.00% | 0.3285 |
| **V3 classifier** | **61.33%** | **0.5531** |

The V3 classifier improves accuracy by:

**+12.66 percentage points**

over the majority baseline.

It improves accuracy by:

**+27.33 percentage points**

over the TF-IDF retrieval baseline.

---

## 14. Human Evidence Evaluation

The retrieved historical examples were separately reviewed for whether they provided meaningful support for the generated response.

Final result:

    134 / 150 supported

or:

    89.33%

This metric measures **historical evidence support**, not overall reply correctness.

A response can therefore have supporting historical evidence while still being incomplete, vague, or inappropriate for the customer's exact situation.

---

## 15. LLM-as-Judge

An LLM-as-judge evaluation harness was implemented to evaluate draft replies on:

- relevance
- helpfulness
- grounding
- safety
- overall quality

A five-example calibration run completed successfully.

However, larger evaluation runs were blocked by API rate/token quota limits.

Therefore:

> **No aggregate LLM-judge score is reported.**

The project does not fabricate or extrapolate a full-set LLM evaluation score from the small calibration run.

The successful calibration file is:

    evaluation/llm_judge_calibration_5.csv

---

## 16. Failure Analysis

The final classifier made:

    58 / 150

incorrect predictions.

Error rate:

    38.67%

### 1. Device Issues Classified as Other Support

This was the largest error category.

Short or noisy device complaints sometimes lacked explicit device keywords.

Example:

    Hey @AppleSupport is it normal for the blue text to overlap like this?

The message describes a device/UI problem but does not contain an obvious device keyword.

#### Hypothesis

The rule-based labeling strategy is too dependent on explicit device vocabulary.

### 2. Update Problems Classified as Device Issues

Example:

    MY IPHONE IS FUCKED OI @AppleSupport SORT OUT YOUR UPDATE

The presence of `iPhone` can dominate the update signal.

#### Hypothesis

The classifier needs stronger semantic separation between device + update complaints and general device problems without relying only on keyword ordering.

### 3. App Issues Classified as Device Issues

Example:

    problems with their Mac apps since HighSierra update

The word `Mac` strongly indicates a device, even though the actual issue is application-related.

#### Hypothesis

Platform terms such as `Mac` and `iPhone` are too strong compared with indirect application signals.

### 4. App Issues Classified as Other Support

Some app problems are expressed indirectly without words such as:

    crash
    freeze
    app

This makes them difficult for the rule-based weak-labeling system to identify.

### 5. Ambiguous Update/App/Device Complaints

Messages containing words such as:

    update
    iPhone
    app
    iOS

can describe multiple related problems.

The current rule-based taxonomy has limited semantic understanding of which problem is primary.

---

## 17. What Is Misleading About My Headline Number?

The headline number is:

    61.33% intent accuracy

This number should **not** be interpreted as 61.33% end-to-end customer-support correctness.

Important limitations:

1. The golden set contains only 150 examples.
2. The dataset is historical Twitter support data rather than a production support queue.
3. The intent taxonomy was designed specifically for this prototype.
4. Historical support replies are not necessarily final resolutions.
5. The reply generator is intentionally simple.
6. The escalation policy uses simple similarity thresholds.
7. The LLM-judge evaluation could not be completed at full scale because of API quota limits.
8. The golden-set class distribution is imbalanced, with `device_issue` representing a large portion of the evaluation set.
9. Evidence support measures whether retrieved historical examples support the draft, not whether the draft completely solves the customer's problem.

Therefore, the 61.33% number is best interpreted as:

> **Performance of the final intent-classification component on a small, human-verified evaluation sample.**

It is not a production automation accuracy estimate.

---

## 18. What Was Not Built

The prototype deliberately does not attempt to implement:

- production customer authentication
- access to customer accounts
- real Apple account operations
- order lookup
- refund processing
- actual Twitter/X integration
- production database infrastructure
- human-agent workflow integration
- real customer conversation state management
- production monitoring
- sophisticated conversation memory
- fine-tuned transformer models
- production-grade LLM response generation

These would be required for a production support system.

---

## 19. Evaluation Integrity

The evaluation pipeline was designed to avoid obvious data leakage.

### Golden-Set Exclusion

All 150 golden examples are excluded from:

    classifier training

and:

    historical retrieval

during evaluation.

### Reproducibility

Evaluation artifacts are stored under:

    evaluation/

including:

    reply_evaluation_results.csv
    retrieval_baseline_results.csv
    human_evidence_annotation.csv
    llm_judge_calibration_5.csv

The complete raw dataset and trained model are excluded from Git.

---

## 20. Repository Structure

    hiver-sde-intern/
    │
    ├── data/
    │   ├── brand_counts.csv
    │   ├── golden_candidates_50_labeled.csv
    │   ├── golden_set.csv
    │   ├── initial_intent_analysis.csv
    │   └── labeling_candidates.csv
    │
    ├── evaluation/
    │   ├── analyze_failures.py
    │   ├── confusion_matrix.csv
    │   ├── create_human_annotation.py
    │   ├── evaluate_baselines.py
    │   ├── evaluate_confusion.py
    │   ├── evaluate_cv.py
    │   ├── evaluate_intent.py
    │   ├── evaluate_replies.py
    │   ├── evaluate_retrieval_baseline.py
    │   ├── human_evidence_annotation.csv
    │   ├── llm_judge.py
    │   ├── llm_judge_calibration_5.csv
    │   ├── reply_evaluation_results.csv
    │   └── retrieval_baseline_results.csv
    │
    ├── src/
    │   ├── escalation.py
    │   ├── generate_reply.py
    │   ├── pipeline.py
    │   ├── retrieval.py
    │   └── train_classifier.py
    │
    ├── data_inspection.py
    ├── decision_log.md
    ├── extract_apple.py
    ├── intent_discovery.py
    ├── reconstruct_conversations.py
    ├── requirements.txt
    ├── sample_for_labeling.py
    └── README.md

---

## 21. Reproduction

### Requirements

Python 3.10+ is recommended.

Install dependencies:

    pip install -r requirements.txt

### Train the Classifier

    python src/train_classifier.py

This creates:

    models/intent_classifier.pkl

The model file is ignored by Git.

### Run the Support Pipeline

    python src/pipeline.py

Enter a customer message when prompted.

Example:

    My iPhone battery is draining very quickly after the latest update

The pipeline returns:

    Intent
    Draft Reply
    Decision
    Reason
    Historical Evidence

---

## 22. Run Evaluation

Run the main reply evaluation:

    python evaluation/evaluate_replies.py

Run the majority/retrieval/classifier baseline comparison:

    python evaluation/evaluate_baselines.py

Run the TF-IDF retrieval baseline:

    python evaluation/evaluate_retrieval_baseline.py

Run failure analysis:

    python evaluation/analyze_failures.py

---

## 23. Reproducing the Headline Result

The main headline result is produced by evaluating the final classifier against the 150-example golden set.

Expected result:

    Intent accuracy: approximately 61.33%
    Macro F1: approximately 0.5531

Small differences may occur if dependencies or preprocessing behavior change.

---

## 24. Decision Log

Non-obvious project decisions are documented separately in:

    decision_log.md

Key decisions include:

- selecting AppleSupport
- reconstructing customer/support pairs
- using nine intents
- merging notification-related cases into device issues
- labeling the main problem rather than every keyword
- using TF-IDF + Logistic Regression
- selecting V3 over V4
- excluding golden examples from training
- excluding golden examples from retrieval
- retrieving multiple historical examples
- grounding replies in historical support behavior
- escalating account/iCloud requests
- escalating purchase/order requests
- using similarity as an escalation signal
- evaluating against simple baselines

---

## 25. Next Week

If this prototype were developed further, the highest-value improvements would be:

### 1. Improve Intent Labeling

Replace keyword-based weak labeling with a higher-quality annotation strategy and possibly a small manually labeled training set.

### 2. Improve Semantic Retrieval

Compare TF-IDF with embedding-based retrieval.

Potential architecture:

    Customer Message
           |
           v
    Embedding Model
           |
           v
    Vector Database
           |
           v
    Top Historical Resolutions

### 3. Improve Reply Generation

Use an LLM constrained by retrieved evidence rather than relying mainly on fixed response templates.

### 4. Improve Escalation

Tune thresholds using a larger validation set and explicitly detect cases where historical replies request additional information rather than providing a resolution.

### 5. Expand Evaluation

Create a larger stratified golden set and complete the LLM-as-judge evaluation with a properly budgeted API setup.

### 6. Measure End-to-End Quality

Future evaluation should separately measure:

- intent correctness
- evidence relevance
- reply correctness
- reply helpfulness
- safety
- escalation precision
- escalation recall
- automation success rate

---

## 26. Limitations

This project is a prototype.

The biggest limitations are:

- small evaluation set
- noisy historical Twitter conversations
- weak-labeling dependence during classifier training
- imbalanced intent distribution
- simple TF-IDF retrieval
- simple template-based response generation
- heuristic escalation thresholds
- incomplete LLM-as-judge evaluation
- historical support replies may be follow-up questions rather than final resolutions

The system should therefore be treated as a **decision-support prototype**, not a production autonomous support agent.

---

## 27. License and Dataset Notice

The code in this repository is provided for the purposes of the take-home assignment and experimentation.

The Customer Support on Twitter dataset is subject to its own license and attribution requirements.

Dataset:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The raw dataset is not redistributed in this repository.

---

## 28. Summary

This project implements a lightweight AI customer-support agent with:

    Intent Classification
            +
    Historical Retrieval
            +
    Evidence-Grounded Reply Drafting
            +
    Escalation Decision

Final evaluation on 150 human-verified examples:

    Intent Accuracy       61.33%
    Macro F1              0.5531
    Evidence Support      89.33%
    Non-empty Replies     100%
    Auto-handle           24%
    Escalation            76%

The V3 classifier outperforms both simple baselines:

    Majority baseline       48.67%
    TF-IDF retrieval        34.00%
    V3 classifier           61.33%

The V3 classifier improves accuracy by:

    12.66 percentage points over the majority baseline
    27.33 percentage points over the TF-IDF retrieval baseline

The project intentionally reports its limitations rather than presenting the headline accuracy as an end-to-end production metric.