# Decision Log — Hiver AI Support Agent

1. **Selected AppleSupport as the brand.** It had enough historical responses for retrieval while remaining focused enough for a compact take-home prototype.
2. **Reconstructed customer→support pairs.** Company responses were linked to their inbound customer messages using the response relationship fields rather than treating isolated tweets as independent cases.
3. **Used a small explicit intent taxonomy.** Nine operational intents were chosen to keep classification interpretable and useful for routing.
4. **Merged notification issues into `device_issue`.** Notifications were too broad to justify a separate intent at this dataset size.
5. **Labelled the main problem, not every keyword.** This avoids routing a battery problem as an update problem merely because the customer mentions an update.
6. **Used TF-IDF + Logistic Regression for the classifier.** It is fast, reproducible, interpretable, and suitable for a small take-home system without requiring a large model-serving stack.
7. **Selected classifier V3.** V3 reached 60.00% accuracy and 0.5307 macro F1 on the golden set; later V4 changes reduced accuracy and were rejected.
8. **Excluded golden examples from classifier training.** This prevents direct label leakage into the evaluation.
9. **Excluded golden examples from retrieval.** This prevents a test customer message from being returned as its own historical evidence.
10. **Retrieved multiple historical cases.** Top-k evidence provides context and makes the support draft auditable instead of relying on one nearest example.
11. **Grounded replies in historical support behavior.** The reply generator uses observed AppleSupport response patterns rather than inventing unsupported product procedures.
12. **Escalated account/iCloud cases.** These can require account-specific verification and should not be handled automatically by this prototype.
13. **Escalated order/purchase cases.** Transaction-specific issues may require information or actions unavailable to the prototype.
14. **Used similarity as an escalation signal.** Strong historical similarity can support automation, while weak/moderate evidence should prefer human review. This is a prototype heuristic, not a safety guarantee.
15. **Compared against simple baselines and analysed failures.** The final evaluation reports a majority baseline, a TF-IDF retrieval baseline, confusion patterns, and representative failures rather than presenting the classifier score alone.

## Evaluation integrity

- Golden set: 150 examples.
- Golden examples excluded from classifier training and retrieval during evaluation.
- Human-reviewed evidence labels: 134/150 supported (89.33%).
- Final intent accuracy: 60.00%.
- Final macro F1: 0.5307.
- Majority baseline accuracy: 48.00%.
- Retrieval baseline accuracy: 34.67%.
- LLM judge: 5-example calibration completed; 40-example run was blocked by organization TPM limits, so no fabricated aggregate score is reported.
