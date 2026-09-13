import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

if API_KEY == "your_actual_api_key_here":
    raise RuntimeError("OPENAI_API_KEY is still the placeholder value")

client = OpenAI(api_key=API_KEY)

INPUT_PATH = ROOT / "evaluation" / "reply_evaluation_results.csv"

# IMPORTANT:
# Save this as a separate file so the previous results are not overwritten.
OUTPUT_PATH = ROOT / "evaluation" / "llm_judge_results_40.csv"

MODEL = "gpt-5.6-luna"

# Smaller batches reduce token usage per request.
BATCH_SIZE = 5

# Number of examples to judge.
SAMPLE_SIZE = 40

RANDOM_STATE = 42


def build_prompt(batch):

    examples = []

    for idx, row in batch.iterrows():

        evidence = []

        for n in range(1, 4):

            customer = str(
                row.get(f"evidence_{n}_customer", "")
            )

            support = str(
                row.get(f"evidence_{n}_support", "")
            )

            similarity = row.get(
                f"evidence_{n}_similarity", ""
            )

            evidence.append(
                f"Evidence {n}:\n"
                f"Historical customer: {customer}\n"
                f"Historical support reply: {support}\n"
                f"Similarity: {similarity}"
            )

        examples.append(
            f"""
EXAMPLE_ID: {idx}

Customer message:
{row["customer_text"]}

Expected intent:
{row["expected_intent"]}

Predicted intent:
{row["predicted_intent"]}

Generated reply:
{row["draft_reply"]}

Escalation decision:
{row["decision"]}

Escalation reason:
{row["decision_reason"]}

Historical evidence:
{chr(10).join(evidence)}
"""
        )

    prompt = f"""
You are evaluating an AI customer-support agent for a
software engineering take-home assignment.

Evaluate every example independently.

For each example give integer scores from 1 to 5.

relevance:
Does the generated reply address the customer's actual problem?

helpfulness:
Would the reply meaningfully help the customer?

grounding:
Is the generated reply supported by the provided historical evidence?

safety:
Is the reply safe and appropriate for customer support?

overall:
What is the overall quality of the generated reply?

Also provide a short explanation.

Return ONLY valid JSON in exactly this structure:

{{
  "results": [
    {{
      "example_id": 0,
      "relevance": 1,
      "helpfulness": 1,
      "grounding": 1,
      "safety": 1,
      "overall": 1,
      "reason": "short explanation"
    }}
  ]
}}

There are {len(batch)} examples in this batch.

{chr(10).join(examples)}
"""

    return prompt


def judge_batch(batch):

    prompt = build_prompt(batch)

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    text = response.output_text.strip()

    # Remove accidental markdown code fences.
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    data = json.loads(text)

    return data["results"]


def main():

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} evaluation examples.")

    # Deterministic representative sample.
    sample_df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    ).sort_index()

    print(f"Selected {len(sample_df)} examples for LLM judging.")
    print(f"Using model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    expected_requests = (
        len(sample_df) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(f"Expected API requests: {expected_requests}")
    print()

    all_results = []

    for start in range(0, len(sample_df), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(sample_df)
        )

        batch = sample_df.iloc[start:end]

        print(
            f"Judging selected examples "
            f"{start + 1}-{end}..."
        )

        try:

            results = judge_batch(batch)

            if len(results) != len(batch):

                print(
                    f"WARNING: Expected {len(batch)} "
                    f"results but received {len(results)}."
                )

            all_results.extend(results)

            print(
                f"Completed {end}/{len(sample_df)}"
            )

        except Exception as e:

            print(
                f"ERROR in batch {start + 1}-{end}: {e}"
            )

            for idx in batch.index:

                all_results.append(
                    {
                        "example_id": idx,
                        "relevance": None,
                        "helpfulness": None,
                        "grounding": None,
                        "safety": None,
                        "overall": None,
                        "reason": f"LLM judge error: {e}"
                    }
                )

    judge_df = pd.DataFrame(all_results)

    judge_df["example_id"] = pd.to_numeric(
        judge_df["example_id"],
        errors="coerce"
    )

    sample_df = sample_df.copy()

    sample_df["example_id"] = sample_df.index

    final_df = sample_df.merge(
        judge_df,
        on="example_id",
        how="left"
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    successful = final_df["overall"].notna().sum()

    print()
    print("===================================")
    print("LLM JUDGE COMPLETE")
    print("===================================")
    print(f"Total sampled examples: {len(final_df)}")
    print(f"Successfully judged: {successful}")
    print(f"Failed: {len(final_df) - successful}")
    print(f"Saved: {OUTPUT_PATH}")
    print("===================================")


if __name__ == "__main__":
    main()