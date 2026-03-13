#!/usr/bin/env python3
"""
llm_judge.py — Frontier model judge for real-world challenge scoring

Scores generated code against a challenge rubric using GPT-4o and Claude 3.5 Sonnet.
Returns the average Completeness Score and logs both judge outputs to the results DB.

Usage:
  python scoring/llm_judge.py \
    --run-id <uuid> \
    --challenge C1-route-planner \
    --code-dir /path/to/generated/code
"""

import argparse
import json
import os
import hashlib
from pathlib import Path

import openai
import anthropic
from results_db import log_judge_output, log_run_end


RUBRIC_DIR = Path(__file__).parent / "rubrics"


def load_rubric(challenge: str) -> dict:
    path = RUBRIC_DIR / f"{challenge}.json"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    with open(path) as f:
        return json.load(f)


def collect_code_snapshot(code_dir: Path, max_chars: int = 40000) -> str:
    """Collect all source files into a single string for the judge."""
    extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sql", ".yaml", ".yml", ".toml"}
    files = sorted(code_dir.rglob("*"))
    snapshot = []
    total = 0

    for f in files:
        if f.suffix in extensions and f.is_file() and "__pycache__" not in str(f):
            try:
                content = f.read_text(errors="replace")
                header = f"\n\n{'='*60}\nFILE: {f.relative_to(code_dir)}\n{'='*60}\n"
                snippet = header + content
                if total + len(snippet) > max_chars:
                    snapshot.append(f"\n\n[... {f.relative_to(code_dir)} truncated — {len(content)} chars ...]")
                    break
                snapshot.append(snippet)
                total += len(snippet)
            except Exception:
                pass

    return "".join(snapshot)


def build_judge_prompt(spec_text: str, rubric: dict, code_snapshot: str) -> str:
    dimensions = "\n".join(
        f"- **{dim['name']}** (0–100): {dim['description']}"
        for dim in rubric["dimensions"]
    )
    return f"""You are an expert software engineering evaluator assessing an AI-generated coding project.

## Project Specification
{spec_text}

## Evaluation Rubric
Score each dimension from 0 to 100 based on the generated code below.

{dimensions}

## Generated Code
{code_snapshot}

## Instructions
1. Carefully read the specification and understand what was required.
2. Examine the generated code.
3. Score each rubric dimension independently from 0–100.
4. Provide brief, specific reasoning for each score.
5. Compute an overall_completeness score (0–100) as your holistic assessment.

Respond ONLY with valid JSON in this exact format:
{{
  {', '.join(f'"{d["name"]}": <score 0-100>' for d in rubric["dimensions"])},
  "overall_completeness": <score 0-100>,
  "reasoning": {{
    {', '.join(f'"{d["name"]}": "<1-2 sentences>"' for d in rubric["dimensions"])},
    "overall": "<2-3 sentences holistic summary>"
  }}
}}"""


def judge_with_gpt4o(prompt: str) -> dict:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=2000
    )
    return json.loads(response.choices[0].message.content)


def judge_with_claude(prompt: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000
    )
    text = response.content[0].text
    # Claude may wrap in ```json ... ``` — strip if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def run_judge(run_id: str, challenge: str, code_dir: Path, spec_path: Path | None = None) -> dict:
    rubric = load_rubric(challenge)

    # Load spec
    if spec_path is None:
        spec_path = Path(f"challenges/{challenge}/SPEC.md")
    spec_text = spec_path.read_text() if spec_path.exists() else f"Challenge: {challenge}"

    code_snapshot = collect_code_snapshot(code_dir)
    prompt = build_judge_prompt(spec_text, rubric, code_snapshot)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    print(f"  Running GPT-4o judge...")
    gpt4o_result = judge_with_gpt4o(prompt)
    gpt4o_score = gpt4o_result.get("overall_completeness", 0)

    print(f"  Running Claude 3.5 Sonnet judge...")
    claude_result = judge_with_claude(prompt)
    claude_score = claude_result.get("overall_completeness", 0)

    delta = abs(gpt4o_score - claude_score)
    avg_score = (gpt4o_score + claude_score) / 2

    print(f"  GPT-4o: {gpt4o_score:.1f}  |  Claude: {claude_score:.1f}  |  Avg: {avg_score:.1f}  |  Delta: {delta:.1f}")
    if delta > 15:
        print(f"  ⚠️  Delta > 15 — flagging for manual review")

    # Log per-dimension scores
    for dim in rubric["dimensions"]:
        name = dim["name"]
        log_judge_output(run_id, "gpt-4o-2024-11-20", name,
                         gpt4o_result.get(name, 0), gpt4o_result.get("reasoning", {}).get(name, ""))
        log_judge_output(run_id, "claude-3-5-sonnet-20241022", name,
                         claude_result.get(name, 0), claude_result.get("reasoning", {}).get(name, ""))

    # Update run record with judge scores
    log_run_end(run_id, {
        "judge_gpt4o_score": gpt4o_score,
        "judge_claude_score": claude_score,
        "completeness_score": avg_score,
    })

    return {
        "gpt4o_score": gpt4o_score,
        "claude_score": claude_score,
        "avg_score": avg_score,
        "delta": delta,
        "flagged": delta > 15,
        "prompt_hash": prompt_hash,
        "gpt4o_detail": gpt4o_result,
        "claude_detail": claude_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args()

    result = run_judge(args.run_id, args.challenge, args.code_dir, args.spec)
    print(json.dumps(result, indent=2))
