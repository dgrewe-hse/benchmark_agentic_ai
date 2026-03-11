#!/usr/bin/env python3
"""
select_models.py — Interactive model registry builder.

Presents the curated model catalog, lets you pick which models to benchmark,
fetches their current HuggingFace revision SHAs, and writes models/registry.yaml.

Usage:
    python3 models/select_models.py

Requirements:
    - Logged in to HuggingFace: hf auth login
    - Gated models (Llama, Phi) require license accepted at huggingface.co
"""

import dataclasses
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table, box

console = Console()

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


# ── Model catalog ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ModelEntry:
    id: str
    tier: str
    hf_repo: str
    family: str
    params_b: Optional[float]
    params_label: str
    precision: str
    gated: bool
    vram_gb_estimate: int
    context_length: int
    vllm_extra_args: str
    architecture: Optional[str] = None
    quantized_path: Optional[str] = None
    notes: str = ""


CATALOG: list[ModelEntry] = [
    # ── Small ──────────────────────────────────────────────────────────────────
    ModelEntry(
        id="qwen2.5-coder-7b",
        tier="small", family="qwen", params_b=7, params_label="7B",
        hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        precision="fp16", gated=False,
        vram_gb_estimate=16, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.85 --max-model-len 32768",
        notes="Primary small-tier Qwen model",
    ),
    ModelEntry(
        id="glm-4-flash",
        tier="small", family="glm4", params_b=9, params_label="9B",
        hf_repo="THUDM/glm-4-9b-chat",
        precision="fp16", gated=False,
        vram_gb_estimate=18, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.85 --max-model-len 32768 --trust-remote-code",
        notes="GLM-4-Flash equivalent; verify exact repo at download time",
    ),
    ModelEntry(
        id="phi-4-mini",
        tier="small", family="phi4", params_b=3.8, params_label="3.8B",
        hf_repo="microsoft/Phi-4-mini-instruct",
        precision="fp16", gated=True,
        vram_gb_estimate=8, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.80 --max-model-len 32768",
        notes="Smallest model; efficiency baseline",
    ),
    ModelEntry(
        id="llama-3.2-3b",
        tier="small", family="llama3", params_b=3, params_label="3B",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        precision="fp16", gated=True,
        vram_gb_estimate=7, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.80 --max-model-len 32768",
        notes="Llama baseline at small scale",
    ),
    # ── Mid ────────────────────────────────────────────────────────────────────
    ModelEntry(
        id="qwen2.5-coder-14b",
        tier="mid", family="qwen", params_b=14, params_label="14B",
        hf_repo="Qwen/Qwen2.5-Coder-14B-Instruct",
        precision="fp16", gated=False,
        vram_gb_estimate=32, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.88 --max-model-len 32768",
    ),
    ModelEntry(
        id="glm-4-air",
        tier="mid", family="glm4", params_b=None, params_label="~?",
        hf_repo="THUDM/glm-4-air",
        precision="fp16", gated=False,
        vram_gb_estimate=26, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.88 --max-model-len 32768 --trust-remote-code",
        notes="Verify HF repo — Zhipu uses multiple aliases",
    ),
    ModelEntry(
        id="llama-3.1-8b",
        tier="mid", family="llama3", params_b=8, params_label="8B",
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        precision="fp16", gated=True,
        vram_gb_estimate=19, context_length=131072,
        vllm_extra_args="--gpu-memory-utilization 0.88 --max-model-len 32768",
        notes="Llama mid-tier baseline",
    ),
    # ── Large ──────────────────────────────────────────────────────────────────
    ModelEntry(
        id="qwen2.5-coder-32b",
        tier="large", family="qwen", params_b=32, params_label="32B",
        hf_repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        precision="fp8", gated=False,
        vram_gb_estimate=26, context_length=131072,
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 16384",
        quantized_path="/data/models/quantized/qwen2.5-coder-32b-fp8",
        notes="Quantise with llm-compressor before serving; see models/quantize.sh",
    ),
    ModelEntry(
        id="deepseek-coder-v2-lite",
        tier="large", family="deepseek", params_b=16, params_label="16B (MoE)",
        hf_repo="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        precision="fp8", gated=False,
        vram_gb_estimate=24, context_length=163840,
        architecture="moe",
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 16384 --trust-remote-code",
        quantized_path="/data/models/quantized/deepseek-coder-v2-lite-fp8",
        notes="MoE — different throughput profile vs dense. Verify vLLM MoE support at install time.",
    ),
    ModelEntry(
        id="glm-4-plus",
        tier="large", family="glm4", params_b=None, params_label="~?",
        hf_repo="THUDM/glm-4-plus",
        precision="fp8", gated=False,
        vram_gb_estimate=34, context_length=131072,
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 16384 --trust-remote-code",
        quantized_path="/data/models/quantized/glm-4-plus-fp8",
        notes="Verify HF repo at download time",
    ),
    # ── XL ─────────────────────────────────────────────────────────────────────
    ModelEntry(
        id="qwen2.5-coder-72b",
        tier="xl", family="qwen", params_b=72, params_label="72B",
        hf_repo="Qwen/Qwen2.5-Coder-72B-Instruct",
        precision="fp8", gated=False,
        vram_gb_estimate=78, context_length=131072,
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 4",
        quantized_path="/data/models/quantized/qwen2.5-coder-72b-fp8",
        notes="XL exclusive; run model_swap.sh before loading",
    ),
    ModelEntry(
        id="deepseek-r1-70b",
        tier="xl", family="deepseek", params_b=70, params_label="70B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        precision="fp8", gated=False,
        vram_gb_estimate=76, context_length=131072,
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 4",
        quantized_path="/data/models/quantized/deepseek-r1-70b-fp8",
        notes="Distilled Llama-70B variant of R1; use over full MoE R1 for memory fit",
    ),
    ModelEntry(
        id="llama-3.3-70b",
        tier="xl", family="llama3", params_b=70, params_label="70B",
        hf_repo="meta-llama/Llama-3.3-70B-Instruct",
        precision="fp8", gated=True,
        vram_gb_estimate=76, context_length=131072,
        vllm_extra_args="--dtype fp8 --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 4",
        quantized_path="/data/models/quantized/llama-3.3-70b-fp8",
        notes="Llama XL baseline",
    ),
]

TIER_ORDER = ["small", "mid", "large", "xl"]
TIER_LABELS = {
    "small": ("SMALL TIER (fp16, up to 3 concurrent)", "cyan"),
    "mid":   ("MID TIER   (fp16, up to 2 concurrent)", "blue"),
    "large": ("LARGE TIER (fp8,  up to 2 concurrent)", "yellow"),
    "xl":    ("XL TIER    (fp8,  exclusive — 1 at a time)", "red"),
}


# ── HuggingFace helpers ───────────────────────────────────────────────────────

def check_hf_login() -> str:
    try:
        from huggingface_hub import whoami
        info = whoami()
        return info["name"]
    except Exception:
        console.print("\n[red]Not logged in to HuggingFace Hub.[/red]")
        console.print("Run: [bold]hf auth login[/bold]\n")
        sys.exit(1)


def fetch_sha(entry: ModelEntry) -> Optional[str]:
    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        info = model_info(entry.hf_repo)
        return info.sha
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name:
            console.print(
                f"  [yellow]GATED[/yellow] {entry.hf_repo} — "
                f"accept license at huggingface.co/{entry.hf_repo}"
            )
        elif "NotFound" in name or "Repository" in name:
            console.print(f"  [red]NOT FOUND[/red] {entry.hf_repo} — check repo name/visibility")
        else:
            console.print(f"  [red]ERROR[/red] {entry.hf_repo}: {e}")
        return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def display_catalog() -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("#",       style="dim", width=3)
    table.add_column("Tier",    width=6)
    table.add_column("Model ID", width=24)
    table.add_column("HF Repo",  width=44)
    table.add_column("Params",   width=10)
    table.add_column("Precision", width=10)
    table.add_column("Gated",    width=6)

    tier_colors = {"small": "cyan", "mid": "blue", "large": "yellow", "xl": "red"}

    for i, m in enumerate(CATALOG, 1):
        color = tier_colors[m.tier]
        gated = "[red]yes[/red]" if m.gated else "[green]no[/green]"
        table.add_row(
            str(i),
            f"[{color}]{m.tier}[/{color}]",
            m.id,
            m.hf_repo,
            m.params_label,
            m.precision,
            gated,
        )

    console.print(table)


def parse_selection(raw: str) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(len(CATALOG)))
    indices = []
    for token in raw.split():
        try:
            i = int(token) - 1
            if 0 <= i < len(CATALOG):
                indices.append(i)
            else:
                console.print(f"  [yellow]Warning: {token} out of range — skipping[/yellow]")
        except ValueError:
            console.print(f"  [yellow]Warning: '{token}' is not a number — skipping[/yellow]")
    return sorted(set(indices))


# ── YAML builder ──────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def build_yaml(selected: list[tuple[ModelEntry, Optional[str]]]) -> str:
    today = date.today().isoformat()
    lines = [
        "# Model Registry",
        f"# Generated by models/select_models.py on {today}",
        "# All models must be pinned to an exact HuggingFace Hub revision SHA before benchmarking.",
        "#",
        "# NEVER benchmark a model without a revision SHA — this breaks reproducibility.",
        "",
        "models:",
        "",
    ]

    by_tier: dict[str, list[tuple[ModelEntry, Optional[str]]]] = {t: [] for t in TIER_ORDER}
    for entry, sha in selected:
        by_tier[entry.tier].append((entry, sha))

    for tier in TIER_ORDER:
        entries = by_tier[tier]
        if not entries:
            continue
        label, _ = TIER_LABELS[tier]
        lines.append(f"  # ── {label} {'─' * max(0, 52 - len(label))}")
        lines.append("")
        for entry, sha in entries:
            if sha:
                rev_line = f"    revision: {sha}  # fetched {today}"
            else:
                rev_line = (
                    "    revision: null"
                    "  # FETCH FAILED — run: "
                    f"python3 -c \"from huggingface_hub import model_info; "
                    f"print(model_info('{entry.hf_repo}').sha)\""
                )
            lines.append(f"  - id: {entry.id}")
            lines.append(f"    hf_repo: {entry.hf_repo}")
            lines.append(rev_line)
            lines.append(f"    family: {entry.family}")
            lines.append(f"    tier: {entry.tier}")
            if entry.params_b is not None:
                lines.append(f"    params_b: {entry.params_b}")
            else:
                lines.append("    params_b: null          # Confirm at download")
            lines.append(f"    precision: {entry.precision}")
            if entry.quantized_path:
                lines.append(f"    quantized_path: {entry.quantized_path}")
            if entry.architecture:
                lines.append(f"    architecture: {entry.architecture}")
            lines.append(f"    vram_gb_estimate: {entry.vram_gb_estimate}")
            lines.append(f"    context_length: {entry.context_length}")
            lines.append(f"    vllm_extra_args: {_fmt(entry.vllm_extra_args)}")
            if entry.notes:
                lines.append(f"    notes: {_fmt(entry.notes)}")
            lines.append("")

    lines += [
        "  # ── REFERENCE (API — judge + evaluator only, NOT benchmarked) ──────────────",
        "",
        "  - id: gpt-4o",
        "    provider: openai",
        "    model_string: gpt-4o-2024-11-20   # Pin to dated version",
        "    tier: reference",
        "    role: [judge, baseline]",
        '    notes: "Primary judge. Always use dated model string, never the \'gpt-4o\' alias."',
        "",
        "  - id: claude-4-6-sonnet",
        "    provider: anthropic",
        "    model_string: claude-4-6-sonnet-20260307   # Pin to dated version",
        "    tier: reference",
        "    role: [judge]",
        '    notes: "Secondary judge for ensemble scoring. Average with GPT-4o."',
        "",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("\n[bold]Model Registry Builder[/bold]")
    console.print("─" * 40)

    username = check_hf_login()
    console.print(f"Logged in as: [green]{username}[/green]\n")

    display_catalog()

    raw = console.input(
        "\nSelect models to benchmark "
        "(space-separated numbers, e.g. [bold]1 3 5[/bold], or [bold]all[/bold]): "
    )
    indices = parse_selection(raw)

    if not indices:
        console.print("[red]No valid models selected. Exiting.[/red]")
        sys.exit(0)

    chosen = [CATALOG[i] for i in indices]
    console.print(f"\nSelected [bold]{len(chosen)}[/bold] model(s):")
    for m in chosen:
        gated = " [yellow](gated — accept license on HF first)[/yellow]" if m.gated else ""
        console.print(f"  • {m.id}  ({m.hf_repo}){gated}")

    confirm = console.input("\nFetch revision SHAs from HuggingFace Hub? [Y/n]: ").strip().lower()
    if confirm == "n":
        console.print("[yellow]Skipping SHA fetch — revisions will be null.[/yellow]")
        results = [(m, None) for m in chosen]
    else:
        results = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Fetching SHAs...", total=len(chosen))
            for m in chosen:
                progress.update(task, description=f"Fetching {m.id}...")
                sha = fetch_sha(m)
                results.append((m, sha))
                progress.advance(task)

        ok = sum(1 for _, sha in results if sha)
        failed = len(results) - ok
        console.print(f"\nSHA fetch complete: [green]{ok} succeeded[/green]"
                      + (f", [yellow]{failed} failed[/yellow]" if failed else ""))

    write = console.input(f"\nWrite [bold]{REGISTRY_PATH}[/bold]? [y/N]: ").strip().lower()
    if write != "y":
        console.print("Aborted — registry not written.")
        sys.exit(0)

    yaml_str = build_yaml(results)
    REGISTRY_PATH.write_text(yaml_str)

    console.print(f"\n[green]Written:[/green] {REGISTRY_PATH}")
    console.print(f"  {len(results)} local model(s) + 2 reference judges")
    if any(sha is None for _, sha in results):
        console.print(
            "\n[yellow]Some revisions are null.[/yellow] "
            "Accept gated licenses on huggingface.co then re-run this script."
        )
    console.print()


if __name__ == "__main__":
    main()
