"""sys1bench CLI.

  sys1bench generate support_tickets --n 1000 --seed 42 --out data/manifests/generated/tickets.jsonl
  sys1bench run --manifest ... --adapter jev_openrouter --model typesafe/jev-1.13 --framings data/framings/support_tickets.yaml
  sys1bench score --predictions results/preds.jsonl --out results/summary.json
  sys1bench canary --adapter jev_openrouter --model typesafe/jev-1.13
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from .adapters import get_adapter, list_adapters
from .framing import apply_corruption, expand_framings, permute_options, strip_options, strip_state
from .framing.expand import load_framings
from .generators import GENERATORS, get_generator
from .report import framing_scorecard, group_rows, markdown_table, scorecard
from .runners import ResponseCache, load_manifest, run_items, write_manifest, write_predictions
from .runners.benchmark_runner import load_predictions
from .runners.canary import run_canary

app = typer.Typer(add_completion=False, help="Benchmark harness for typed System One decision models.")


def _adapter_from(adapter: str | None, model: str | None, config: str | None):
    cfg = {}
    if config:
        cfg = yaml.safe_load(Path(config).read_text()) or {}
    adapter = adapter or cfg.pop("adapter")
    if model:
        cfg["model_id"] = model
    return get_adapter(adapter, **cfg)


@app.command()
def adapters():
    """List registered adapters (including entry-point plugins)."""
    for a in list_adapters():
        typer.echo(a)


@app.command()
def generators():
    for g in sorted(GENERATORS):
        typer.echo(f"{g}@{GENERATORS[g].version}")


@app.command()
def generate(name: str, out: Path, n: int = 1000, seed: int = 42, cardinality: int | None = None,
             target_tokens: int | None = None, distractor_density: float = 0.0, label_noise: float = 0.0,
             unknowable_frac: float = 0.0, none_correct_frac: float = 0.0, rules_out: Path | None = None):
    """Generate a Tier G manifest (and optional regex rules file)."""
    g = get_generator(name, n=n, seed=seed, cardinality=cardinality, target_tokens=target_tokens,
                      distractor_density=distractor_density, label_noise=label_noise, unknowable_frac=unknowable_frac,
                      none_correct_frac=none_correct_frac)
    items = g.generate()
    write_manifest(items, out)
    if rules_out:
        rules_out.write_text(yaml.safe_dump(g.regex_rules(), sort_keys=False))
    typer.echo(f"wrote {len(items)} items to {out} ({g.generator_id})")


@app.command()
def run(manifest: Path, out: Path, adapter: str | None = None, model: str | None = None, config: Path | None = None,
        framings: Path | None = None, permutations: int = 0, corruption: bool = False, short_circuit: bool = False,
        cache: Path = Path("cache.sqlite"), concurrency: int = 1, suite: str | None = None, limit: int | None = None):
    """Run a manifest through an adapter with optional framing / permutation / control expansions."""
    ad = _adapter_from(adapter, model, str(config) if config else None)
    items = load_manifest(manifest)
    if limit:
        items = items[:limit]
    if hasattr(ad, "fit"):
        ad.fit(items)
    rows_all = []
    if framings:
        items = expand_framings(items, load_framings(framings))
    if permutations:
        items = permute_options(items, permutations)
    c = ResponseCache(cache)
    rows_all += run_items(items, ad, c, suite=suite, arm="main", concurrency=concurrency, progress=True)
    base = [i for i in items if i.permutation_id == "p0" and all(q.framing_id == "f0" for q in i.questions.values())]
    if corruption:
        rows_all += run_items(apply_corruption(base), ad, c, suite=suite, arm="corruption", concurrency=concurrency)
    if short_circuit:
        rows_all += run_items(strip_state(base), ad, c, suite=suite, arm="state_only", concurrency=concurrency)
        rows_all += run_items(strip_options(base), ad, c, suite=suite, arm="options_only", concurrency=concurrency)
    write_predictions(rows_all, out)
    typer.echo(f"wrote {len(rows_all)} prediction rows to {out}; cache size {len(c)}")


@app.command()
def score(predictions: list[Path], out: Path = Path("results/summary.json"), table: Path | None = None):
    """Score prediction files into scorecards (per model x question) and a markdown table."""
    rows = []
    for p in predictions:
        rows += load_predictions(p)
    main = [r for r in rows if r.arm in (None, "main")]
    cards = {}
    for (model, qk), rs in group_rows(main, "model_id", "question_key").items():
        cards[f"{model} / {qk}"] = scorecard(rs)
        cards[f"{model} / {qk}"]["framing_variants"] = framing_scorecard([r for r in rows if r.model_id == model and r.question_key == qk])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=1, default=str))
    md = markdown_table(cards)
    if table:
        table.write_text(md)
    typer.echo(md)


@app.command()
def canary(adapter: str, model: str | None = None, manifest: Path = Path("data/canary/canary.jsonl"),
           store: Path = Path("results/canary"), config: Path | None = None):
    """Run the drift canary for a hosted model."""
    ad = _adapter_from(adapter, model, str(config) if config else None)
    res = run_canary(load_manifest(manifest), ad, store)
    typer.echo(json.dumps(res, indent=1))
    if res["drift_suspected"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()


@app.command("sweep-cardinality")
def sweep_cardinality(out: Path, adapter: Optional[str] = None, model: Optional[str] = None, config: Optional[Path] = None,
                      ks: str = "2,4,6,8,10,12", n: int = 300, seed: int = 42, cache: Path = Path("cache.sqlite"), concurrency: int = 1,
                      generator: str = "support_tickets", question_key: str = "queue"):
    """Suite C: accuracy / ECE-over-floor / latency vs number of options (use --generator rag_relevance --question-key best_passage for K up to 255)."""
    from .runners.sweeps import cardinality_sweep

    ad = _adapter_from(adapter, model, str(config) if config else None)
    res = cardinality_sweep(ad, [int(k) for k in ks.split(",")], n=n, seed=seed, generator=generator, question_key=question_key,
                            cache=ResponseCache(cache), concurrency=concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        typer.echo(f"K={k:3d} n={v.get('n')} acc={v.get('accuracy', float('nan')):.3f} ECE/floor={v.get('ece_over_floor', float('nan')):.2f} p50={v.get('latency', {}).get('p50', float('nan')):.0f}ms")


@app.command("sweep-length")
def sweep_length(out: Path, adapter: Optional[str] = None, model: Optional[str] = None, config: Optional[Path] = None,
                 lengths: str = "128,256,512,1024,2048,4096", n: int = 300, seed: int = 42, cache: Path = Path("cache.sqlite"), concurrency: int = 1):
    """Suite C: accuracy / truncation / latency vs state length in approximate tokens."""
    from .runners.sweeps import length_sweep

    ad = _adapter_from(adapter, model, str(config) if config else None)
    res = length_sweep(ad, [int(x) for x in lengths.split(",")], n=n, seed=seed, cache=ResponseCache(cache), concurrency=concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    for L, v in res.items():
        q = v["by_question"].get("queue", {})
        typer.echo(f"L={L:5d} tokens~{v['state_tokens_mean']:.0f} queue acc={q.get('accuracy', float('nan')):.3f} trunc={q.get('truncation_rate', float('nan')):.2f} err={q.get('error_rate', float('nan')):.2f} p50={q.get('latency', {}).get('p50', float('nan')):.0f}ms")


@app.command("sweep-budget")
def sweep_budget(out: Path, budgets: str = "64,128,192,256,384,512", k: int = 12, n: int = 300, seed: int = 42,
                 checkpoint: str = "english", max_len: int = 1024, device: str = "cuda", cache: Path = Path("cache.sqlite")):
    """Suite C (Laya): accuracy at fixed K as the option token budget head_max_len grows."""
    from .runners.sweeps import budget_sweep

    def factory(b: int):
        return get_adapter("laya_local", checkpoint=checkpoint, head_max_len=b, max_len=max_len, device=device)

    res = budget_sweep(factory, [int(b) for b in budgets.split(",")], k=k, n=n, seed=seed, cache=ResponseCache(cache))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    for b, v in res.items():
        typer.echo(f"head_max_len={b:4d} (~{v['tokens_per_option_approx']:.0f} tok/option) acc={v.get('accuracy', float('nan')):.3f} ECE/floor={v.get('ece_over_floor', float('nan')):.2f}")


@app.command()
def interference(manifest: Path, out: Path, target: str = "queue", adapter: Optional[str] = None, model: Optional[str] = None,
                 config: Optional[Path] = None, qs: str = "2,5,10,20", limit: int = 300, cache: Path = Path("cache.sqlite"), concurrency: int = 1):
    """Suite E: does co-asking other questions change the target question's answer?"""
    from .runners.sweeps import interference_sweep

    ad = _adapter_from(adapter, model, str(config) if config else None)
    items = load_manifest(manifest)[:limit]
    res = interference_sweep(ad, items, target, qs=[0] + [int(q) for q in qs.split(",")], cache=ResponseCache(cache), concurrency=concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    typer.echo(f"alone: acc={res['alone'].get('accuracy', float('nan')):.3f}")
    for kind, d in res["by_kind"].items():
        for Q, v in d.items():
            typer.echo(f"{kind:11s} Q={Q:2d} JSD={v['mean_jsd_vs_alone']:.4f} flips={v['argmax_flip_rate']:.3f} dAcc={v['accuracy_delta_vs_alone']:+.3f} p50={v['latency_p50_ms']:.0f}ms cost/q={v['cost_per_question_usd']}")


@app.command()
def report(results_dir: Path, out: Optional[Path] = None, title: str = "sys1bench results"):
    """Build the markdown results report for one or more model result directories (hosted and local kept apart)."""
    from .report.results_doc import build_report

    md = build_report([results_dir] if (results_dir / "preds_tickets.jsonl").exists() else sorted(p for p in results_dir.iterdir() if p.is_dir()), title=title)
    out = out or (results_dir / "REPORT.md")
    out.write_text(md)
    typer.echo(f"wrote {out}")
