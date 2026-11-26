#!/usr/bin/env python3
"""Deliverable 4 evaluation: four systems and the control, on the same 27 questions and criteria.

  baseline      Deliverable 1 direct prompt
  d2            Deliverable 2 pipeline
  agentes       Deliverable 3 agent system (every agent on the base model)
  agentes_ft    Deliverable 4: the same system, the Planner on the fine-tuned adapter
  control_icl   control: base weights, the Planner gets the 4 nearest training examples in its prompt

All five run in the same session with one copy of the base model in memory (the adapter is switched
on only for the Planner of agentes_ft). Writes results/deliverable4/<config>.json, one trace per
question for the three agent configurations, a per-question comparison and the LaTeX table.

    python scripts/evaluate_d4.py --adapter usuario/sql4business-planificador-qlora
    python scripts/evaluate_d4.py --adapter results/deliverable4/adapter --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sql4business.evaluation import environment, load_questions  # noqa: E402
from sql4business.evaluation_d3 import run_config  # noqa: E402
from sql4business.icl_control import RetrievalFewShotPlanner, load_examples  # noqa: E402
from sql4business.model import MODEL_ID, GenerationSettings, HuggingFaceGenerator  # noqa: E402

CONFIGS = ("baseline", "d2", "agentes", "agentes_ft", "control_icl")
AGENT_CONFIGS = ("agentes", "agentes_ft", "control_icl")
LABELS = {
    "baseline": "(1) Prompt directo (D1)",
    "d2": "(2) Pipeline (D2)",
    "agentes": "(3) Agentes, modelo base (D3)",
    "agentes_ft": "(4) Agentes + Planificador ajustado (D4)",
    "control_icl": "Control: base + mismos datos en el prompt",
}


def default_adapter() -> tuple[str, str | None]:
    published = ROOT / "results" / "deliverable4" / "adapter_hf.json"
    if published.exists():
        data = json.loads(published.read_text(encoding="utf-8"))
        return data["repo_id"], data.get("revision")
    return str(ROOT / "results" / "deliverable4" / "adapter"), None


def repeated_plans(trace: list[dict]) -> tuple[int, int]:
    """(feedback messages to the Planner, new plans identical to the plan that was sent back)."""

    plans = [m["contenido"] for m in trace if m["tipo"] == "plan"]
    feedback = sum(1 for m in trace if m["para"] == "Planificador" and m["tipo"].startswith("replanificar"))
    def key(plan):
        return (plan["operation"], [step["sql"] for step in plan["steps"]])
    repeated = sum(1 for a, b in zip(plans, plans[1:]) if key(a) == key(b))
    return feedback, repeated


def planner_stats(result: dict) -> dict:
    stats = {"feedback": 0, "repeated": 0}
    for record in result["records"]:
        feedback, repeated = repeated_plans(record["result"].get("trace", []))
        stats["feedback"] += feedback
        stats["repeated"] += repeated
    return stats


def latex_table(results: dict[str, dict]) -> str:
    def cell(config, set_name, metric, group="global"):
        data = results[config]["summary"].get(set_name, {})
        if metric not in data:
            return "--"
        correct, total = data[metric][group]
        return f"{correct}/{total}"

    lines = [
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Oficial (15)}} & \multicolumn{3}{c}{\textbf{Paráfrasis (12)}} & \textbf{Comb.} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Sistema} & sql & answer & e2e & sql & answer & e2e & e2e & \textbf{Llam.} \\",
        r"\midrule",
    ]
    for config in CONFIGS:
        if config not in results:
            continue
        summary = results[config]["summary"]
        combined = "--"
        if "e2e" in summary.get("official", {}):
            c = [summary[s]["e2e"]["combinada"] for s in ("official", "heldout") if s in summary]
            combined = f"{sum(x[0] for x in c)}/{sum(x[1] for x in c)}"
        calls = [summary[s]["model_calls"] for s in ("official", "heldout") if s in summary]
        row = [LABELS[config]] + [cell(config, s, m) for s in ("official", "heldout") for m in ("sql", "answer", "e2e")]
        row += [combined, f"{sum(calls) / len(calls):.1f}"]
        if config == "control_icl":
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def per_question(results: dict[str, dict]) -> str:
    configs = [c for c in ("d2", "agentes", "agentes_ft", "control_icl") if c in results]
    by_id = {c: {(r["set"], r["id"]): r for r in results[c]["records"]} for c in configs}
    keys = list(by_id[configs[0]])
    lines = ["| Pregunta | Tipo | " + " | ".join(configs) + " |", "|---|---|" + "---|" * len(configs)]
    for key in keys:
        marks = [by_id[c][key]["result"]["e2e"] for c in configs]
        if len(set(marks)) > 1:
            kind = by_id[configs[0]][key]["type"]
            lines.append(f"| {key[1]} | {kind} | " + " | ".join("✓" if m else "✗" for m in marks) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    adapter, revision = default_adapter()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", default=adapter, help="Hugging Face repo id or local folder of the adapter")
    parser.add_argument("--adapter-revision", default=revision)
    parser.add_argument("--db", default=ROOT / "data" / "business.db", type=Path)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--icl-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=ROOT / "results" / "deliverable4", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tex", default=ROOT / "docs" / "deliverable4_results.tex", type=Path)
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    official = load_questions(ROOT / "data" / "questions.json")
    sets = {"official": official["questions"], "heldout": load_questions(ROOT / "data" / "paraphrases.json")["questions"]}
    if args.limit:
        sets = {name: questions[: args.limit] for name, questions in sets.items()}
        args.output_dir = args.output_dir / "prueba"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = HuggingFaceGenerator(args.model, GenerationSettings(), adapter=args.adapter, adapter_revision=args.adapter_revision)
    planners = {
        "agentes_ft": generator.adapted(),
        "control_icl": RetrievalFewShotPlanner(generator, load_examples(ROOT / "data" / "finetune" / "train.jsonl"), k=args.icl_k),
    }
    meta = {**environment(generator), "adapter": args.adapter, "adapter_revision": args.adapter_revision,
            "icl_k": args.icl_k, "limit": args.limit}
    for config in configs:
        path = args.output_dir / f"{config}.json"
        if args.resume and path.exists():
            print(f"skip {config}: {path} exists")
            continue
        started = time.perf_counter()
        trace_dir = args.output_dir / f"trazas_{config}" if config in AGENT_CONFIGS else None
        result = run_config(config, generator, args.db, sets, official["schema"], trace_dir=trace_dir,
                            planner_generator=planners.get(config))
        if config in AGENT_CONFIGS:
            result["planner"] = planner_stats(result)
        result["meta"] = {**meta, "seconds": round(time.perf_counter() - started, 1)}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"saved {path}")

    results = {c: json.loads((args.output_dir / f"{c}.json").read_text(encoding="utf-8"))
               for c in CONFIGS if (args.output_dir / f"{c}.json").exists()}
    print()
    for config, result in results.items():
        cells = []
        for set_name, data in result["summary"].items():
            metrics = " ".join(f"{m}={v['global'][0]}/{v['global'][1]}" for m, v in data.items() if isinstance(v, dict))
            cells.append(f"{set_name}: {metrics} calls={data['model_calls']}")
        extra = f" planner={result['planner']}" if "planner" in result else ""
        print(f"{config:12} " + " | ".join(cells) + extra)
    (args.output_dir / "por_pregunta.md").write_text(per_question(results), encoding="utf-8")
    tex = args.tex if not args.limit else args.output_dir / "deliverable4_results.tex"
    tex.write_text(latex_table(results), encoding="utf-8")
    print(f"\nwrote {tex} and {args.output_dir / 'por_pregunta.md'}")


if __name__ == "__main__":
    main()
