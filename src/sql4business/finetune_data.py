"""Deliverable 4: training data for the fine-tuned Planificador.

Every example is an (input, output) pair for the Planner's role:

  input   the exact prompt the Planner reads at run time (prompts/planificador.md rendered with every
          technique on, the grounded schema and the question; for repair examples also the previous
          plan and the Verificador's feedback, written by the Verificador's own checks).
  output  the correct plan as one JSON object (answer_needs, periods, steps, composition).

Source: question templates over data/business.db (10 products, March to August 2026). No model
writes any output: each correct plan is built by code from the task's parameters, executed with the
Analyst's tools (validation, read-only SQLite, deterministic composition) and checked against a
reference value computed independently in Python from the raw tables. A task is dropped if the two
disagree or if its top item is a tie.

Two kinds of examples attack the failure diagnosed in Deliverables 1-3 (planning combined questions)
and the evidence of Deliverable 3 (the Planner repeats its plan after the Verificador's feedback):

  plan     question -> correct plan. Growth questions name the later period first in half of the
           templates (the q12 bias), and category questions need the category filter (q04).
  repair   question + a wrong previous plan + the feedback the Verificador really produces for it
           -> correct plan. Wrong plans: reversed growth (q12), dropped category filter (q04), share
           computed as growth (q11-p), two values with a direct operation (q11), growth instead of a
           comparison of leaders (q10), a listing instead of the top item (q03), a shifted month.

Separation from the evaluation (27 questions of data/questions.json and data/paraphrases.json):
  * no training step computes a numeric result set equal to any step of an evaluation question;
  * no training task has the same operation and result sets as an evaluation question;
  * no training question has a character 5-gram Jaccard similarity >= 0.5 with an evaluation question.
The report written next to the data counts what each rule removed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import AgentSystem, critic_problems
from .composition import ComposedAnswer, compose_results
from .pipeline import expand_step_references, validate_plan
from .sql_tools import connect_read_only, execute_read_only, normalize_text


SEED = 4
JACCARD_LIMIT = 0.5
DATA_FROM, DATA_TO = "2026-03-01", "2026-08-31"


# --------------------------------------------------------------------------- vocabulary


@dataclass(frozen=True)
class Period:
    key: str
    start: str
    end: str
    es_in: str     # "en agosto de 2026"
    es_of: str     # "de agosto de 2026"
    es_short: str  # "agosto de 2026"
    en_in: str     # "in August 2026"
    en_short: str  # "August 2026"

    @property
    def whole(self) -> bool:
        return self.start == DATA_FROM and self.end == DATA_TO


MONTHS = [
    ("03", "marzo", "March", "31"), ("04", "abril", "April", "30"), ("05", "mayo", "May", "31"),
    ("06", "junio", "June", "30"), ("07", "julio", "July", "31"), ("08", "agosto", "August", "31"),
]


def _periods() -> list[Period]:
    found = []
    for number, es, en, last in MONTHS:
        year = random.Random(number).choice([" de 2026", " de 2026", ""])
        found.append(Period(f"m{number}", f"2026-{number}-01", f"2026-{number}-{last}", f"en {es}{year}", f"de {es}{year}",
                            f"{es}{year}", f"in {en} 2026", f"{en} 2026"))
    ranges = [
        ("03", "04"), ("05", "06"), ("07", "08"), ("03", "05"), ("06", "08"), ("04", "06"), ("04", "05"), ("06", "07"),
    ]
    by_number = {m[0]: m for m in MONTHS}
    for first, last in ranges:
        a, b = by_number[first], by_number[last]
        joiner = " y " if int(last) - int(first) == 1 else " a "
        es = f"{a[1]}{joiner}{b[1]}"
        found.append(Period(f"r{first}{last}", f"2026-{first}-01", f"2026-{last}-{b[3]}", f"entre {a[1]} y {b[1]}",
                            f"de {es}", f"{a[1]}-{b[1]}", f"between {a[2]} and {b[2]}", f"{a[2]}-{b[2]}"))
    found.append(Period("all", DATA_FROM, DATA_TO, "en todo el semestre", "del semestre", "todo el semestre",
                        "in the whole semester", "the whole semester"))
    return found


PERIODS = _periods()
MONTH_PERIODS = [p for p in PERIODS if p.key.startswith("m")]
RANGE_PERIODS = [p for p in PERIODS if p.key.startswith("r")]
WHOLE = PERIODS[-1]


@dataclass(frozen=True)
class Metric:
    key: str
    expr: str
    es: tuple[str, ...]  # masculine singular noun phrases, so "que el de ..." works
    en: tuple[str, ...]
    rank_es: str         # "más ingresos"
    rank_en: str
    py: Callable[[list[tuple[int, str, int, int]]], float]


METRICS = [
    Metric("revenue", "SUM(s.amount)", ("el ingreso", "el ingreso total", "el monto vendido", "el total facturado"),
           ("revenue", "total revenue"), "más ingresos", "the most revenue", lambda rows: sum(r[3] for r in rows)),
    Metric("units", "SUM(s.quantity)", ("el total de unidades vendidas", "el número de unidades vendidas"),
           ("units sold",), "más unidades vendidas", "the most units sold", lambda rows: sum(r[2] for r in rows)),
    Metric("count", "COUNT(*)", ("el número de ventas", "el número de transacciones"),
           ("number of sales",), "más ventas", "the most sales", lambda rows: len(rows)),
    Metric("avg", "AVG(s.amount)", ("el ingreso promedio por venta", "el monto promedio por venta"),
           ("average revenue per sale",), "", "", lambda rows: sum(r[3] for r in rows) / len(rows)),
]
METRIC = {m.key: m for m in METRICS}


@dataclass(frozen=True)
class Scope:
    kind: str  # "all", "category", "product"
    value: str = ""

    def es(self, rng: random.Random) -> str:
        if self.kind == "category":
            return rng.choice([f" de la categoría {self.value}", f" de la categoria {self.value}", f" en la categoría {self.value}"])
        if self.kind == "product":
            return rng.choice([f" del producto {self.value}", f" de {self.value}"])
        return rng.choice(["", " de la tienda"])

    def en(self) -> str:
        return {"category": f" of the {self.value} category", "product": f" of {self.value}"}.get(self.kind, "")

    def condition(self) -> str | None:
        if self.kind == "category":
            return f"p.category = '{self.value}'"
        if self.kind == "product":
            return f"p.name = '{self.value}'"
        return None


# --------------------------------------------------------------------------- database facts


class Facts:
    """The raw tables, for reference values computed without SQL."""

    def __init__(self, db_path: str | Path):
        conn = sqlite3.connect(Path(db_path))
        try:
            self.products = conn.execute("SELECT product_id, name, category, price FROM products").fetchall()
            self.sales = conn.execute("SELECT product_id, sale_date, quantity, amount FROM sales").fetchall()
            self.inventory = {pid: (stock, reorder) for pid, stock, reorder in conn.execute("SELECT * FROM inventory")}
        finally:
            conn.close()
        self.name = {pid: name for pid, name, _, _ in self.products}
        self.category = {pid: cat for pid, _, cat, _ in self.products}
        self.price = {pid: price for pid, _, _, price in self.products}
        self.categories = sorted(set(self.category.values()))

    def rows(self, scope: Scope, period: Period, products: set[int] | None = None) -> list[tuple[int, str, int, int]]:
        def keep(row):
            pid, day = row[0], row[1]
            if not period.start <= day <= period.end:
                return False
            if products is not None and pid not in products:
                return False
            if scope.kind == "category":
                return self.category[pid] == scope.value
            if scope.kind == "product":
                return self.name[pid] == scope.value
            return True
        return [row for row in self.sales if keep(row)]

    def value(self, metric: Metric, scope: Scope, period: Period) -> float | None:
        rows = self.rows(scope, period)
        return metric.py(rows) if rows else None

    def ranking(self, entity: str, metric: Metric, period: Period, scope: Scope = Scope("all"),
                products: set[int] | None = None) -> list[tuple[str, float]]:
        groups: dict[str, list] = {}
        for row in self.rows(scope, period, products):
            key = self.name[row[0]] if entity == "product" else self.category[row[0]]
            groups.setdefault(key, []).append(row)
        return sorted(((k, metric.py(v)) for k, v in groups.items()), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- SQL builders


JOIN = "FROM sales s JOIN products p ON s.product_id = p.product_id"


def _where(*conditions: str | None) -> str:
    kept = [c for c in conditions if c]
    return (" WHERE " + " AND ".join(kept)) if kept else ""


def _dates(period: Period) -> str | None:
    return None if period.whole else f"s.sale_date BETWEEN '{period.start}' AND '{period.end}'"


def scalar_sql(metric: Metric, scope: Scope, period: Period) -> str:
    source = JOIN if scope.kind != "all" else "FROM sales s"
    return f"SELECT {metric.expr} {source}{_where(scope.condition(), _dates(period))}"


def top_sql(entity: str, metric: Metric, period: Period, scope: Scope = Scope("all"), limit: bool = True) -> str:
    column, group = ("p.name", "p.product_id") if entity == "product" else ("p.category", "p.category")
    tail = " LIMIT 1" if limit else ""
    return f"SELECT {column} {JOIN}{_where(scope.condition(), _dates(period))} GROUP BY {group} ORDER BY {metric.expr} DESC{tail}"


def _period_json(label: str, period: Period) -> dict[str, str]:
    return {"label": label, "from": period.start, "to": period.end}


def _step(index: int, purpose: str, sql: str) -> dict[str, str]:
    return {"id": f"step_{index}", "purpose": purpose, "sql": sql}


def plan_json(answer_needs: str, periods: list[dict[str, str]], steps: list[dict[str, str]], operation: str) -> dict[str, Any]:
    return {"answer_needs": answer_needs, "periods": periods, "steps": steps, "composition": {"operation": operation}}


# --------------------------------------------------------------------------- tasks


@dataclass
class Task:
    family: str
    question: str
    language: str
    plan: dict[str, Any]
    reference: Any                  # value computed in Python, without SQL
    asks_for: str                   # what the Verificador's checklist would declare
    wrong_plans: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (kind, plan)
    later_first: bool = False


def _accentless(text: str, rng: random.Random) -> str:
    """Half of the official questions have no accents or opening marks; mimic both styles."""

    if rng.random() < 0.35:
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
        text = text.lstrip("¿¡")
    return text


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _q(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b([Dd])e el\b", r"\1el", text)
    text = re.sub(r"\b([Aa]) el\b", r"\1l", text)
    return text[0] + text[1:2].upper() + text[2:] if text.startswith("¿") else _cap(text)


def scalar_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    scopes = [Scope("all")] + [Scope("category", c) for c in facts.categories] + [Scope("product", n) for _, n, _, _ in facts.products]
    for metric in METRICS:
        for scope in scopes:
            for period in PERIODS:
                value = facts.value(metric, scope, period)
                if value is None:
                    continue
                english = rng.random() < 0.15
                noun = rng.choice(metric.es)
                if english:
                    question = rng.choice([
                        f"What was the {rng.choice(metric.en)}{scope.en()} {period.en_in}?",
                        f"How much was the {rng.choice(metric.en)}{scope.en()} {period.en_in}?",
                    ])
                elif metric.key == "units" and rng.random() < 0.5:
                    where = {"category": f" de la categoría {scope.value}", "product": f" del {scope.value}"}.get(scope.kind, "")
                    question = f"¿Cuántas unidades{where} se vendieron {period.es_in}?"
                else:
                    question = rng.choice([
                        f"¿Cuál fue {noun}{scope.es(rng)} {period.es_in}?",
                        f"¿Cuánto fue {noun}{scope.es(rng)} {period.es_in}?",
                        f"Necesito {noun}{scope.es(rng)} {period.es_in}.",
                        f"¿Me puedes decir {noun}{scope.es(rng)} {period.es_in}?",
                    ])
                label = period.en_short if english else period.es_short
                plan = plan_json(
                    "one number", [] if period.whole else [_period_json(label, period)],
                    [_step(1, f"{metric.key}{' of ' + scope.value if scope.value else ''} in {period.en_short}",
                           scalar_sql(metric, scope, period))],
                    "direct",
                )
                wrong = []
                if scope.kind == "category":
                    wrong.append(("filtro_faltante", plan_json(plan["answer_needs"], plan["periods"],
                                  [_step(1, plan["steps"][0]["purpose"], scalar_sql(metric, Scope("all"), period))], "direct")))
                if period in MONTH_PERIODS and period.key != "m08":
                    shifted = MONTH_PERIODS[MONTH_PERIODS.index(period) + 1]
                    wrong.append(("mes_corrido", plan_json(plan["answer_needs"], plan["periods"],
                                  [_step(1, plan["steps"][0]["purpose"], scalar_sql(metric, scope, shifted))], "direct")))
                tasks.append(Task("directa_valor", _q(_accentless(question, rng) if not english else question),
                                  "en" if english else "es", plan, value, "number", wrong))
    return tasks


def top_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    for entity in ("product", "category"):
        for metric in (METRIC["revenue"], METRIC["units"], METRIC["count"]):
            scopes = [Scope("all")] + ([Scope("category", c) for c in facts.categories] if entity == "product" else [])
            for scope in scopes:
                for period in PERIODS:
                    ranking = facts.ranking(entity, metric, period, scope)
                    if len(ranking) < 2 or ranking[0][1] == ranking[1][1]:
                        continue
                    english = rng.random() < 0.15
                    noun_es = "producto" if entity == "product" else "categoría"
                    within = f" de la categoría {scope.value}" if scope.kind == "category" else ""
                    if english:
                        within_en = f" in the {scope.value} category" if scope.value else ""
                        question = f"Which {entity}{within_en} had {metric.rank_en} {period.en_in}?"
                    else:
                        question = rng.choice([
                            f"¿Qué {noun_es}{within} tuvo {metric.rank_es} {period.es_in}?",
                            f"¿Cuál fue el {noun_es}{within} con {metric.rank_es} {period.es_in}?" if entity == "product"
                            else f"¿Cuál fue la categoría con {metric.rank_es} {period.es_in}?",
                        ])
                    label = period.en_short if english else period.es_short
                    plan = plan_json(
                        f"one {entity} name", [] if period.whole else [_period_json(label, period)],
                        [_step(1, f"top {entity} by {metric.key} in {period.en_short}", top_sql(entity, metric, period, scope))],
                        "direct",
                    )
                    wrong = [("listado", plan_json(plan["answer_needs"], plan["periods"],
                              [_step(1, plan["steps"][0]["purpose"], top_sql(entity, metric, period, scope, limit=False))], "direct"))]
                    if scope.kind == "category":
                        wrong.append(("filtro_faltante", plan_json(plan["answer_needs"], plan["periods"],
                                      [_step(1, plan["steps"][0]["purpose"], top_sql(entity, metric, period))], "direct")))
                    tasks.append(Task("directa_lider", _q(_accentless(question, rng) if not english else question),
                                      "en" if english else "es", plan, ranking[0][0], "name", wrong))
    return tasks


def lookup_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    for pid, name, category, price in facts.products:
        stock, reorder = facts.inventory[pid]
        for question, sql, value, purpose in (
            (rng.choice([f"¿Cuál es el precio de {name}?", f"¿Cuánto cuesta el producto {name}?"]),
             f"SELECT price FROM products WHERE name = '{name}'", price, "price"),
            (rng.choice([f"¿Cuánto stock queda de {name}?", f"¿Cuántas unidades de {name} hay en inventario?"]),
             f"SELECT i.stock FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE p.name = '{name}'", stock, "stock"),
            (f"¿Cuál es el punto de reorden de {name}?",
             f"SELECT i.reorder_point FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE p.name = '{name}'", reorder, "reorder point"),
        ):
            plan = plan_json("one number", [], [_step(1, f"{purpose} of {name}", sql)], "direct")
            tasks.append(Task("directa_ficha", _q(_accentless(question, rng)), "es", plan, value, "number"))
    for category in facts.categories:
        members = sorted(n for _, n, c, _ in facts.products if c == category)
        sql = f"SELECT name FROM products WHERE category = '{category}'"
        plan = plan_json("a list of product names", [], [_step(1, f"products of {category}", sql)], "direct")
        tasks.append(Task("directa_lista", _q(_accentless(rng.choice([
            f"¿Qué productos pertenecen a la categoría {category}?", f"Lista los productos de la categoría {category}."]), rng)),
            "es", plan, members, "list"))
    for threshold in (10000, 20000, 30000, 50000):
        members = sorted(n for _, n, _, p in facts.products if p > threshold)
        sql = f"SELECT name FROM products WHERE price > {threshold}"
        plan = plan_json("a list of product names", [], [_step(1, f"products priced above {threshold}", sql)], "direct")
        tasks.append(Task("directa_lista", _q(_accentless(f"¿Qué productos cuestan más de {threshold} pesos?", rng)), "es",
                          plan, members, "list"))
    above = sorted(facts.name[pid] for pid, (s, r) in facts.inventory.items() if s >= r)
    sql = "SELECT p.name FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.stock >= i.reorder_point"
    plan = plan_json("a list of product names", [], [_step(1, "products at or above their reorder point", sql)], "direct")
    tasks.append(Task("directa_lista", "¿Qué productos tienen stock suficiente, igual o sobre su punto de reorden?", "es",
                      plan, above, "list"))
    return tasks


def growth_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    pairs = [(a, b) for a in MONTH_PERIODS for b in MONTH_PERIODS if a.start < b.start]
    pairs += [(a, b) for a in RANGE_PERIODS for b in RANGE_PERIODS if a.end < b.start]
    scopes = [Scope("all")] + [Scope("category", c) for c in facts.categories] + [
        Scope("product", n) for _, n, _, _ in facts.products]
    for metric in (METRIC["revenue"], METRIC["units"], METRIC["count"], METRIC["avg"]):
        for scope in scopes:
            for early, late in pairs:
                first, second = facts.value(metric, scope, early), facts.value(metric, scope, late)
                if not first or second is None:
                    continue
                english = rng.random() < 0.15
                later_first = rng.random() < 0.5
                noun = rng.choice(metric.es)
                s = scope.es(rng)
                if english:
                    m = rng.choice(metric.en)
                    question = (f"Was the {m}{scope.en()} {late.en_in} higher or lower than {early.en_in}, and by what percent?"
                                if later_first else
                                f"By what percent did the {m}{scope.en()} change from {early.en_short} to {late.en_short}?")
                elif later_first:
                    question = rng.choice([
                        f"¿{_cap(noun)}{s} {late.es_of} fue mayor o menor que el {early.es_of}, y en qué porcentaje?",
                        f"¿Cuánto cambió, en porcentaje, {noun}{s} {late.es_in} respecto {early.es_of}?",
                        f"Compara {noun}{s} {late.es_of} con el {early.es_of}: ¿subió o bajó, y en qué porcentaje?",
                        f"¿En qué porcentaje varió {noun}{s} {late.es_in} frente a {early.es_short}?",
                    ])
                else:
                    question = rng.choice([
                        f"¿En qué porcentaje cambió {noun}{s} entre {early.es_short} y {late.es_short}?",
                        f"¿Cuál fue el crecimiento porcentual de {noun}{s} desde {early.es_short} hasta {late.es_short}?",
                        f"¿Cuánto creció o cayó, en porcentaje, {noun}{s} de {early.es_short} a {late.es_short}?",
                    ])
                labels = (early.en_short, late.en_short) if english else (early.es_short, late.es_short)
                periods = [_period_json(labels[0], early), _period_json(labels[1], late)]
                if later_first:
                    periods.reverse()  # periods are listed as the question names them; steps go in time order
                what = f"{metric.key}{' of ' + scope.value if scope.value else ''}"
                steps = [_step(1, f"{what} in {early.en_short} (earlier period)", scalar_sql(metric, scope, early)),
                         _step(2, f"{what} in {late.en_short} (later period)", scalar_sql(metric, scope, late))]
                plan = plan_json(f"{metric.key} of both periods and their percentage change", periods, steps, "growth_pct")
                reversed_steps = [_step(1, steps[1]["purpose"], steps[1]["sql"]), _step(2, steps[0]["purpose"], steps[0]["sql"])]
                wrong = [("orden_invertido", plan_json(plan["answer_needs"], periods, reversed_steps, "growth_pct"))]
                if scope.kind == "category":
                    wrong.append(("filtro_faltante", plan_json(plan["answer_needs"], periods,
                                  [_step(1, steps[0]["purpose"], scalar_sql(metric, Scope("all"), early)),
                                   _step(2, steps[1]["purpose"], scalar_sql(metric, Scope("all"), late))], "growth_pct")))
                text = question if english else _accentless(question, rng)
                tasks.append(Task("crecimiento", _q(text), "en" if english else "es", plan,
                                  (second - first) / first * 100, "percent_change", wrong, later_first))
    return tasks


def share_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    combos = [(Scope("all"), Scope("category", c)) for c in facts.categories]
    combos += [(Scope("category", c), Scope("product", n)) for _, n, c, _ in facts.products]
    combos += [(Scope("all"), Scope("product", n)) for _, n, _, _ in facts.products]
    for metric in (METRIC["revenue"], METRIC["units"], METRIC["count"]):
        for whole, part in combos:
            for period in PERIODS:
                total, sub = facts.value(metric, whole, period), facts.value(metric, part, period)
                if not total or sub is None:
                    continue
                english = rng.random() < 0.15
                part_es = f"la categoría {part.value}" if part.kind == "category" else f"el producto {part.value}"
                whole_es = "de la tienda" if whole.kind == "all" else f"de la categoría {whole.value}"
                noun = rng.choice(metric.es)
                two_values = rng.random() < 0.35
                if english:
                    part_en = f"the {part.value} category" if part.kind == "category" else part.value
                    whole_en = "the store's" if whole.kind == "all" else f"the {whole.value} category's"
                    question = f"What share of {whole_en} {rng.choice(metric.en)} {period.en_in} came from {part_en}?"
                elif two_values and metric.key == "units":
                    question = f"¿Cuántas unidades {whole_es} se vendieron {period.es_in} y cuántas de esas corresponden a {part_es}?"
                elif two_values:
                    question = f"¿Cuál fue {noun} {whole_es} {period.es_in}, y cuánto de eso corresponde a {part_es}?"
                else:
                    question = rng.choice([
                        f"¿Qué porcentaje de {noun} {whole_es} {period.es_in} corresponde a {part_es}?",
                        f"¿Qué parte, en porcentaje, de {noun} {whole_es} {period.es_in} aportó {part_es}?",
                    ])
                label = period.en_short if english else period.es_short
                periods = [] if period.whole else [_period_json(label, period)]
                steps = [_step(1, f"total {metric.key}{' of ' + whole.value if whole.value else ''}", scalar_sql(metric, whole, period)),
                         _step(2, f"{metric.key} of {part.value}", scalar_sql(metric, part, period))]
                needs = "the total, the part and the part's share" if two_values else "the part's share of the total"
                plan = plan_json(needs, periods, steps, "share_pct")
                wrong = [("parte_como_crecimiento", plan_json(needs, periods, steps, "growth_pct"))]
                if two_values:
                    wrong.append(("valores_descartados", plan_json(needs, periods, steps, "direct")))
                text = question if english else _accentless(question, rng)
                tasks.append(Task("participacion", _q(text), "en" if english else "es", plan, sub / total * 100, "share", wrong))
    return tasks


def compare_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    pairs = [(a, b) for a in PERIODS[:-1] for b in PERIODS[:-1] if a.key != b.key and (a.end < b.start or b.end < a.start)]
    for entity in ("product", "category"):
        for metric in (METRIC["revenue"], METRIC["units"], METRIC["count"]):
            for first, second in pairs:
                tops = []
                for period in (first, second):
                    ranking = facts.ranking(entity, metric, period)
                    if len(ranking) < 2 or ranking[0][1] == ranking[1][1]:
                        break
                    tops.append(ranking[0][0])
                if len(tops) < 2:
                    continue
                english = rng.random() < 0.15
                noun = "producto" if entity == "product" else "categoría"
                article = "el" if entity == "product" else "la"
                if english:
                    question = (f"Was the {entity} with {metric.rank_en} {first.en_in} the same as {second.en_in}?")
                else:
                    question = rng.choice([
                        f"¿{_cap(article)} {noun} con {metric.rank_es} {first.es_in} fue {'el mismo' if entity == 'product' else 'la misma'} que {second.es_in}?",
                        f"¿Qué {noun} tuvo {metric.rank_es} {first.es_in} y cuál {second.es_in}? ¿Es {'el mismo' if entity == 'product' else 'la misma'}?",
                        f"¿Cambió {article} {noun} líder en {metric.rank_es.replace('más ', '')} entre {first.es_short} y {second.es_short}?",
                    ])
                labels = (first.en_short, second.en_short) if english else (first.es_short, second.es_short)
                periods = [_period_json(labels[0], first), _period_json(labels[1], second)]
                steps = [_step(1, f"top {entity} by {metric.key} in {first.en_short}", top_sql(entity, metric, first)),
                         _step(2, f"top {entity} by {metric.key} in {second.en_short}", top_sql(entity, metric, second))]
                plan = plan_json(f"the top {entity} of each period and whether it is the same", periods, steps, "compare_equal")
                column = "p.name" if entity == "product" else "p.category"
                as_growth = [_step(i + 1, s["purpose"], s["sql"].replace(f"SELECT {column}", f"SELECT {column}, {metric.expr}"))
                             for i, s in enumerate(steps)]
                wrong = [("comparacion_como_crecimiento", plan_json(plan["answer_needs"], periods, as_growth, "growth_pct"))]
                text = question if english else _accentless(question, rng)
                tasks.append(Task("comparacion", _q(text), "en" if english else "es", plan,
                                  {"first": tops[0], "second": tops[1]}, "same_or_changed", wrong))
    return tasks


def filter_rank_tasks(facts: Facts, rng: random.Random) -> list[Task]:
    tasks = []
    filters = [
        ("stock bajo su punto de reorden", "whose stock is below the reorder point",
         "SELECT p.product_id FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.stock < i.reorder_point",
         {pid for pid, (s, r) in facts.inventory.items() if s < r}),
        ("stock igual o sobre su punto de reorden", "whose stock is at or above the reorder point",
         "SELECT p.product_id FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.stock >= i.reorder_point",
         {pid for pid, (s, r) in facts.inventory.items() if s >= r}),
    ]
    for threshold in (10000, 20000, 30000):
        filters.append((f"precio menor a {threshold} pesos", f"priced below {threshold} pesos",
                        f"SELECT product_id FROM products WHERE price < {threshold}",
                        {pid for pid, p in facts.price.items() if p < threshold}))
    for es, en, filter_sql, members in filters:
        for metric in (METRIC["revenue"], METRIC["units"], METRIC["count"]):
            for period in PERIODS:
                ranking = facts.ranking("product", metric, period, products=members)
                if len(ranking) < 2 or ranking[0][1] == ranking[1][1]:
                    continue
                english = rng.random() < 0.15
                if english:
                    question = f"Among the products {en}, which one had {metric.rank_en} {period.en_in}?"
                else:
                    question = rng.choice([
                        f"De los productos con {es}, ¿cuál tuvo {metric.rank_es} {period.es_in}?",
                        f"Entre los productos con {es}, ¿cuál fue el que tuvo {metric.rank_es} {period.es_in}?",
                    ])
                label = period.en_short if english else period.es_short
                rank = (f"SELECT p.name {JOIN}{_where(_dates(period))} GROUP BY p.product_id ORDER BY {metric.expr} DESC")
                steps = [_step(1, f"ids of products {en}", filter_sql), _step(2, f"products ranked by {metric.key}", rank)]
                plan = plan_json("one product name", [] if period.whole else [_period_json(label, period)], steps, "filter_then_rank")
                text = question if english else _accentless(question, rng)
                tasks.append(Task("filtro_ranking", _q(text), "en" if english else "es", plan, ranking[0][0], "name"))
    return tasks


FAMILIES = (scalar_tasks, top_tasks, lookup_tasks, growth_tasks, share_tasks, compare_tasks, filter_rank_tasks)


# --------------------------------------------------------------------------- execution and checks


def run_plan(conn: sqlite3.Connection, plan: dict[str, Any]) -> tuple[dict[str, Any], list, ComposedAnswer, list[str]]:
    """Execute a plan exactly as the Analyst's tools do (no model involved)."""

    canonical = validate_plan(plan)
    expanded = expand_step_references(canonical["steps"])
    results = [execute_read_only(conn, sql).rows for sql in expanded]
    composed = compose_results(canonical["operation"], results, conn=conn)
    return canonical, results, composed, expanded


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return a == b


def matches_reference(task: Task, composed: ComposedAnswer) -> bool:
    value = composed.value
    if task.family in ("crecimiento",):
        return _close(value["growth_pct"], task.reference)
    if task.family == "participacion":
        return _close(value["share_pct"], task.reference)
    if task.family == "comparacion":
        return value["first"] == task.reference["first"] and value["second"] == task.reference["second"]
    if task.family == "filtro_ranking":
        return value["winner"] == task.reference
    if task.family == "directa_lista":
        return sorted(row[0] for row in value) == sorted(task.reference)
    return _close(value, task.reference)


def _norm_value(v: Any) -> Any:
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 6)
    return v


def row_set(rows) -> frozenset:
    return frozenset(tuple(_norm_value(v) for v in row) for row in rows)


def _numeric(rows) -> bool:
    return any(isinstance(v, (int, float)) and not isinstance(v, bool) for row in rows for v in row)


def ngrams(text: str, n: int = 5) -> set[str]:
    text = f" {normalize_text(text)} "
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


@dataclass
class EvalIndex:
    """What the evaluation questions compute and say, to keep the training data apart from them."""

    step_sets: set[frozenset]
    task_signatures: set[tuple]
    grams: list[tuple[str, set[str]]]

    @classmethod
    def build(cls, question_files: list[Path]) -> "EvalIndex":
        step_sets, signatures, grams = set(), set(), []
        for path in question_files:
            for question in json.loads(Path(path).read_text(encoding="utf-8"))["questions"]:
                sets = [row_set(rows) for rows in question["gold_result"]]
                step_sets |= {s for s, rows in zip(sets, question["gold_result"]) if _numeric(rows)}
                signatures.add((question.get("combine", "direct"), frozenset(sets)))
                grams.append((question["id"], ngrams(question["question"])))
        return cls(step_sets, signatures, grams)

    def overlap(self, operation: str, results: list) -> str | None:
        sets = [row_set(rows) for rows in results]
        if (operation, frozenset(sets)) in self.task_signatures:
            return "misma_tarea"
        if any(s in self.step_sets for s, rows in zip(sets, results) if _numeric(rows)):
            return "mismo_resultado_numerico"
        return None

    def closest(self, question: str) -> tuple[str, float]:
        mine = ngrams(question)
        return max(((qid, jaccard(mine, other)) for qid, other in self.grams), key=lambda kv: kv[1])


# --------------------------------------------------------------------------- building the dataset


def _compact(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False)


def _split_key(task: Task) -> str:
    """Train/dev split by computation, so the dev set has computations never seen in training."""

    return hashlib.sha256(json.dumps([s["sql"] for s in task.plan["steps"]] + [task.plan["composition"]], sort_keys=True).encode()).hexdigest()


def build_dataset(
    db_path: str | Path,
    eval_files: list[Path],
    out_dir: str | Path,
    plan_examples: int = 900,
    repair_examples: int = 400,
    dev_fraction: float = 0.1,
    review_sample: int = 60,
    seed: int = SEED,
) -> dict[str, Any]:
    rng = random.Random(seed)
    facts = Facts(db_path)
    system = AgentSystem(db_path, generator=None)  # only for the Planner prompt and the Verificador checks
    index = EvalIndex.build(eval_files)
    conn = connect_read_only(db_path)
    removed = {"referencia_distinta": 0, "misma_tarea": 0, "mismo_resultado_numerico": 0, "pregunta_similar": 0}
    kept: list[tuple[Task, list]] = []
    seen_questions: set[str] = set()
    try:
        for family in FAMILIES:
            for task in family(facts, rng):
                _, results, composed, _ = run_plan(conn, task.plan)
                if not matches_reference(task, composed):
                    removed["referencia_distinta"] += 1
                    continue
                reason = index.overlap(task.plan["composition"]["operation"], results)
                if reason:
                    removed[reason] += 1
                    continue
                if index.closest(task.question)[1] >= JACCARD_LIMIT:
                    removed["pregunta_similar"] += 1
                    continue
                if normalize_text(task.question) in seen_questions:
                    continue
                seen_questions.add(normalize_text(task.question))
                kept.append((task, results))

        # Balanced sample of plan examples: every family is represented, combined questions weigh more.
        weights = {"directa_valor": 0.22, "directa_lider": 0.10, "directa_ficha": 0.04, "directa_lista": 0.02,
                   "crecimiento": 0.22, "participacion": 0.16, "comparacion": 0.14, "filtro_ranking": 0.10}
        by_family: dict[str, list] = {}
        for item in kept:
            by_family.setdefault(item[0].family, []).append(item)
        chosen = []
        for family, items in by_family.items():
            rng.shuffle(items)
            chosen += items[: max(1, round(weights[family] * plan_examples))]
        rng.shuffle(chosen)

        examples = []
        for task, _ in chosen:
            examples.append({
                "kind": "plan", "family": task.family, "language": task.language, "later_first": task.later_first,
                "question": task.question, "prompt": system.planner.build_prompt(task.question),
                "completion": _compact(task.plan), "split_key": _split_key(task),
            })

        # Repair examples: the wrong plan is executed and judged by the Verificador's own checks; only
        # wrong plans that the checks catch become examples (the Planner never sees others at run time).
        repairs = []
        candidates = [(task, kind, wrong) for task, _ in kept for kind, wrong in task.wrong_plans]
        rng.shuffle(candidates)
        quota: dict[str, int] = {}
        per_kind = max(1, repair_examples // 7)
        for task, kind, wrong in candidates:
            if len(repairs) >= repair_examples:
                break
            if quota.get(kind, 0) >= per_kind:
                continue
            try:
                canonical, results, composed, executed = run_plan(conn, wrong)
            except Exception:  # noqa: BLE001 - a wrong plan that does not run is not a Verificador case
                continue
            analysis = {"asks_for": task.asks_for, "periods": task.plan["periods"]}
            problems, _ = critic_problems(task.question, analysis, composed, canonical["steps"],
                                          executed, system.date_range, system.labels)
            if not problems:
                continue
            feedback = " ".join(problems)
            previous = {"steps": canonical["steps"], "operation": canonical["operation"]}
            quota[kind] = quota.get(kind, 0) + 1
            repairs.append({
                "kind": "repair", "repair": kind, "family": task.family, "language": task.language,
                "later_first": task.later_first, "question": task.question, "feedback": feedback,
                "prompt": system.planner.build_prompt(task.question, feedback, previous),
                "completion": _compact(task.plan), "split_key": _split_key(task),
            })
        examples += repairs
        rng.shuffle(examples)
    finally:
        conn.close()

    # Split by computation: every example of a computation goes to the same side.
    keys = sorted({e["split_key"] for e in examples})
    rng.shuffle(keys)
    dev_keys = set(keys[: max(1, int(len(keys) * dev_fraction))])
    train = [e for e in examples if e["split_key"] not in dev_keys]
    dev = [e for e in examples if e["split_key"] in dev_keys]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review = rng.sample(examples, min(review_sample, len(examples)))
    with (out / "revision_manual.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "tipo", "familia", "pregunta", "retroalimentacion", "plan_correcto",
                         "revisor", "plan_correcto_si_no", "comentario"])
        for n, row in enumerate(review, start=1):
            writer.writerow([n, row["kind"], row["family"], row["question"], row.get("feedback", ""), row["completion"], "", "", ""])

    max_similarity = max(index.closest(e["question"]) for e in examples)
    report = {
        "seed": seed,
        "source": "plantillas de preguntas sobre data/business.db; planes construidos por código (sin modelo generador)",
        "candidate_tasks": len(kept) + sum(removed.values()),
        "removed": removed,
        "kept_tasks": len(kept),
        "examples": len(examples),
        "train": len(train),
        "dev": len(dev),
        "by_kind": _count(examples, "kind"),
        "by_family": _count(examples, "family"),
        "by_repair": _count([e for e in examples if e["kind"] == "repair"], "repair"),
        "by_language": _count(examples, "language"),
        "growth_later_first": sum(1 for e in examples if e["family"] == "crecimiento" and e["later_first"]),
        "growth_total": sum(1 for e in examples if e["family"] == "crecimiento"),
        "max_question_similarity_to_eval": {"question_id": max_similarity[0], "jaccard_5gram": round(max_similarity[1], 3)},
        "review_sample": len(review),
    }
    (out / "reporte.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
    return dict(sorted(counts.items()))
