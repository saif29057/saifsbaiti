"""
chat.py
Chat RAG sur la collection Qdrant "invoices" avec recherche par filtres de metadata.

Pipeline :
  1. Le LLM (Groq) transforme la question en JSON : filtres metadata + requête sémantique.
  2. Les filtres sont validés/normalisés (noms de champs, valeurs exactes, opérateurs).
  3. Qdrant : scroll(filtre) pour les statistiques exactes + recherche vectorielle filtrée (top-k).
  4. Le LLM répond à partir des factures trouvées.

Usage :
    python chat.py            # mode terminal
    python chat.py --debug    # affiche les filtres générés
    streamlit run app.py      # interface web
"""

import argparse
import difflib
import json
import os
import re
import sys
from functools import lru_cache

from groq import Groq

import qdrant_store as store

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 10

STRING_FIELDS = ["invoice_id", "order_id", "bill_to", "ship_to", "city", "state", "country",
                 "postal_code", "ship_mode", "item", "category", "sub_category",
                 "feature_code", "feature_prefix", "date"]
NUMBER_FIELDS = ["year", "month", "quantity", "rate", "amount", "subtotal",
                 "discount", "discount_pct", "shipping", "total", "balance_due"]
INT_FIELDS = {"year", "month", "quantity"}
# champs pour lesquels on donne la liste des valeurs possibles au LLM
ENUM_FIELDS = ["country", "category", "sub_category", "ship_mode", "feature_prefix", "year"]
OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}

_groq = None


def groq_client():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


@lru_cache(maxsize=1)
def distinct_values():
    """Valeurs distinctes de chaque champ texte (pour normaliser les filtres et l'interface)."""
    metas = [p.payload for p in store.scroll_all()]
    values = {f: sorted({m[f] for m in metas if m.get(f) not in ("", None)}, key=str)
              for f in STRING_FIELDS + ["year"]}
    values["hierarchy"] = {c: sorted({m["sub_category"] for m in metas if m["category"] == c})
                           for c in values["category"]}
    return values


@lru_cache(maxsize=1)
def filter_prompt():
    values = distinct_values()
    return f"""Tu convertis une question sur des factures en filtres de metadata.
Champs texte : {", ".join(STRING_FIELDS)}
Champs numériques : {", ".join(NUMBER_FIELDS)}
Valeurs possibles :
{chr(10).join(f"- {f}: {values[f]}" for f in ENUM_FIELDS)}
- sous-catégories par catégorie : {values["hierarchy"]}
- feature_prefix : FUR=Furniture, OFF=Office Supplies, TEC=Technology
- date au format YYYY-MM-DD ; bill_to = nom du client ; feature_code ex: FUR-CH-4421

Réponds UNIQUEMENT en JSON :
{{"filters": [{{"field": "<champ>", "op": "<$eq|$ne|$gt|$gte|$lt|$lte|$in|$nin>", "value": <valeur>}}],
  "semantic_query": "<texte EN ANGLAIS pour la recherche sémantique (produit, description), ou chaîne vide>",
  "sort_by": "<champ numérique ou null>", "sort_desc": true}}
Règles : n'ajoute un filtre que si la question l'exige explicitement (client, pays, date, montant, code...).
Un produit décrit en langage naturel (ex : "chaise en cuir") va dans semantic_query, avec au plus
un filtre sub_category cohérent avec la hiérarchie ci-dessus ; ne combine jamais des valeurs contradictoires.
"pas cher" / "le plus cher" -> sort_by total (sort_desc false / true). Pour $in/$nin, value est une liste.
Pour une période (ex : année 2012) utilise le champ year ; pour un mois, year + month."""


def ask_llm(messages, json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = groq_client().chat.completions.create(model=MODEL, messages=messages,
                                                 temperature=0, **kwargs)
    return resp.choices[0].message.content


def _normalize_value(field, value):
    """Ramène une valeur texte à une valeur existante (casse, fautes de frappe)."""
    if field in NUMBER_FIELDS:
        return int(value) if field in INT_FIELDS else float(value)
    value = str(value)
    lower = {str(k).lower(): k for k in distinct_values().get(field, [])}
    if value.lower() in lower:
        return lower[value.lower()]
    close = difflib.get_close_matches(value.lower(), list(lower), n=1, cutoff=0.8)
    return lower[close[0]] if close else value


def clean_filters(filters):
    """Valide les filtres du LLM -> liste [{field, op, value}] utilisable par Qdrant."""
    clauses = []
    for f in filters or []:
        field, op, value = f.get("field"), f.get("op", "$eq"), f.get("value")
        if field not in STRING_FIELDS + NUMBER_FIELDS or op not in OPERATORS or value is None:
            continue
        if field in STRING_FIELDS and op in ("$gt", "$gte", "$lt", "$lte"):
            continue  # Qdrant : les plages ne s'appliquent qu'aux nombres
        try:
            if op in ("$in", "$nin"):
                value = [_normalize_value(field, v) for v in (value if isinstance(value, list) else [value])]
            else:
                value = _normalize_value(field, value)
        except (TypeError, ValueError):
            continue
        clauses.append({"field": field, "op": op, "value": value})
    return clauses


def parse_question(question):
    raw = ask_llm([{"role": "system", "content": filter_prompt()},
                   {"role": "user", "content": question}], json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def compute_stats(metas):
    n = len(metas)
    total = sum(m["total"] for m in metas)
    return {
        "nb_factures": n,
        "somme_total": round(total, 2),
        "somme_amount": round(sum(m["amount"] for m in metas), 2),
        "somme_shipping": round(sum(m["shipping"] for m in metas), 2),
        "somme_discount": round(sum(m["discount"] for m in metas), 2),
        "moyenne_total": round(total / n, 2) if n else 0,
    }


def search(question, extra_filters=None):
    """extra_filters : filtres imposés (ex : barre latérale de l'interface), ajoutés à ceux du LLM."""
    plan = parse_question(question)
    clauses = clean_filters(plan.get("filters")) + list(extra_filters or [])
    qfilter = store.to_qdrant_filter(clauses)
    semantic = (plan.get("semantic_query") or "").strip()

    # Toutes les factures qui respectent les filtres -> stats exactes
    matched = [p.payload for p in store.scroll_all(qfilter)] if qfilter else None
    relaxed = False
    if matched == [] and semantic:
        # filtres trop stricts : on retombe sur la recherche sémantique seule
        clauses, relaxed = list(extra_filters or []), True
        qfilter = store.to_qdrant_filter(clauses)
        matched = [p.payload for p in store.scroll_all(qfilter)] if qfilter else None

    sort_by = plan.get("sort_by")
    if semantic or not qfilter:
        # avec un tri ("le moins cher"...) on prend plus de candidats sémantiques avant de trier
        limit = TOP_K * 3 if sort_by in NUMBER_FIELDS else TOP_K
        payloads = [p.payload for p in store.search(semantic or question, qfilter, limit)]
    else:
        payloads = matched

    if sort_by in NUMBER_FIELDS:
        payloads = sorted(payloads, key=lambda m: m.get(sort_by, 0),
                          reverse=plan.get("sort_desc", True))

    return {
        "filters": clauses,
        "semantic_query": semantic,
        "sort_by": sort_by if sort_by in NUMBER_FIELDS else None,
        "relaxed": relaxed,
        "stats": compute_stats(matched) if matched is not None else None,
        "results": payloads[:TOP_K * 3],
    }


def answer(question, history=None, extra_filters=None):
    history = history if history is not None else []
    found = search(question, extra_filters)
    context = "\n\n".join(f"[{m['source']}]\n{m['document']}" for m in found["results"]) \
        or "Aucune facture trouvée."
    stats_txt = (f"Statistiques exactes sur TOUTES les factures correspondant aux filtres "
                 f"{found['filters']} : {json.dumps(found['stats'], ensure_ascii=False)}"
                 ) if found["stats"] else ""
    if found["semantic_query"]:
        stats_txt += ("\nLes factures ci-dessous sont les plus proches sémantiquement de la demande "
                      f"(triées par {found['sort_by']}) ; vérifie qu'elles correspondent au produit demandé.")
    if found["relaxed"]:
        stats_txt += "\nAucune facture ne respectait les filtres détectés : recherche élargie sans filtre."
    system = ("Tu es un assistant qui répond aux questions sur des factures SuperStore. "
              "Base-toi uniquement sur les factures et statistiques fournies. Pour les comptes et "
              "les sommes, utilise les statistiques exactes (et non la liste d'exemples). "
              "Cite les numéros de facture. Réponds dans la langue de la question.")
    user = f"{stats_txt}\n\nFactures pertinentes :\n{context}\n\nQuestion : {question}"
    reply = ask_llm([{"role": "system", "content": system}, *history[-6:],
                     {"role": "user", "content": user}])
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    return {**found, "reply": reply}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="affiche les filtres générés")
    debug = parser.parse_args().debug
    print(f"Chat factures Qdrant ({store.count()} factures, modèle {MODEL}). 'exit' pour quitter.\n")
    history = []
    while True:
        try:
            q = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        try:
            res = answer(q, history)
            if debug:
                print(f"[filtres] {json.dumps(res['filters'], ensure_ascii=False)}")
                print(f"[sémantique] {res['semantic_query']!r}  [tri] {res['sort_by']}")
            print(f"\nAssistant > {res['reply']}\n")
        except Exception as e:
            print(f"[erreur] {e}\n")


if __name__ == "__main__":
    main()
