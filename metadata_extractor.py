"""
metadata_extractor.py
Extraction du texte + des metadata structurées depuis une facture PDF SuperStore.

Metadata extraites :
    invoice_id, order_id, date, year, month, ship_mode,
    bill_to, ship_to, city, state, country, postal_code,
    item, quantity, rate, category, sub_category,
    feature_code (ex: FUR-CH-4421), feature_prefix (FUR / OFF / TEC),
    subtotal, discount, discount_pct, shipping, total, amount, balance_due
"""

import re
from datetime import datetime
from pathlib import Path

from PyPDF2 import PdfReader

MONEY = r"\$?-?[\d,]+\.\d{2}"
CODE_RE = re.compile(r"\b([A-Z]{3})-([A-Z]{2})-(\d+)\b")


def _money(s):
    return float(s.replace("$", "").replace(",", "")) if s else 0.0


def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _between(lines, start, end):
    """Retourne les lignes entre le label `start` et le label `end`."""
    try:
        i = lines.index(start)
        j = lines.index(end, i + 1)
    except ValueError:
        return []
    return [l for l in lines[i + 1:j] if l != ":"]


def extract_metadata(text, filename=""):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meta = {"source": filename}

    # --- identifiants -----------------------------------------------------
    m = re.search(r"#\s*(\d+)", text)
    meta["invoice_id"] = m.group(1) if m else Path(filename).stem.split("_")[-1]
    m = re.search(r"Order ID\s*:\s*(\S+)", text)
    meta["order_id"] = m.group(1) if m else ""

    # --- Bill To ----------------------------------------------------------
    bill = _between(lines, "Bill To", "Ship To")
    meta["bill_to"] = " ".join(bill)
    if not meta["bill_to"]:
        # fallback sur le nom du fichier : invoice_<Nom>_<id>.pdf
        parts = Path(filename).stem.split("_")
        meta["bill_to"] = parts[1] if len(parts) >= 3 else ""

    # --- Ship To + date + ship mode + balance -----------------------------
    # Layout : Ship To : <adresse...> <Date> <ShipMode> <$Balance> Date : ...
    block = _between(lines, "Ship To", "Date")
    date_idx = next((k for k, l in enumerate(block)
                     if re.match(r"^[A-Z][a-z]{2} \d{1,2},? \d{4}$", l)), None)
    if date_idx is not None:
        addr = " ".join(block[:date_idx])
        meta["ship_to"] = re.sub(r"\s*,\s*", ", ", addr).strip(", ")
        raw_date = block[date_idx].replace(",", "")
        rest = block[date_idx + 1:]
    else:
        meta["ship_to"] = ""
        raw_date, rest = "", block

    try:
        d = datetime.strptime(raw_date, "%b %d %Y")
        meta["date"], meta["year"], meta["month"] = d.strftime("%Y-%m-%d"), d.year, d.month
    except ValueError:
        meta["date"], meta["year"], meta["month"] = "", 0, 0

    money_rest = [l for l in rest if re.fullmatch(MONEY, l)]
    meta["ship_mode"] = next((l for l in rest if not re.fullmatch(MONEY, l)), "")
    meta["balance_due"] = _money(money_rest[0]) if money_rest else 0.0

    # Adresse : "postal, city, state, country" (postal optionnel)
    addr_parts = [p.strip() for p in meta["ship_to"].split(",") if p.strip()]
    meta["postal_code"] = addr_parts.pop(0) if addr_parts and addr_parts[0].isdigit() else ""
    meta["city"] = addr_parts[0] if len(addr_parts) > 0 else ""
    meta["state"] = addr_parts[1] if len(addr_parts) > 1 else ""
    meta["country"] = addr_parts[-1] if len(addr_parts) > 2 else ""

    # --- Ligne article + feature code -------------------------------------
    meta.update(item="", quantity=0, rate=0.0, amount=0.0,
                category="", sub_category="", feature_code="", feature_prefix="")
    code_idx = next((k for k, l in enumerate(lines) if CODE_RE.search(l)), None)
    if code_idx is not None:
        cm = CODE_RE.search(lines[code_idx])
        meta["feature_code"] = cm.group(0)
        meta["feature_prefix"] = cm.group(1)
        # "Chairs, Furniture, FUR-CH-4421" -> sub_category, category
        cats = [c.strip() for c in lines[code_idx].split(",")]
        if len(cats) >= 3:
            meta["sub_category"], meta["category"] = cats[0], cats[1]
        # lignes précédentes : <item...> <qty> <$rate> <$amount>
        if code_idx >= 4 and re.fullmatch(MONEY, lines[code_idx - 1]):
            meta["amount"] = _money(lines[code_idx - 1])
            meta["rate"] = _money(lines[code_idx - 2])
            meta["quantity"] = int(lines[code_idx - 3]) if lines[code_idx - 3].isdigit() else 0
            start = lines.index("Amount") + 1 if "Amount" in lines else code_idx - 4
            meta["item"] = " ".join(lines[start:code_idx - 3])

    # --- Totaux : valeurs puis labels (Subtotal, [Discount], Shipping, Total)
    meta.update(subtotal=0.0, discount=0.0, discount_pct=0.0, shipping=0.0, total=0.0)
    if code_idx is not None and "Subtotal" in lines:
        values = [l for l in lines[code_idx + 1:lines.index("Subtotal")] if re.fullmatch(MONEY, l)]
        labels = []
        for l in lines[lines.index("Subtotal"):]:
            if l == ":":
                continue
            if l.startswith(("Subtotal", "Discount", "Shipping", "Total")):
                labels.append(l)
            elif labels and l.startswith("Notes"):
                break
        for label, val in zip(labels, values):
            if label.startswith("Subtotal"):
                meta["subtotal"] = _money(val)
            elif label.startswith("Discount"):
                meta["discount"] = _money(val)
                p = re.search(r"(\d+(?:\.\d+)?)%", label)
                meta["discount_pct"] = float(p.group(1)) if p else 0.0
            elif label.startswith("Shipping"):
                meta["shipping"] = _money(val)
            elif label.startswith("Total"):
                meta["total"] = _money(val)

    meta["is_valid"] = bool(meta["feature_code"] and meta["total"])
    return meta


def build_document(meta, raw_text):
    """Texte propre et lisible, utilisé pour l'embedding."""
    if not meta["is_valid"]:
        return raw_text
    return (
        f"Invoice #{meta['invoice_id']} (Order {meta['order_id']}) dated {meta['date']}.\n"
        f"Bill to: {meta['bill_to']}. Ship to: {meta['ship_to']}. Ship mode: {meta['ship_mode']}.\n"
        f"Item: {meta['item']} | Category: {meta['category']} / {meta['sub_category']} | "
        f"Feature code: {meta['feature_code']}.\n"
        f"Quantity: {meta['quantity']} x ${meta['rate']:.2f} = amount ${meta['amount']:.2f}.\n"
        f"Subtotal: ${meta['subtotal']:.2f}, discount: ${meta['discount']:.2f} ({meta['discount_pct']:g}%), "
        f"shipping: ${meta['shipping']:.2f}, total: ${meta['total']:.2f}."
    )


def process_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    text = extract_text(pdf_path)
    meta = extract_metadata(text, pdf_path.name)
    return build_document(meta, text), meta


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else next((Path(__file__).parent / "1000+ PDF_Invoice_Folder").glob("*.pdf"))
    doc, meta = process_pdf(target)
    print(doc, "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
