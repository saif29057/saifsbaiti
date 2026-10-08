"""
ingestion.py
Lit tous les PDF du dossier, extrait texte + metadata, et les indexe
dans la collection Qdrant "invoices".

Usage :
    python ingestion.py            # ingestion (upsert)
    python ingestion.py --reset    # supprime et recrée la collection
"""

import argparse
import sys
import uuid

from qdrant_client import models

from metadata_extractor import process_pdf
from qdrant_store import BASE_DIR, COLLECTION_NAME, count, create_collection, embed

sys.stdout.reconfigure(encoding="utf-8")

PDF_DIR = BASE_DIR / "1000+ PDF_Invoice_Folder"
BATCH_SIZE = 100


def ingest(reset=False):
    client = create_collection(reset)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDF trouvés dans {PDF_DIR}")

    docs, metas = [], []
    skipped = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            doc, meta = process_pdf(pdf)
        except Exception as e:  # PDF corrompu
            skipped.append((pdf.name, str(e)))
            continue
        if not meta["is_valid"]:
            skipped.append((pdf.name, "facture vide / template"))
            continue
        docs.append(doc)
        metas.append(meta)
        if i % 200 == 0:
            print(f"  extraction {i}/{len(pdfs)}")

    for start in range(0, len(docs), BATCH_SIZE):
        end = start + BATCH_SIZE
        vectors = embed(docs[start:end])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, meta["source"])),  # id stable = nom du fichier
                vector=vec,
                payload={**meta, "document": doc},
            )
            for doc, meta, vec in zip(docs[start:end], metas[start:end], vectors)
        ]
        client.upsert(COLLECTION_NAME, points=points)
        print(f"  indexé {min(end, len(docs))}/{len(docs)}")

    print(f"\nTerminé : {len(docs)} factures dans '{COLLECTION_NAME}' "
          f"(total collection : {count()}), {len(skipped)} ignorées.")
    for name, reason in skipped:
        print(f"  - {name} : {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="recréer la collection")
    ingest(parser.parse_args().reset)
