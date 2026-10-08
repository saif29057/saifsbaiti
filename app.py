"""
app.py
Petite interface web (Streamlit) pour le chat factures sur Qdrant.

Lancement :
    streamlit run app.py
"""

# qdrant_store (fastembed -> onnxruntime) doit être importé AVANT pandas :
# sous Windows, l'ordre inverse provoque un crash (conflit de DLL).
import qdrant_store as store  # noqa: I001
import chat

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Invoices Chat", page_icon="🧾", layout="wide")

TABLE_COLS = ["invoice_id", "date", "bill_to", "country", "city", "category", "sub_category",
              "feature_code", "item", "quantity", "amount", "discount", "shipping", "total"]


def to_df(payloads):
    df = pd.DataFrame(payloads)
    return df[[c for c in TABLE_COLS if c in df.columns]] if not df.empty else df


# --------------------------------------------------------------------------- sidebar
nb = store.count()
st.sidebar.title("🧾 Invoices")
st.sidebar.caption(f"Qdrant · collection **{store.COLLECTION_NAME}** · {nb} factures")

if nb == 0:
    st.warning("La collection est vide. Lance d'abord : `python ingestion.py --reset`")
    st.stop()

values = chat.distinct_values()
st.sidebar.subheader("Filtres")
f_country = st.sidebar.multiselect("Pays", values["country"])
f_category = st.sidebar.multiselect("Catégorie", values["category"])
f_ship = st.sidebar.multiselect("Ship mode", values["ship_mode"])
f_year = st.sidebar.multiselect("Année", values["year"])
f_client = st.sidebar.multiselect("Client (Bill To)", values["bill_to"])
f_total = st.sidebar.slider("Total ($)", 0, 80000, (0, 80000), step=100)

extra = []
for field, sel in [("country", f_country), ("category", f_category), ("ship_mode", f_ship),
                   ("year", f_year), ("bill_to", f_client)]:
    if sel:
        extra.append({"field": field, "op": "$in", "value": list(sel)})
if f_total[0] > 0:
    extra.append({"field": "total", "op": "$gte", "value": float(f_total[0])})
if f_total[1] < 80000:
    extra.append({"field": "total", "op": "$lte", "value": float(f_total[1])})

show_debug = st.sidebar.toggle("Afficher filtres & sources", value=True)
if st.sidebar.button("🗑️ Effacer la conversation"):
    st.session_state.pop("messages", None)
    st.session_state.pop("history", None)
    st.rerun()

# --------------------------------------------------------------------------- tabs
tab_chat, tab_search = st.tabs(["💬 Chat", "🔎 Recherche par filtres"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])

    if not st.session_state.messages:
        st.info("Exemples : *combien de factures pour Aaron Bergman et quel est le total ?* · "
                "*les 3 factures Technology en Germany avec le plus gros total en 2012* · "
                "*une chaise de bureau en cuir pas chère*")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("details") is not None and show_debug:
                d = msg["details"]
                with st.expander(f"Filtres & sources ({len(d['results'])} factures)"):
                    st.json({"filtres": d["filters"], "recherche_sémantique": d["semantic_query"],
                             "tri": d["sort_by"], "filtres_élargis": d["relaxed"],
                             "statistiques": d["stats"]})
                    st.dataframe(to_df(d["results"]), hide_index=True, width="stretch")

    if question := st.chat_input("Pose une question sur les factures…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Recherche dans Qdrant…"):
            try:
                res = chat.answer(question, st.session_state.history, extra_filters=extra)
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["reply"],
                     "details": {k: res[k] for k in ("filters", "semantic_query", "sort_by",
                                                     "relaxed", "stats", "results")}})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Erreur : {e}"})
        st.rerun()

with tab_search:
    st.caption("Recherche directe dans Qdrant avec les filtres de la barre latérale (sans LLM).")
    query = st.text_input("Recherche sémantique (optionnel, en anglais : ex. *leather office chair*)")
    qfilter = store.to_qdrant_filter(extra)
    if query:
        rows = [p.payload for p in store.search(query, qfilter, limit=50)]
    else:
        rows = [p.payload for p in store.scroll_all(qfilter)]

    stats = chat.compute_stats(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factures", stats["nb_factures"])
    c2.metric("Total", f"${stats['somme_total']:,.2f}")
    c3.metric("Moyenne", f"${stats['moyenne_total']:,.2f}")
    c4.metric("Remises", f"${stats['somme_discount']:,.2f}")

    df = to_df(rows)
    st.dataframe(df, hide_index=True, width="stretch")
    if not df.empty:
        st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"),
                           "invoices.csv", "text/csv")
