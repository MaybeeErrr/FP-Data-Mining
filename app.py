"""
app.py
======
Antarmuka Streamlit untuk sistem Multi-Agentic LLM PT Retailindo Nusantara.
Menjalankan `multi_agent_graph.invoke()` (lewat agent_core.MultiAgentSystem)
dan menampilkan hasil + log interaksi antar-agent + evaluasi (Soal 4).

Cara jalan:
    streamlit run app.py

GROQ_API_KEY diambil dari (urutan prioritas):
    1. st.secrets["GROQ_API_KEY"]  (disarankan untuk deploy: Streamlit Cloud / HF Spaces "secrets")
    2. Environment variable GROQ_API_KEY
    3. Input manual di sidebar (kalau 2 sumber di atas kosong)
"""

import os
import time

import streamlit as st

from agent_core import AVAILABLE_AGENTS, DIVISI_LABEL, MultiAgentSystem

st.set_page_config(page_title="Multi-Agent PT Retailindo Nusantara", page_icon="🏬", layout="wide")


# ============================================================
# Sidebar: API Key & status sistem
# ============================================================
def get_groq_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    return key


with st.sidebar:
    st.title("⚙️ Konfigurasi")
    default_key = get_groq_api_key()
    groq_api_key = st.text_input(
        "GROQ_API_KEY",
        value=default_key,
        type="password",
        help="Dapatkan API key gratis di https://console.groq.com/keys",
    )

    csv_dir = st.text_input(
        "Folder dataset tiket (opsional)",
        value="",
        help="Path folder berisi CSV dataset tiket publik (Kaggle). Kosongkan kalau tidak punya "
             "-- sistem tetap jalan dengan SOP retail sintetis + dokumen profil perusahaan.",
    )

    st.divider()
    st.markdown("**Divisi yang tersedia:**")
    for a in AVAILABLE_AGENTS:
        st.markdown(f"- {DIVISI_LABEL[a]}")

    st.divider()
    show_eval = st.checkbox("Tampilkan Evaluator Agent (Soal 4)", value=True)


# ============================================================
# Inisialisasi sistem (cached supaya vector DB tidak dibangun ulang tiap interaksi)
# ============================================================
@st.cache_resource(show_spinner="Membangun knowledge base, vector DB, dan graph multi-agent...")
def load_system(api_key: str, csv_dir: str):
    return MultiAgentSystem(groq_api_key=api_key, csv_dir=csv_dir or None)


st.title("🏬 Multi-Agentic LLM System — PT Retailindo Nusantara")
st.caption(
    "Orchestrator akan merutekan permintaan ke satu/lebih agent divisi (RAG + tool calling); "
    "agent divisi bisa saling berkonsultasi (mis. Inventory → Finance saat stok habis)."
)

if not groq_api_key:
    st.warning("Masukkan GROQ_API_KEY di sidebar untuk memulai (gratis di console.groq.com/keys).")
    st.stop()

try:
    system = load_system(groq_api_key, csv_dir)
except Exception as e:
    st.error(f"Gagal menginisialisasi sistem: {e}")
    st.stop()

st.success("Sistem siap. Silakan masukkan permintaan/keluhan di bawah.")


# ============================================================
# Input & eksekusi
# ============================================================
contoh_query = "Pelanggan komplain karena Sepatu Lari X1 habis stoknya di cabang Yogyakarta, mohon solusinya"

query = st.text_area(
    "Masukkan permintaan / keluhan",
    placeholder=contoh_query,
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("🚀 Jalankan", type="primary", use_container_width=True)
with col2:
    if st.button("Gunakan contoh skenario"):
        query = contoh_query
        run_clicked = True

if run_clicked:
    if not query.strip():
        st.warning("Tulis permintaan terlebih dahulu.")
        st.stop()

    with st.spinner("Menjalankan Orchestrator → Agent Divisi → Aggregator..."):
        try:
            result = system.run(query)
        except Exception as e:
            st.error(f"Eksekusi gagal: {e}")
            st.stop()

    st.divider()
    st.subheader("📋 Hasil")

    st.markdown(f"**Query:** {query}")
    st.markdown(f"**Routing Orchestrator:** {', '.join(DIVISI_LABEL[d] for d in result['route'])}")

    tabs = st.tabs(["Jawaban Tiap Agent", "Log Interaksi Antar-Agent", "Rekomendasi Akhir"])

    with tabs[0]:
        for divisi, jawaban in result["outputs"].items():
            with st.expander(f"🧩 {DIVISI_LABEL[divisi]}", expanded=True):
                st.write(jawaban)
                sources = result["sources"].get(divisi, [])
                if sources:
                    st.caption("Sumber RAG yang dipakai:")
                    for s in sources:
                        st.caption(f"- {s}")

    with tabs[1]:
        if result.get("interaction_log"):
            for log in result["interaction_log"]:
                st.info(log)
        else:
            st.write("(Tidak ada konsultasi antar-agent pada permintaan ini)")

    with tabs[2]:
        st.write(result.get("final_answer", "(tidak ada rekomendasi akhir)"))

    st.caption(f"⏱️ Latensi eksekusi: {result.get('_latency', 0):.2f} detik")

    # ---------------- Evaluator Agent ----------------
    if show_eval:
        st.divider()
        st.subheader("📊 Evaluator Agent (Soal 4)")
        with st.spinner("Menjalankan evaluasi (Accuracy, Effectiveness, Efficiency, Explainability, Hallucination)..."):
            try:
                eval_result = system.evaluate(query, result)
            except Exception as e:
                st.error(f"Evaluasi gagal: {e}")
                eval_result = None

        if eval_result:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Effectiveness**")
                st.json(eval_result["effectiveness"])
                st.markdown("**Efficiency**")
                st.json(eval_result["efficiency"])
            with c2:
                st.markdown("**Explainability**")
                st.json(eval_result["explainability"])

            st.markdown("**Accuracy / Hallucination** (faithfulness score terhadap konteks RAG + tool)")
            rows = []
            for divisi, v in eval_result["accuracy_hallucination"].items():
                rows.append({
                    "Agent": DIVISI_LABEL[divisi],
                    "Faithfulness Score": v["faithfulness_score"],
                    "Indikasi Hallucination": v["hallucination_flag"],
                })
            st.table(rows)
