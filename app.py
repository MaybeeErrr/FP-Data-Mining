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

import html
import os
import time

import streamlit as st
import streamlit.components.v1 as components

from agent_core import AVAILABLE_AGENTS, DIVISI_LABEL, MultiAgentSystem

st.set_page_config(
    page_title="Control Tower — PT Retailindo Nusantara",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Design tokens & per-divisi identitas visual
# ============================================================
DIVISI_COLOR = {
    "marketing": "#A78BFA",
    "customer_service": "#60A5FA",
    "inventory": "#F2A93B",
    "hr": "#34D399",
    "finance": "#FB7185",
}
DIVISI_ICON = {
    "marketing": "📣",
    "customer_service": "💬",
    "inventory": "📦",
    "hr": "🧑‍💼",
    "finance": "💰",
}


def esc(text) -> str:
    """Escape teks (termasuk hasil LLM) sebelum dirender sebagai HTML mentah."""
    return html.escape(str(text)).replace("\n", "<br>")


# ============================================================
# Global styling
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0A0E17;
  --panel:#131A2B;
  --panel-2:#171F33;
  --border:#232C42;
  --text:#EDEFF5;
  --muted:#8A93AC;
  --amber:#F2A93B;
  --teal:#34D399;
  --rose:#FB7185;
  --blue:#60A5FA;
}

html, body, [class*="css"]{
  font-family:'Inter', sans-serif;
}
h1,h2,h3,h4{
  font-family:'Space Grotesk', sans-serif !important;
}
code, .mono{
  font-family:'IBM Plex Mono', monospace !important;
}

.stApp{
  background:
    radial-gradient(circle at 10% 0%, rgba(242,169,59,0.06), transparent 40%),
    radial-gradient(circle at 90% 10%, rgba(96,165,250,0.06), transparent 35%),
    var(--bg);
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

section[data-testid="stSidebar"]{
  background:var(--panel);
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p{ color:var(--muted); }

.stTextInput input, .stTextArea textarea{
  background:var(--panel-2) !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
  font-family:'IBM Plex Mono', monospace;
}
.stTextInput input:focus, .stTextArea textarea:focus{
  border-color:var(--amber) !important;
  box-shadow:0 0 0 1px var(--amber) !important;
}

.stButton > button{
  background:var(--amber);
  color:#171106;
  border:none;
  border-radius:10px;
  font-weight:600;
  font-family:'Space Grotesk', sans-serif;
  letter-spacing:.2px;
  transition:transform .08s ease, box-shadow .15s ease;
}
.stButton > button:hover{
  transform:translateY(-1px);
  box-shadow:0 6px 18px rgba(242,169,59,0.25);
  color:#171106;
}
.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]){
  background:transparent;
  color:var(--text);
  border:1px solid var(--border);
}

[data-testid="stAlert"]{
  background:var(--panel-2) !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
  color:var(--text) !important;
}

.stTabs [data-baseweb="tab-list"]{
  gap:4px;
  border-bottom:1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  background:transparent;
  color:var(--muted);
  font-family:'Space Grotesk', sans-serif;
  font-weight:600;
  border-radius:8px 8px 0 0;
}
.stTabs [aria-selected="true"]{
  color:var(--amber) !important;
  border-bottom:2px solid var(--amber) !important;
}

hr, [data-testid="stDivider"]{ border-color:var(--border) !important; }

.stCheckbox label p, .stToggle label p{ color:var(--text) !important; }

::-webkit-scrollbar{ width:8px; height:8px; }
::-webkit-scrollbar-thumb{ background:#2A3350; border-radius:8px; }

/* Sembunyikan hint bawaan Streamlit "Press Enter to apply" / "Press Ctrl+Enter to apply"
   yang muncul di pojok kanan-bawah text_area saat ada perubahan yang belum di-submit. */
[data-testid="InputInstructions"]{ display:none !important; }

/* ---- custom components ---- */
.eyebrow{
  font-family:'IBM Plex Mono', monospace;
  font-size:12px;
  letter-spacing:2px;
  color:var(--amber);
  text-transform:uppercase;
  margin-bottom:4px;
}
.barcode{
  height:34px;
  background:repeating-linear-gradient(90deg, var(--text) 0 2px, transparent 2px 5px, var(--text) 5px 6px, transparent 6px 11px);
  opacity:.18;
  border-radius:4px;
}
.hero-title{
  font-size:38px;
  font-weight:700;
  color:var(--text);
  margin:6px 0 2px 0;
  line-height:1.15;
}
.hero-sub{
  color:var(--muted);
  font-size:15px;
  max-width:760px;
  margin-bottom:18px;
}

.pipeline-label{
  font-family:'IBM Plex Mono', monospace;
  font-size:11px;
  letter-spacing:1.5px;
  text-transform:uppercase;
  color:var(--muted);
  margin:20px 0 10px 0;
}
.pipeline{
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:9px;
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:16px;
  padding:14px 16px;
  margin:0 0 26px 0;
}
.node{
  display:inline-flex;
  align-items:center;
  gap:8px;
  font-family:'IBM Plex Mono', monospace;
  font-size:12.5px;
  font-weight:600;
  padding:8px 15px 8px 11px;
  border-radius:999px;
  background:var(--panel-2);
  border:1px solid var(--border);
  color:var(--text);
  white-space:nowrap;
  transition:border-color .15s ease, transform .15s ease, background .15s ease;
}
.node:hover{
  transform:translateY(-1px);
  border-color:rgba(255,255,255,0.28);
}
.node .dot{
  width:7px;
  height:7px;
  border-radius:50%;
  flex-shrink:0;
  box-shadow:0 0 6px currentColor;
}
.node.core{
  border-color:rgba(242,169,59,0.45);
  background:rgba(242,169,59,0.08);
  color:var(--amber);
}
.node.core .dot{ background:var(--amber); color:var(--amber); }
.pipeline-sep{
  color:var(--muted);
  font-size:13px;
  opacity:0.5;
  padding:0 1px;
}

.section-label{
  font-family:'IBM Plex Mono', monospace;
  font-size:12px;
  letter-spacing:1.5px;
  color:var(--muted);
  text-transform:uppercase;
  margin:22px 0 10px 0;
}

.divisi-grid{ display:flex; flex-direction:column; gap:6px; }
.divisi-chip{
  display:flex; align-items:center; gap:8px;
  padding:6px 10px;
  border-radius:8px;
  background:var(--panel-2);
  border-left:3px solid transparent;
  font-size:13px;
  color:var(--text);
}

.route-row{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 4px 0; }
.route-chip{
  font-family:'IBM Plex Mono', monospace;
  font-size:12.5px;
  font-weight:600;
  padding:6px 12px;
  border-radius:999px;
  color:#0A0E17;
}

.agent-card{
  background:var(--panel);
  border:1px solid var(--border);
  border-left:4px solid var(--border);
  border-radius:12px;
  padding:16px 18px;
  margin-bottom:14px;
}
.agent-card-head{
  display:flex; align-items:center; gap:10px; margin-bottom:8px;
}
.agent-card-title{
  font-family:'Space Grotesk', sans-serif;
  font-weight:600;
  font-size:15.5px;
  color:var(--text);
}
.agent-card-body{
  color:#C9CFE0;
  font-size:14.5px;
  line-height:1.6;
}
.source-row{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.source-chip{
  font-family:'IBM Plex Mono', monospace;
  font-size:11.5px;
  padding:3px 9px;
  border-radius:6px;
  background:var(--panel-2);
  border:1px solid var(--border);
  color:var(--muted);
}

.timeline-item{
  display:flex; gap:12px;
  padding:10px 0;
  border-bottom:1px dashed var(--border);
  font-size:13.5px;
  color:#C9CFE0;
}
.timeline-item:last-child{ border-bottom:none; }
.timeline-dot{
  min-width:8px; height:8px; border-radius:50%;
  background:var(--blue); margin-top:5px;
}

.final-card{
  background:linear-gradient(180deg, rgba(242,169,59,0.09), rgba(242,169,59,0.02));
  border:1px solid rgba(242,169,59,0.35);
  border-radius:14px;
  padding:20px 22px;
  color:var(--text);
  font-size:15px;
  line-height:1.65;
}

.metric-grid{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:6px; }
.metric-card{
  flex:1; min-width:150px;
  background:var(--panel-2);
  border:1px solid var(--border);
  border-radius:12px;
  padding:14px 16px;
}
.metric-label{
  font-family:'IBM Plex Mono', monospace;
  font-size:11px; letter-spacing:1px; text-transform:uppercase;
  color:var(--muted); margin-bottom:6px;
}
.metric-value{
  font-family:'Space Grotesk', sans-serif;
  font-size:22px; font-weight:700; color:var(--text);
}
.metric-value.good{ color:var(--teal); }
.metric-value.bad{ color:var(--rose); }

.score-row{
  display:flex; align-items:center; gap:12px;
  padding:10px 0; border-bottom:1px solid var(--border);
}
.score-row:last-child{ border-bottom:none; }
.score-name{
  width:170px; font-size:13.5px; color:var(--text); flex-shrink:0;
}
.score-bar-track{
  flex:1; height:8px; border-radius:6px; background:var(--panel-2); overflow:hidden;
}
.score-bar-fill{ height:100%; border-radius:6px; }
.score-val{
  font-family:'IBM Plex Mono', monospace; font-size:12.5px; width:42px; text-align:right; color:var(--muted);
}
.flag-badge{
  font-family:'IBM Plex Mono', monospace; font-size:11px; padding:3px 8px; border-radius:6px; margin-left:8px;
}
</style>
""",
    unsafe_allow_html=True,
)


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
    st.markdown(
        '<div class="eyebrow">PT Retailindo Nusantara</div>'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;font-size:20px;color:#EDEFF5;">'
        '📦 Control Tower</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="barcode" style="margin:12px 0 18px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Konfigurasi</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-label">Divisi Tersedia</div>', unsafe_allow_html=True)
    chips = "".join(
        f'<div class="divisi-chip" style="border-left-color:{DIVISI_COLOR[a]};">'
        f'<span>{DIVISI_ICON[a]}</span><span>{DIVISI_LABEL[a]}</span></div>'
        for a in AVAILABLE_AGENTS
    )
    st.markdown(f'<div class="divisi-grid">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Opsi</div>', unsafe_allow_html=True)
    show_eval = st.toggle("Tampilkan Evaluator Agent (Soal 4)", value=True)


# ============================================================
# Inisialisasi sistem (cached supaya vector DB tidak dibangun ulang tiap interaksi)
# ============================================================
@st.cache_resource(show_spinner="Membangun knowledge base, vector DB, dan graph multi-agent...")
def load_system(api_key: str, csv_dir: str):
    return MultiAgentSystem(groq_api_key=api_key, csv_dir=csv_dir or None)


# ---------------- Hero ----------------
st.markdown('<div class="eyebrow">Multi-Agentic LLM System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🏬 Control Tower — PT Retailindo Nusantara</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Orchestrator merutekan setiap permintaan ke satu/lebih agent divisi '
    '(RAG + tool calling). Agent divisi bisa saling berkonsultasi secara peer-to-peer '
    '&mdash; misalnya Inventory menghubungi Finance saat stok habis.</div>',
    unsafe_allow_html=True,
)

pipeline_nodes = "".join(
    f'<div class="node">'
    f'<span class="dot" style="background:{DIVISI_COLOR[a]};color:{DIVISI_COLOR[a]};"></span>'
    f'{DIVISI_ICON[a]} {DIVISI_LABEL[a]}</div>'
    for a in AVAILABLE_AGENTS
)
st.markdown('<div class="pipeline-label">Alur Agent</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="pipeline">'
    f'<div class="node core"><span class="dot"></span>🧭 Orchestrator</div>'
    f'<span class="pipeline-sep">→</span>'
    f'{pipeline_nodes}'
    f'<span class="pipeline-sep">→</span>'
    f'<div class="node core"><span class="dot"></span>🧾 Aggregator</div>'
    f'</div>',
    unsafe_allow_html=True,
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

st.markdown('<div class="section-label">Permintaan / Keluhan</div>', unsafe_allow_html=True)
query = st.text_area(
    "Masukkan permintaan / keluhan",
    placeholder=contoh_query,
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_clicked = st.button("🚀 Jalankan", type="primary", use_container_width=True)
with col2:
    if st.button("✨ Gunakan contoh skenario", use_container_width=True):
        query = contoh_query
        run_clicked = True

st.markdown(
    '<div style="margin:-6px 0 4px 0;color:#5B6580;font-size:11.5px;'
    'font-family:\'IBM Plex Mono\',monospace;">'
    'Enter untuk jalankan &middot; Shift+Enter untuk baris baru</div>',
    unsafe_allow_html=True,
)

# Enter = jalankan, Shift+Enter = baris baru (perilaku ala chat LLM).
# st.text_area bawaan Streamlit tidak punya opsi ini, jadi disuntik lewat JS
# yang menyadap textarea di parent document dan mengklik tombol "Jalankan".
components.html(
    """
    <script>
    (function () {
        function bind() {
            var doc = window.parent.document;
            var textareas = doc.querySelectorAll('textarea');
            var target = null;
            for (var i = 0; i < textareas.length; i++) {
                var ta = textareas[i];
                var label = (ta.getAttribute('aria-label') || '');
                if (label.indexOf('Masukkan permintaan') !== -1) { target = ta; break; }
            }
            if (!target) { return false; }
            if (target.dataset.enterRunBound === "1") { return true; }
            target.dataset.enterRunBound = "1";

            target.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && e.keyCode !== 229) {
                    e.preventDefault();
                    e.stopPropagation();
                    target.blur();
                    setTimeout(function () {
                        var buttons = doc.querySelectorAll('button');
                        for (var j = 0; j < buttons.length; j++) {
                            var txt = (buttons[j].innerText || '').trim();
                            if (txt.indexOf('Jalankan') !== -1) {
                                buttons[j].click();
                                break;
                            }
                        }
                    }, 80);
                }
            });
            return true;
        }
        var attempts = 0;
        var timer = setInterval(function () {
            attempts++;
            if (bind() || attempts > 40) { clearInterval(timer); }
        }, 250);
    })();
    </script>
    """,
    height=0,
)

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

    st.markdown('<div id="hasil-section"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">📋 Hasil</div>', unsafe_allow_html=True)

    # Auto-scroll ke bagian Hasil supaya user tidak perlu scroll manual
    # setiap kali selesai menjalankan query (mirip behavior chat).
    components.html(
        """
        <script>
        (function () {
            var doc = window.parent.document;
            var attempts = 0;
            var maxAttempts = 30; // ~3 detik total, cukup untuk render selesai

            function tryScroll() {
                attempts++;
                var el = doc.getElementById('hasil-section');
                if (el) {
                    el.scrollIntoView({behavior: 'smooth', block: 'start'});
                    // Ulangi sekali lagi setelah konten (tabs, dsb.) selesai render,
                    // supaya posisi scroll tetap akurat walau tinggi halaman berubah.
                    setTimeout(function () {
                        el.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }, 400);
                    return;
                }
                if (attempts < maxAttempts) {
                    setTimeout(tryScroll, 100);
                }
            }
            tryScroll();
        })();
        </script>
        """,
        height=0,
    )

    st.markdown(f"**Query:** {esc(query)}", unsafe_allow_html=True)

    route_chips = "".join(
        f'<span class="route-chip" style="background:{DIVISI_COLOR.get(d, "#8A93AC")};">'
        f'{DIVISI_ICON.get(d, "🔹")} {DIVISI_LABEL.get(d, d)}</span>'
        for d in result["route"]
    )
    st.markdown(
        f'<div style="margin:6px 0 4px 0;color:#8A93AC;font-size:13px;">Routing Orchestrator</div>'
        f'<div class="route-row">{route_chips}</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["🧩 Jawaban Tiap Agent", "🔗 Log Interaksi Antar-Agent", "✅ Rekomendasi Akhir"])

    with tabs[0]:
        for divisi, jawaban in result["outputs"].items():
            color = DIVISI_COLOR.get(divisi, "#8A93AC")
            sources = result["sources"].get(divisi, [])
            source_html = ""
            if sources:
                chips = "".join(f'<span class="source-chip">{esc(s)}</span>' for s in sources)
                source_html = f'<div class="source-row">{chips}</div>'
            st.markdown(
                f'<div class="agent-card" style="border-left-color:{color};">'
                f'<div class="agent-card-head">'
                f'<span style="font-size:18px;">{DIVISI_ICON.get(divisi, "🧩")}</span>'
                f'<span class="agent-card-title">{DIVISI_LABEL.get(divisi, divisi)}</span>'
                f'</div>'
                f'<div class="agent-card-body">{esc(jawaban)}</div>'
                f'{source_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tabs[1]:
        if result.get("interaction_log"):
            items = "".join(
                f'<div class="timeline-item"><div class="timeline-dot"></div><div>{esc(log)}</div></div>'
                for log in result["interaction_log"]
            )
            st.markdown(f'<div>{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="agent-card" style="border-left-color:#232C42;color:#8A93AC;">'
                '(Tidak ada konsultasi antar-agent pada permintaan ini)</div>',
                unsafe_allow_html=True,
            )

    with tabs[2]:
        final_text = result.get("final_answer") or "(tidak ada rekomendasi akhir)"
        st.markdown(f'<div class="final-card">{esc(final_text)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-top:14px;color:#8A93AC;font-size:12.5px;font-family:\'IBM Plex Mono\',monospace;">'
        f'⏱ Latensi eksekusi: {result.get("_latency", 0):.2f} detik</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Evaluator Agent ----------------
    if show_eval:
        st.markdown("---")
        st.markdown('<div class="section-label">📊 Evaluator Agent (Soal 4)</div>', unsafe_allow_html=True)
        with st.spinner("Menjalankan evaluasi (Accuracy, Effectiveness, Efficiency, Explainability, Hallucination)..."):
            try:
                eval_result = system.evaluate(query, result)
            except Exception as e:
                st.error(f"Evaluasi gagal: {e}")
                eval_result = None

        if eval_result:
            eff = eval_result["effectiveness"]
            effi = eval_result["efficiency"]
            completed = eff.get("task_completed", False)

            metrics_html = f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-label">Task Completed</div>
                <div class="metric-value {'good' if completed else 'bad'}">{'Ya' if completed else 'Tidak'}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-value">{effi.get('latency_seconds', 0)}s</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Agent Dipanggil</div>
                <div class="metric-value">{effi.get('agents_called', 0)}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Status Efisiensi</div>
                <div class="metric-value {'good' if effi.get('status') == 'OK' else 'bad'}" style="font-size:15px;">
                  {esc(effi.get('status', '-'))}
                </div>
              </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
            expl = eval_result["explainability"]
            expl_chips = "".join(
                f'<span class="route-chip" style="background:{DIVISI_COLOR.get(d, "#8A93AC")};">'
                f'{DIVISI_LABEL.get(d, d)} · {v["jumlah_sumber"]} sumber '
                f'{"✓" if v["punya_sitasi"] else "✗"}</span>'
                for d, v in expl.items()
            )
            st.markdown(f'<div class="route-row">{expl_chips}</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">Accuracy / Hallucination '
                '<span style="color:#5B6580;">(faithfulness terhadap konteks RAG + tool)</span></div>',
                unsafe_allow_html=True,
            )
            rows_html = ""
            for divisi, v in eval_result["accuracy_hallucination"].items():
                score = v["faithfulness_score"]
                flagged = v["hallucination_flag"]
                bar_color = "#FB7185" if score < 0.6 else ("#F2A93B" if score < 0.8 else "#34D399")
                badge = (
                    '<span class="flag-badge" style="background:rgba(251,113,133,0.15);color:#FB7185;">⚠ risiko halusinasi</span>'
                    if flagged else
                    '<span class="flag-badge" style="background:rgba(52,211,153,0.15);color:#34D399;">✓ aman</span>'
                )
                rows_html += f"""
                <div class="score-row">
                  <div class="score-name">{DIVISI_ICON.get(divisi,'')} {esc(DIVISI_LABEL.get(divisi, divisi))}</div>
                  <div class="score-bar-track"><div class="score-bar-fill" style="width:{score*100:.0f}%;background:{bar_color};"></div></div>
                  <div class="score-val">{score:.2f}</div>
                  {badge}
                </div>
                """
            st.markdown(f'<div class="agent-card" style="border-left-color:#232C42;">{rows_html}</div>', unsafe_allow_html=True)
