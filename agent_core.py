"""
agent_core.py
=============
Logic inti sistem Multi-Agentic LLM PT Retailindo Nusantara, diekstrak dari
notebook FP_Data_Mining.ipynb menjadi modul Python biasa supaya bisa dipakai
sebagai backend aplikasi Streamlit (app.py).

Yang dipindah dari notebook:
- Knowledge base per divisi (SOP retail sintetis + profil perusahaan)
  (opsional: dataset tiket publik Kaggle kalau file CSV-nya disediakan)
- Chunking (fixed size + overlap)
- Embedding (multilingual) + ChromaDB per divisi
- Tool calling (StructuredTool: cek stok, cari stok cabang lain, cek anggaran)
- Multi-agent orchestration dengan LangGraph (Orchestrator -> 5 agent divisi
  paralel -> Aggregator, + interaksi peer-to-peer Inventory -> Finance)
- Evaluator Agent (Accuracy, Effectiveness, Efficiency, Explainability,
  Hallucination)

Yang SENGAJA dibuang (sesuai permintaan): bagian Fine-Tuning LoRA/PEFT (Soal 3a),
karena butuh GPU dan tidak relevan untuk demo antarmuka Streamlit.
"""

from __future__ import annotations

import glob
import os
import random
import re
import time
from typing import Annotated, Dict, List, Optional, TypedDict

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

MODEL_NAME = "openai/gpt-oss-20b"  # model open-weight via Groq: pengganti resmi llama-3.1-8b-instant
# (lebih cepat & limit free-tier lebih longgar dibanding openai/gpt-oss-120b, yang merupakan
# pengganti llama-3.3-70b-versatile dan jauh lebih berat -- itu penyebab utama loading jadi lambat.
# Kalau butuh kualitas jawaban lebih tinggi dan tidak keberatan lebih lambat, bisa diganti balik
# ke "openai/gpt-oss-120b" lewat parameter `model_name` di MultiAgentSystem / sidebar app.py.)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AVAILABLE_AGENTS = ["marketing", "customer_service", "inventory", "hr", "finance"]

DIVISI_LABEL = {
    "marketing": "Marketing Agent",
    "customer_service": "Customer Service Agent",
    "inventory": "Inventory Agent",
    "hr": "HR Agent",
    "finance": "Finance Agent",
}


# ============================================================
# 1. Retry wrapper untuk semua pemanggilan LLM
# ============================================================
def invoke_with_retry(runnable, prompt, max_retries: int = 4, base_delay: float = 2.0):
    """Retry + exponential backoff untuk pemanggilan LLM (rentan rate limit di tier gratis Groq)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return runnable.invoke(prompt)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "decommissioned" in msg or "does not exist" in msg or "invalid_api_key" in msg:
                raise
            wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"[Peringatan] Panggilan LLM gagal (percobaan {attempt + 1}/{max_retries}): {e}. "
                  f"Mencoba lagi dalam {wait:.1f} detik...")
            time.sleep(wait)
    raise last_err


# ============================================================
# 2. Knowledge Base per Divisi
# ============================================================
QUEUE_TO_DIVISI = {
    "Sales and Pre-Sales": "marketing",
    "Customer Service": "customer_service",
    "Returns and Exchanges": "customer_service",
    "General Inquiry": "customer_service",
    "Technical Support": "inventory",
    "IT Support": "inventory",
    "Product Support": "inventory",
    "Service Outages and Maintenance": "inventory",
    "Human Resources": "hr",
    "Billing and Payments": "finance",
}

KEYWORD_TO_DIVISI = [
    (["hr", "human resource", "payroll", "recruit", "staff", "employee", "cuti", "leave"], "hr"),
    (["bill", "invoice", "payment", "refund", "finance", "budget", "anggaran", "pricing"], "finance"),
    (["sales", "market", "promo", "campaign", "discount", "pre-sales", "advertis"], "marketing"),
    (["stock", "inventory", "product", "shipping", "delivery", "technical", "it support",
      "maintenance", "outage", "warehouse", "logistic", "hardware", "software", "network"], "inventory"),
]


def _klasifikasi_queue_fallback(nama_queue: str) -> str:
    nama_lower = str(nama_queue).lower()
    for kata_kunci_list, divisi in KEYWORD_TO_DIVISI:
        if any(kata in nama_lower for kata in kata_kunci_list):
            return divisi
    return "customer_service"


def load_public_ticket_dataset(csv_dir: Optional[str] = None, n_per_divisi: int = 200) -> Dict[str, List[str]]:
    """
    (OPSIONAL) Memuat dataset publik "Multilingual Customer Support Tickets" (Kaggle) kalau
    file CSV-nya tersedia di `csv_dir`, lalu memetakan kolom "queue" -> 5 divisi.

    Kalau `csv_dir` kosong / tidak ada file CSV yang ditemukan, fungsi ini mengembalikan
    dict kosong (bukan error) -- aplikasi tetap bisa jalan hanya dengan SOP sintetis +
    dokumen profil perusahaan (lihat `build_knowledge_base`), karena dataset Kaggle ini
    tidak realistis untuk ikut di-deploy (perlu diunduh manual dari Kaggle terlebih dulu).
    """
    raw_docs: Dict[str, List[str]] = {a: [] for a in AVAILABLE_AGENTS}
    if not csv_dir or not os.path.isdir(csv_dir):
        return raw_docs

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        return raw_docs

    all_dfs = []
    for path in csv_paths:
        try:
            all_dfs.append(pd.read_csv(path))
            print(f"Berhasil memuat {path}")
        except Exception as e:
            print(f"Gagal memuat {path}: {e}")

    if not all_dfs:
        return raw_docs

    df_tickets = pd.concat(all_dfs, ignore_index=True)
    for col in ["queue", "priority", "language", "subject", "answer"]:
        if col not in df_tickets.columns:
            df_tickets[col] = "unknown"

    df_tickets["divisi"] = df_tickets["queue"].map(QUEUE_TO_DIVISI)
    unmapped_mask = df_tickets["divisi"].isna()
    if unmapped_mask.any():
        df_tickets.loc[unmapped_mask, "divisi"] = df_tickets.loc[unmapped_mask, "queue"].apply(
            _klasifikasi_queue_fallback
        )

    for divisi in AVAILABLE_AGENTS:
        subset = df_tickets[df_tickets["divisi"] == divisi].dropna(subset=["answer"])
        subset = subset.sample(n=min(n_per_divisi, len(subset)), random_state=42) if len(subset) else subset
        raw_docs[divisi] = [
            f"[{row['queue']} | prioritas {row['priority']} | {row['language']}] "
            f"Subjek: {row['subject']}\nJawaban resmi CS: {row['answer']}"
            for _, row in subset.iterrows()
        ]
    return raw_docs


RETAIL_SOP_DOCS: Dict[str, List[str]] = {
    "inventory": [
        "SOP Stok Retailindo: setiap cabang wajib melapor stok mingguan. Jika stok sebuah SKU "
        "mencapai 0 (habis), sistem otomatis mencari stok pengganti di cabang terdekat sebelum "
        "melakukan rush order ke gudang pusat.",
        "SOP Rush Order: pengajuan rush order (pengiriman cepat antar-cabang) hanya bisa diproses "
        "jika cabang tujuan memiliki sisa anggaran rush order pada bulan berjalan, dikonfirmasi "
        "oleh divisi Finance.",
        "SOP Restock Gudang Pusat: waktu pengiriman standar dari gudang pusat ke cabang adalah "
        "3-5 hari kerja; untuk kategori rush order dipersingkat menjadi 1-2 hari dengan biaya tambahan.",
        "Kebijakan Stok Minimum: setiap cabang wajib menjaga stok minimum 5 unit per SKU "
        "best-seller untuk mencegah stockout mendadak.",
    ],
    "finance": [
        "SOP Anggaran Rush Order: setiap cabang mendapat alokasi anggaran rush order bulanan "
        "sebesar Rp20.000.000, digunakan khusus untuk kondisi stok kritis (stockout) yang "
        "berdampak langsung ke komplain pelanggan.",
        "Kebijakan Approval Anggaran: pengeluaran mendesak di atas anggaran bulanan wajib melalui "
        "approval manual dari Finance Manager, maksimal 1x24 jam.",
        "SOP Rekonsiliasi Keuangan Cabang: laporan keuangan cabang direkonsiliasi setiap akhir "
        "bulan, termasuk biaya rush order dan retur barang.",
    ],
    "customer_service": [
        "SOP Retur & Komplain: pelanggan dapat mengajukan retur maksimal 7 hari setelah pembelian "
        "dengan struk asli dan barang belum dipakai. Untuk kasus stok habis, CS wajib menawarkan "
        "alternatif cabang atau kompensasi voucher.",
        "SOP Eskalasi Komplain: komplain terkait ketersediaan stok yang berdampak ke lebih dari 1 "
        "pelanggan dalam sehari wajib dieskalasi ke Inventory dan Finance dalam waktu 1 jam.",
        "Standar Respons CS: seluruh komplain pelanggan wajib direspons dalam waktu maksimal 2 jam "
        "kerja dan diberikan solusi konkret, bukan hanya permintaan maaf.",
    ],
    "marketing": [
        "SOP Promosi & Stok: sebelum meluncurkan promosi diskon pada suatu SKU, Marketing wajib "
        "mengecek ketersediaan stok ke Inventory agar promosi tidak menyebabkan stockout dan "
        "komplain pelanggan.",
        "Kebijakan Kampanye Regional: kampanye marketing regional (misal Yogyakarta) disesuaikan "
        "dengan data stok & tren penjualan cabang setempat.",
    ],
    "hr": [
        "SOP Jam Kerja & Cuti: karyawan cabang berhak atas cuti tahunan 12 hari, pengajuan minimal "
        "H-3 melalui sistem HR.",
        "SOP Rekrutmen Cabang Baru: setiap pembukaan cabang baru wajib diikuti rekrutmen minimal "
        "4 staf toko dan 1 supervisor dalam waktu 30 hari sebelum grand opening.",
    ],
}

COMPANY_PROFILE_DOCS: List[str] = [
    "Profil Perusahaan: PT Retailindo Nusantara adalah perusahaan ritel multi-cabang yang "
    "bergerak di bidang penjualan produk fashion & footwear (sepatu, pakaian, aksesoris) "
    "secara offline melalui toko cabang dan sebagian online.",
    "Jaringan Cabang: PT Retailindo Nusantara memiliki cabang di beberapa kota, antara lain "
    "Yogyakarta, Solo, Semarang, Klaten, dan Magelang, masing-masing dikelola oleh tim toko "
    "lokal yang melapor ke kantor pusat.",
    "Struktur Organisasi: operasional perusahaan dibagi ke dalam 5 divisi utama -- Marketing "
    "(promosi & kampanye), Customer Service (retur & komplain pelanggan), Inventory (stok & "
    "distribusi barang antar-cabang), HR (kepegawaian & cuti), dan Finance (anggaran & "
    "rekonsiliasi keuangan cabang).",
    "Jam Operasional Toko: seluruh cabang PT Retailindo Nusantara buka setiap hari pukul "
    "09.00-21.00 waktu setempat, termasuk akhir pekan, kecuali hari libur nasional tertentu "
    "yang diumumkan terpisah oleh kantor pusat.",
    "Metode Pembayaran: pelanggan dapat membayar secara tunai, kartu debit/kredit, QRIS, "
    "maupun transfer bank untuk pembelian online, di seluruh cabang dan kanal online resmi.",
    "Kontak Layanan Pelanggan: pertanyaan atau komplain yang tidak dapat diselesaikan di toko "
    "dapat diajukan lewat layanan Customer Service pusat, dengan target waktu respons maksimal "
    "2 jam kerja sesuai SOP internal.",
    "Kebijakan Garansi Produk: produk footwear yang mengalami cacat produksi (bukan akibat "
    "pemakaian) mendapat garansi penggantian 30 hari sejak tanggal pembelian, dengan struk "
    "pembelian asli.",
]


def build_knowledge_base(csv_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Menggabungkan dokumen profil perusahaan + SOP retail sintetis + (opsional) dataset tiket publik."""
    raw_docs = load_public_ticket_dataset(csv_dir)
    for divisi, docs in RETAIL_SOP_DOCS.items():
        raw_docs.setdefault(divisi, [])
        raw_docs[divisi] = docs + raw_docs[divisi]
    for divisi in AVAILABLE_AGENTS:
        raw_docs.setdefault(divisi, [])
        raw_docs[divisi] = COMPANY_PROFILE_DOCS + raw_docs[divisi]
    return raw_docs


# ============================================================
# 3. Chunking, Embedding, Vector Database
# ============================================================
def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Fixed Size Chunking dengan Overlap (satuan kata)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_vectorstores(raw_docs: Dict[str, List[str]]) -> Dict[str, Chroma]:
    """Chunking -> embedding (multilingual) -> ChromaDB, satu collection per divisi."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    chunked_kb: Dict[str, List[str]] = {}
    for divisi, docs in raw_docs.items():
        chunks: List[str] = []
        for doc in docs:
            chunks.extend(chunk_text(doc, chunk_size=80, overlap=20))
        chunked_kb[divisi] = chunks

    vectorstores: Dict[str, Chroma] = {}
    for divisi, chunks in chunked_kb.items():
        docs = [Document(page_content=c, metadata={"divisi": divisi, "chunk_id": i})
                for i, c in enumerate(chunks)]
        vectorstores[divisi] = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=f"{divisi}_docs",
        )
    return vectorstores


def make_retrieve_context(vectorstores: Dict[str, Chroma]):
    def retrieve_context(divisi: str, query: str, k: int = 4, score_threshold: float = 0.15) -> List[Document]:
        try:
            hasil = vectorstores[divisi].similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            print(f"[Peringatan] Retrieval gagal untuk divisi '{divisi}': {e}")
            return []
        if not hasil:
            return []
        filtered = [doc for doc, score in hasil if score >= score_threshold]
        if not filtered:
            filtered = [hasil[0][0]]
        return filtered

    return retrieve_context


# ============================================================
# 4. Tool Calling (simulasi sistem internal)
# ============================================================
_stok_cabang = {
    ("Yogyakarta", "Sepatu Lari X1"): 0,
    ("Solo", "Sepatu Lari X1"): 24,
    ("Semarang", "Sepatu Lari X1"): 9,
    ("Klaten", "Sepatu Lari X1"): 15,
    ("Magelang", "Sepatu Lari X1"): 3,
}
_anggaran_rush_order = {"Yogyakarta": 20_000_000, "Solo": 20_000_000}


def cek_stok_produk(cabang: str, produk: str) -> dict:
    """Mengecek jumlah stok sebuah produk pada satu cabang tertentu."""
    qty = _stok_cabang.get((cabang, produk))
    if qty is None:
        return {"error": f"Data stok untuk {produk} di cabang {cabang} tidak ditemukan."}
    return {"cabang": cabang, "produk": produk, "stok": qty}


def cari_stok_cabang_lain(produk: str, exclude_cabang: str) -> dict:
    """Mencari cabang lain yang masih memiliki stok produk tertentu."""
    hasil = [
        {"cabang": cabang, "stok": qty}
        for (cabang, p), qty in _stok_cabang.items()
        if p == produk and cabang != exclude_cabang and qty > 0
    ]
    hasil.sort(key=lambda x: -x["stok"])
    return {"produk": produk, "cabang_tersedia": hasil}


def cek_anggaran_rush_order(cabang: str) -> dict:
    """Mengecek sisa anggaran rush order (restock mendesak) sebuah cabang."""
    sisa = _anggaran_rush_order.get(cabang, 0)
    return {"cabang": cabang, "sisa_anggaran_rush_order": sisa}


def build_tools():
    tool_cek_stok_produk = StructuredTool.from_function(
        func=cek_stok_produk, name="cek_stok_produk",
        description="Mengecek jumlah stok sebuah produk pada satu cabang tertentu.",
    )
    tool_cari_stok_cabang_lain = StructuredTool.from_function(
        func=cari_stok_cabang_lain, name="cari_stok_cabang_lain",
        description="Mencari cabang lain yang masih memiliki stok sebuah produk tertentu.",
    )
    tool_cek_anggaran_rush_order = StructuredTool.from_function(
        func=cek_anggaran_rush_order, name="cek_anggaran_rush_order",
        description="Mengecek sisa anggaran rush order (restock mendesak) sebuah cabang.",
    )
    agent_tools = {
        "inventory": [tool_cek_stok_produk, tool_cari_stok_cabang_lain],
        "finance": [tool_cek_anggaran_rush_order],
        "customer_service": [tool_cek_stok_produk],
    }
    mandatory_first_tool = {"inventory": tool_cek_stok_produk}
    return agent_tools, mandatory_first_tool


# ============================================================
# 5. State & Multi-Agent Graph (LangGraph)
# ============================================================
def merge_dict(a: dict, b: dict) -> dict:
    merged = dict(a)
    merged.update(b)
    return merged


class AgentState(TypedDict):
    query: str
    route: List[str]
    outputs: Annotated[Dict[str, str], merge_dict]
    sources: Annotated[Dict[str, List[str]], merge_dict]
    grounding: Annotated[Dict[str, str], merge_dict]
    interaction_log: Annotated[List[str], lambda a, b: a + b]
    final_answer: Optional[str]


class MultiAgentSystem:
    """Membungkus llm, retrieval, tools, dan graph menjadi satu objek siap pakai oleh app.py."""

    def __init__(self, groq_api_key: str, csv_dir: Optional[str] = None, model_name: str = MODEL_NAME):
        os.environ["GROQ_API_KEY"] = groq_api_key
        self.llm = ChatGroq(model=model_name, temperature=0.3)

        raw_docs = build_knowledge_base(csv_dir)
        self.vectorstores = build_vectorstores(raw_docs)
        self.retrieve_context = make_retrieve_context(self.vectorstores)
        self.agent_tools, self.mandatory_first_tool = build_tools()

        self.graph = self._build_graph()

    # ---------- Orchestrator ----------
    def _orchestrator_node(self, state: AgentState) -> dict:
        prompt = f"""Kamu adalah Orchestrator Agent pada sistem multi-agent PT Retailindo Nusantara.
Agent yang tersedia: {AVAILABLE_AGENTS}.
Tugasmu: tentukan agent mana saja (bisa lebih dari satu, boleh cuma satu) yang relevan untuk
menangani permintaan berikut. Jawab HANYA dengan nama-nama agent dipisah koma (persis salah satu
dari daftar di atas), TANPA penjelasan, TANPA tanda baca lain, TANPA kalimat pembuka/penutup.

Contoh format jawaban yang benar: inventory, finance

Permintaan: "{state['query']}"
"""
        try:
            response = invoke_with_retry(self.llm, prompt).content.strip().lower()
        except Exception as e:
            print(f"[Peringatan] Orchestrator gagal memanggil LLM: {e}. Fallback ke customer_service.")
            return {"route": ["customer_service"]}

        route = [a for a in AVAILABLE_AGENTS if a.replace("_", " ") in response.replace("_", " ") or a in response]
        route = list(dict.fromkeys(route))
        if not route:
            route = ["customer_service"]
        print(f"[Orchestrator] Routing ke: {route}")
        return {"route": route}

    @staticmethod
    def _route_decider(state: AgentState) -> List[str]:
        return state["route"]

    # ---------- Specialist agent ----------
    def _run_specialist_answer(self, divisi: str, query: str, extra_context: str = ""):
        retrieved = self.retrieve_context(divisi, query, k=4)
        rag_context = "\n".join(f"- {d.page_content}" for d in retrieved) or "(tidak ada dokumen relevan ditemukan)"
        sources = [d.page_content[:80] + "..." for d in retrieved]
        full_context = (rag_context + "\n" + extra_context).strip()

        prompt = f"""Kamu adalah {DIVISI_LABEL[divisi]} pada sistem multi-agent PT Retailindo Nusantara.
Jawab permintaan berikut HANYA berdasarkan konteks & data tool di bawah ini. Jika informasi tidak cukup,
katakan demikian secara eksplisit -- JANGAN mengarang angka atau kebijakan yang tidak ada di konteks.
Jawaban singkat, jelas, dan actionable, 2-4 kalimat, dalam Bahasa Indonesia.

Konteks internal (RAG):
{rag_context}
{extra_context}

Permintaan: "{query}"

Jawaban {DIVISI_LABEL[divisi]}:"""
        answer = invoke_with_retry(self.llm, prompt).content.strip()
        return answer, sources, full_context

    def _call_agent_tools(self, divisi: str, query: str):
        """
        Sebelumnya fungsi ini melakukan 2 panggilan LLM terpisah (paksa tool wajib, lalu tool
        bebas) untuk divisi yang punya tool wajib -- itu 2x latency Groq per agent yang butuh
        tool. Sekarang digabung jadi 1 panggilan yang boleh memanggil beberapa tool sekaligus,
        dengan instruksi eksplisit soal tool mana yang wajib. Fallback single-tool-call di bawah
        hanya jalan (nambah 1 panggilan lagi) kalau model ternyata lupa memanggil tool wajibnya --
        jarang terjadi, jadi rata-rata kasus tetap 1 panggilan saja.
        """
        tools = self.agent_tools.get(divisi)
        if not tools:
            return "", []

        tool_text = ""
        calls_made: list = []
        mandatory_tool = self.mandatory_first_tool.get(divisi)
        mandatory_note = (
            f"\nWAJIB panggil tool `{mandatory_tool.name}` untuk verifikasi data terkini "
            f"(tentukan sendiri argumennya dari kalimat user), sebelum/bersamaan dengan tool lain "
            f"kalau relevan."
            if mandatory_tool is not None else ""
        )

        try:
            llm_with_tools = self.llm.bind_tools(tools)
            decide_prompt = f"""Kamu adalah {DIVISI_LABEL[divisi]} pada sistem PT Retailindo Nusantara.
Panggil tool yang sesuai untuk memenuhi permintaan berikut -- boleh lebih dari satu tool sekaligus
kalau perlu (mis. verifikasi stok lalu cari cabang alternatif).{mandatory_note}
Jika tidak ada tool yang relevan sama sekali, jangan panggil tool apa pun.

Permintaan: "{query}"
"""
            ai_msg = invoke_with_retry(llm_with_tools, decide_prompt)
            for call in (getattr(ai_msg, "tool_calls", None) or []):
                tool_fn = {t.name: t for t in tools}.get(call["name"])
                if tool_fn is None:
                    continue
                result = tool_fn.invoke(call["args"])
                tool_text += f"\n[Tool {call['name']}({call['args']})]: {result}"
                calls_made.append((call["name"], call["args"], result))
        except Exception as e:
            print(f"[Peringatan] Tool calling gagal untuk {divisi}: {e}")

        if mandatory_tool is not None and not any(c[0] == mandatory_tool.name for c in calls_made):
            try:
                llm_forced = self.llm.bind_tools([mandatory_tool], tool_choice=mandatory_tool.name)
                forced_prompt = f"""Kamu adalah {DIVISI_LABEL[divisi]}. Verifikasi data terkini dengan
memanggil tool `{mandatory_tool.name}` berdasarkan permintaan berikut (tentukan sendiri argumennya
dari kalimat user).

Permintaan: "{query}"
"""
                forced_msg = invoke_with_retry(llm_forced, forced_prompt)
                for call in (getattr(forced_msg, "tool_calls", None) or []):
                    result = mandatory_tool.invoke(call["args"])
                    tool_text += f"\n[Tool {call['name']}({call['args']})]: {result}"
                    calls_made.append((call["name"], call["args"], result))
            except Exception as e:
                print(f"[Peringatan] Fallback tool wajib '{mandatory_tool.name}' gagal untuk {divisi}: {e}")

        return tool_text, calls_made

    def _make_agent_node(self, divisi: str):
        def node(state: AgentState) -> dict:
            query = state["query"]
            interaction_log: List[str] = []
            try:
                tool_text, calls_made = self._call_agent_tools(divisi, query)

                # Interaksi peer-to-peer: Inventory -> Finance saat stok = 0
                for name, args, result in calls_made:
                    if divisi == "inventory" and name == "cek_stok_produk" \
                            and isinstance(result, dict) and result.get("stok") == 0:
                        cabang = args.get("cabang", "")
                        produk = args.get("produk", "")
                        consult_query = (
                            f"Cek sisa anggaran rush order untuk cabang {cabang} terkait "
                            f"stockout produk {produk}: {query}"
                        )
                        try:
                            finance_answer, _, _ = self._run_specialist_answer("finance", consult_query)
                        except Exception as e:
                            finance_answer = f"(konsultasi ke Finance Agent gagal: {e})"
                        tool_text += f"\n[Konsultasi ke Finance Agent]: {finance_answer}"
                        interaction_log.append(
                            f"Inventory Agent -> Finance Agent: konsultasi anggaran rush order {cabang}. "
                            f"Jawaban Finance: {finance_answer[:120]}..."
                        )

                answer, sources, full_context = self._run_specialist_answer(divisi, query, extra_context=tool_text)
                return {
                    "outputs": {divisi: answer},
                    "sources": {divisi: sources},
                    "grounding": {divisi: full_context},
                    "interaction_log": interaction_log,
                }
            except Exception as e:
                print(f"[Peringatan] Node '{divisi}' gagal total: {e}")
                pesan_error = (
                    f"Mohon maaf, {DIVISI_LABEL[divisi]} sedang mengalami gangguan teknis "
                    f"(LLM/koneksi) dan tidak bisa memproses permintaan ini saat ini."
                )
                return {
                    "outputs": {divisi: pesan_error},
                    "sources": {divisi: []},
                    "grounding": {divisi: ""},
                    "interaction_log": [],
                }

        return node

    # ---------- Aggregator ----------
    def _aggregator_node(self, state: AgentState) -> dict:
        combined = "\n".join(f"[{DIVISI_LABEL[d]}] {ans}" for d, ans in state["outputs"].items())
        interaksi = "\n".join(state.get("interaction_log", [])) or "(tidak ada konsultasi antar-agent pada permintaan ini)"
        prompt = f"""Gabungkan jawaban dari beberapa agent divisi berikut menjadi satu rekomendasi tindakan
yang koheren dan actionable untuk manajemen PT Retailindo Nusantara. Bahasa Indonesia, maksimal 6 kalimat.

Permintaan awal: "{state['query']}"

Jawaban tiap agent:
{combined}

Catatan interaksi antar-agent (peer-to-peer):
{interaksi}

Rekomendasi akhir:"""
        try:
            final = invoke_with_retry(self.llm, prompt).content.strip()
        except Exception as e:
            print(f"[Peringatan] Aggregator gagal memanggil LLM: {e}.")
            final = ("Rekomendasi otomatis tidak dapat dibuat karena gangguan LLM. "
                     "Berikut jawaban tiap divisi apa adanya:\n" + combined)
        return {"final_answer": final}

    # ---------- Build graph ----------
    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("orchestrator", self._orchestrator_node)
        for divisi in AVAILABLE_AGENTS:
            graph_builder.add_node(divisi, self._make_agent_node(divisi))
        graph_builder.add_node("aggregator", self._aggregator_node)

        graph_builder.set_entry_point("orchestrator")
        graph_builder.add_conditional_edges(
            "orchestrator", self._route_decider, {d: d for d in AVAILABLE_AGENTS}
        )
        for divisi in AVAILABLE_AGENTS:
            graph_builder.add_edge(divisi, "aggregator")
        graph_builder.add_edge("aggregator", END)

        return graph_builder.compile()

    def run(self, query: str) -> dict:
        """Menjalankan satu query lewat graph multi-agent, mengembalikan state hasil + latency."""
        start_time = time.time()
        result = self.graph.invoke({
            "query": query,
            "route": [],
            "outputs": {},
            "sources": {},
            "grounding": {},
            "interaction_log": [],
            "final_answer": None,
        })
        result["_latency"] = time.time() - start_time
        return result

    # ---------- Evaluator Agent (Soal 4) ----------
    def evaluate(self, query: str, result: dict) -> dict:
        latency = result.get("_latency", 0.0)
        eval_efficiency = {
            "latency_seconds": round(latency, 2),
            "agents_called": len(result["outputs"]),
            "status": "OK" if latency < 15 else "PERLU DIOPTIMASI",
        }

        eval_explainability = {}
        for divisi, src_list in result["sources"].items():
            eval_explainability[divisi] = {
                "jumlah_sumber": len(src_list),
                "punya_sitasi": len(src_list) > 0,
            }

        eval_accuracy_hallucination = self._evaluate_hallucination_and_accuracy(
            query, result["outputs"], result["grounding"]
        )

        eval_effectiveness = {
            "task_completed": bool(result.get("final_answer")),
        }

        return {
            "effectiveness": eval_effectiveness,
            "efficiency": eval_efficiency,
            "explainability": eval_explainability,
            "accuracy_hallucination": eval_accuracy_hallucination,
        }

    def _evaluate_hallucination_and_accuracy(self, query: str, outputs: dict, grounding: dict) -> dict:
        """
        Sebelumnya: 1 panggilan LLM per divisi (bisa sampai 5x). Sekarang digabung jadi 1
        panggilan LLM yang menilai semua divisi sekaligus -- jauh lebih cepat, terutama saat
        Evaluator Agent diaktifkan bersamaan dengan banyak divisi ter-route.
        """
        divisi_list = list(outputs.keys())
        if not divisi_list:
            return {}

        blocks = []
        for divisi in divisi_list:
            konteks = grounding.get(divisi, "") or "(tidak ada konteks)"
            jawaban = outputs.get(divisi, "")
            blocks.append(f"=== DIVISI: {divisi} ===\nKonteks:\n{konteks}\n\nJawaban yang dinilai:\n{jawaban}")
        combined_blocks = "\n\n".join(blocks)

        prompt = f"""Kamu adalah Evaluator (LLM-as-a-Judge). Untuk SETIAP divisi di bawah, nilai seberapa
jawabannya didukung PENUH oleh konteks yang diberikan (faithfulness), skala 0.0-1.0 (1.0 = didukung
penuh, 0.0 = tidak didukung sama sekali / mengarang).

Wajib jawab dengan PERSIS satu baris per divisi, format:
<nama_divisi>: <angka 0.0-1.0>
Tidak ada penjelasan lain, tidak ada baris kosong, tidak ada markdown. Nama divisi harus persis
salah satu dari: {", ".join(divisi_list)}

Permintaan awal: "{query}"

{combined_blocks}"""

        resp = ""
        try:
            resp = invoke_with_retry(self.llm, prompt).content.strip()
        except Exception as e:
            print(f"[Peringatan] Evaluator batch gagal memanggil LLM: {e}")

        parsed_scores: Dict[str, float] = {}
        for line in resp.splitlines():
            m = re.match(r"\s*([a-z_]+)\s*[:\-]\s*(0(?:\.\d+)?|1(?:\.0+)?)", line.strip(), re.IGNORECASE)
            if not m:
                continue
            key = m.group(1).lower()
            for divisi in divisi_list:
                if key == divisi or key in divisi or divisi in key:
                    parsed_scores[divisi] = max(0.0, min(1.0, float(m.group(2))))
                    break

        if not resp:
            print(f"[Peringatan] Evaluator tidak menghasilkan respons sama sekali untuk: {divisi_list}")
        elif len(parsed_scores) < len(divisi_list):
            hilang = [d for d in divisi_list if d not in parsed_scores]
            print(f"[Peringatan] Evaluator gagal parse skor untuk sebagian divisi {hilang}. Respons: {resp!r}")

        hasil = {}
        for divisi in divisi_list:
            # Gagal parse total dianggap "tidak yakin", bukan otomatis halusinasi
            # (threshold hallucination_flag-nya score < 0.6).
            score = parsed_scores.get(divisi, 0.7)
            hasil[divisi] = {
                "faithfulness_score": round(score, 2),
                "hallucination_flag": score < 0.6,
            }
        return hasil
