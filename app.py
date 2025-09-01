import os, gradio as gr
from rag.pipeline import CryptoRAGPipeline
from rag.tools import get_price, get_fear_greed

pipe: CryptoRAGPipeline | None = None
DEFAULT_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _ensure_pipe(dense_model: str | None = None, reranker_model: str | None = None):
    global pipe
    if pipe is None:
        pipe = CryptoRAGPipeline(
            dense_model=dense_model or DEFAULT_DENSE,
            reranker_model=reranker_model or DEFAULT_RERANK
        )
    return pipe

def setup_pipeline(dense_model, reranker_model):
    _ensure_pipe(dense_model, reranker_model)
    return "✅ Pipeline initialised."

def add_openai_key(key):
    p = _ensure_pipe()
    key = (key or "").strip()
    if not key:
        return "Please paste an OpenAI API key"
    p.set_openai(key)
    return "🔐 OpenAI key set (not stored on disk)."

def add_files(files):
    p = _ensure_pipe()
    paths = [f.name for f in (files or [])]
    if not paths:
        return "No files uploaded."
    p.add_local_files(paths)
    return f"📄 Added {len(paths)} file(s)."

def add_urls(urls_text):
    p = _ensure_pipe()
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    if not urls:
        return "No URLs provided."
    p.add_urls(urls)
    return f"🔗 Added {len(urls)} URL(s)."

def build_index():
    p = _ensure_pipe()
    p.build()
    return "🧱 Index built (hybrid: BM25 + Dense)."

#def answer(query, k, alpha, top_k_rerank, filter_coin, stream_enable, model):
def answer(query, k, alpha, top_k_rerank, stream_enable, model):
    p = _ensure_pipe()
 
    try:
        result = p.ask(
            query, k=int(k), alpha=float(alpha),
            top_k_rerank=int(top_k_rerank),
            filters=None, stream=stream_enable
        )
    except Exception as e:
        yield f"❌ Error while routing: {e}"
        return

    # Tool route (non-stream)
    if result["route"] == "tools":
        # Auto-detect coin from the user's query and show its price.
        from rag.tools import get_price_any, get_price_multi, get_fear_greed
        try:
            coin_id, price = get_price_any(query, "usd")
        except Exception as e:
            yield f"🔧 Tool route: error resolving coin/price — {e}"
            return

        # Always include Fear & Greed (market mood)
        parts = []
        if price is not None:
            parts.append(f"{coin_id} price ≈ ${price}")
        else:
            parts.append(f"{coin_id} price unavailable")

        try:
            fng = get_fear_greed()
            if fng:
                parts.append(f"Fear&Greed: {fng.get('value')} – {fng.get('value_classification')}")
        except Exception:
            pass

        # (Optional) If user didn’t specify a coin clearly, also show a quick trio: ETH, SOL, XRP
        if coin_id not in {"ethereum", "solana", "ripple"} and any(w in query.lower() for w in ["price", "quote"]):
            try:
                batch = get_price_multi(["ethereum", "solana", "ripple"], "usd")
                trio = []
                if "ethereum" in batch and "usd" in batch["ethereum"]:
                    trio.append(f"ETH ${batch['ethereum']['usd']}")
                if "solana" in batch and "usd" in batch["solana"]:
                    trio.append(f"SOL ${batch['solana']['usd']}")
                if "ripple" in batch and "usd" in batch["ripple"]:
                    trio.append(f"XRP ${batch['ripple']['usd']}")
                if trio:
                    parts.append("Also: " + " | ".join(trio))
            except Exception:
                pass

        yield "🔧 " + " | ".join(parts)
        return


    # Retrieval not ready / no results
    if result["route"] == "not_ready":
        reason = result.get("reason")
        if reason == "index_empty":
            yield "⚠️ Your knowledge base is empty. Upload PDF/TXT/MD or add URLs, then click **Build Index**."
        elif reason == "build_failed":
            yield "⚠️ Index not built. Try clicking **Build Index** (after adding docs/URLs)."
        elif reason == "no_results":
            yield "🤔 No matches retrieved. Try a simpler query, different keywords, or ingest more sources; then rebuild."
        else:
            yield "⚠️ Retrieval not ready. Please ingest and build."
        return

    # RAG route
    contexts = result["contexts"]

    # Stream tokens → progressively yield the growing string
    if stream_enable:
        full = ""
        try:
            for token in p.answer_stream(query, contexts, model=model):
                full += token
                yield full
        except Exception as e:
            yield f"❌ Error while streaming: {e}"
        return
    else:
        # Non-streaming fallback (join all tokens)
        try:
            text = "".join(p.answer_stream(query, contexts, model=model))
        except Exception as e:
            yield f"❌ Error while generating: {e}"
            return
        yield text

def _push_status(msg: str, history: list[str] | None, keep: int = 10):
    # 1 line per message; strip newlines
    line = (msg or "").strip().replace("\n", " ")
    hist = (history or []) + [line]
    hist = hist[-keep:]                      # keep last 5
    text = "\n".join(hist)                   # render as multi-line
    return hist, text

# Wrappers that call your original functions and push into the rolling buffer
def setup_pipeline_s(dense_model, reranker_model, history):
    msg = setup_pipeline(dense_model, reranker_model)
    return _push_status(msg, history)

def add_openai_key_s(key, history):
    msg = add_openai_key(key)
    return _push_status(msg, history)

def add_files_s(files, history):
    msg = add_files(files)
    return _push_status(msg, history)

def add_urls_s(urls_text, history):
    msg = add_urls(urls_text)
    return _push_status(msg, history)

def build_index_s(history):
    msg = build_index()
    return _push_status(msg, history)

def on_load_s(history):
    # If you want MANUAL init, return a neutral line here instead
    return _push_status("👋 Ready. Click 'Initialize pipeline' to begin.", history)

with gr.Blocks(
    title="Crypto RAG Chatbot",
    css="""
#status-box { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; margin-top: 12px; }
#status-body { white-space: pre-wrap; line-height: 1.25; max-height: calc(1.25em * 5 + 12px); overflow: auto; }
"""
) as demo:
    gr.Markdown(
    "# 🟠 Crypto RAG Chatbot:<br>"
    "<span style='font-size:0.95rem; line-height:1.4;'>"
    "Step 1: click Initialize pipeline, enter OpenAI Key,Step 2: Upload documents and Paste links,Step 3: Build Index, Step 4: Ask away<br>" 
    "</span>"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1) Init & Keys")
            dense = gr.Textbox(value=DEFAULT_DENSE, label="Embedding model")
            rerank = gr.Textbox(value=DEFAULT_RERANK, label="Reranker model")
            btn_init = gr.Button("Initialize pipeline")
            #status = gr.Markdown("...")
            key = gr.Textbox(type="password", label="OpenAI API Key (required for chat)")
            btn_key = gr.Button("Set OpenAI Key")

            gr.Markdown("### 2) Ingest Data")
            files = gr.File(label="Upload .pdf / .txt / .md", file_count="multiple")
            btn_files = gr.Button("Add files")
            urls = gr.Textbox(lines=3, label="URLs (one per line)")
            btn_urls = gr.Button("Add URLs")
            btn_build = gr.Button("3) Build Index")

            gr.Markdown("### 3) Query Settings")
            k = gr.Slider(2, 15, value=8, step=1, label="Top-K retrieve")
            alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="Hybrid alpha (BM25↔Dense)")
            topk_rerank = gr.Slider(1, 10, value=5, step=1, label="Top-K after reranker")
            #filter_coin = gr.Textbox(value="", label="Metadata filter: coin (optional)")
            stream_toggle = gr.Checkbox(value=True, label="Streaming")
            model = gr.Textbox(value="gpt-4o-mini", label="Chat model")

        with gr.Column(scale=2):
            # NEW: wider status in the chat column
            #status = gr.Markdown("...", elem_id="status-banner")
            gr.Markdown("### 4) Chat")
            q = gr.Textbox(label="Ask a crypto question", lines=2)
            btn_ask = gr.Button("Ask")
            a = gr.Markdown("...")
            with gr.Group(elem_id="status-box"):
                gr.Markdown("**Status showing below (last 10 statuses):**")
                status = gr.Markdown("...", elem_id="status-body")

    status_state = gr.State([])

    # on load
    # remove auto load 
    # demo.load(on_load_s, [status_state], [status_state, status])

    # init / keys / ingest / build → use the “_s” wrappers
    btn_init.click( setup_pipeline_s, [dense, rerank, status_state], [status_state, status] )
    btn_key.click(  add_openai_key_s, [key, status_state],          [status_state, status] )
    btn_files.click(add_files_s,      [files, status_state],        [status_state, status] )
    btn_urls.click( add_urls_s,       [urls, status_state],         [status_state, status] )
    btn_build.click(build_index_s,    [status_state],               [status_state, status] )

    # chat output remains the same (streams into `a`)
    btn_ask.click(answer, [q, k, alpha, topk_rerank, stream_toggle, model], [a])

if __name__ == "__main__":
    # Set share=True if you want a public link locally
    demo.launch()
