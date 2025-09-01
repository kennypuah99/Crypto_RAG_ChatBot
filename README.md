# Crypto_RAG_ChatBot
 This is a cryptocurrency-focused Retrieval-Augmented Generation (RAG) app. It retrieves from your uploaded documents and added URLs, reranks results, and generates  answers with a chat LLM. It can also route "price" queries to a live price tool for major coins.

1) OPENAI API KEY (REQUIRED; PASTE-ONLY, NEVER SAVED)
-----------------------------------------------------
- The chat model requires an OpenAI API key.
- Paste your key into the "OpenAI API Key" field in the UI.
- The key is kept in memory for the current session only:
  - It is not written to disk, not bundled in the repository, and not logged.
  - Restarting or refreshing the Space clears it.
- The key is used only to generate chat responses for this app.

2) QUICK COST ESTIMATE (GPT-4o-mini PRICING)
--------------------------------------------
Pricing used: (https://platform.openai.com/docs/pricing)
- Input:  $0.15 per 1,000,000 tokens
- Output: $0.60 per 1,000,000 tokens

Assumptions:
- 200 input tokens per query
- 500 output tokens per query
- 20 queries total

Per-query cost:
- Input:  200 / 1,000,000 * $0.15 = $0.00003
- Output: 500 / 1,000,000 * $0.60 = $0.00030
- Total per query = $0.00003 + $0.00030 = $0.00033

20-query session:
- $0.00033 * 20 = $0.0066

Result: Approximately six-tenths of a cent ($0.0066) for the full try, well under the $0.50 budget.

3) FREE OPEN-SOURCE RETRIEVAL MODELS
------------------------------------
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Reranker:   cross-encoder/ms-marco-MiniLM-L-6-v2
These run locally in the Space and are free (no API costs).

4) UPLOADING .PDF / .TXT / .MD FILES (FOR RAG)
----------------------------------------------
- Click "Add files" and select any combination of .pdf, .txt, or .md.
- The app extracts text (PDFs via pypdf), splits into chunks, and stores them
  with metadata for retrieval.
- After adding files, click "Build Index" so they are included in search.

5) ADDING MULTIPLE URLS (FOR RAG)
---------------------------------
- Paste one URL per line in the "URLs" box and click "Add URLs".
- The app fetches and parses the pages (static articles and PDFs work best).
- After adding URLs, click "Build Index" to include them in retrieval.

6) BUILD INDEX AND RETRIEVAL OPTIONS
------------------------------------
After you add files and/or URLs, click "Build Index". You can tune:

- Top-K retrieve (k): how many candidates to pull initially (e.g., 6 to 10).
- Hybrid alpha (BM25 <-> Dense): blend between keyword BM25 and dense
  similarity. 0.0 = all dense, 1.0 = all BM25. A balanced default is 0.5.
- Rerank Top-K: how many of the retrieved candidates the cross-encoder reranks
  (e.g., 3 to 8). The final answer uses the top reranked passages.

If results seem off:
- Increase Top-K for better recall.
- Adjust alpha (higher favors keywords; lower favors semantic similarity).
- Increase Rerank Top-K for stronger final ordering (slightly more CPU).

7) STREAMING RESPONSES (SELECTABLE)
---------------------------------
- Streaming ON: words appear live as the model generates. 
- Streaming OFF: you receive a single final answer after generation finishes.
Toggle this with the "Streaming" checkbox.

8) CHAT MODEL CHOICE (FIXED FOR NOW)
------------------------------------
- The chat model is fixed in this version of the app for reliability and cost
  control. If you need a different model, it can be changed in code and
  redeployed; the UI currently does not expose a model selector.

9) LIVE PRICE SEARCH FOR MAJOR COINS + ROUTING
----------------------------------------------
- If your question looks like a price query (for example: "BTC price",
  "price of ETH", "SOL price in USD"), the app routes to a tool instead of RAG:
  - It calls a public price API (for example, CoinGecko) to get the latest
    price for major coins such as BTC, ETH, SOL, and XRP.
  - It can also show the Fear and Greed Index for market sentiment.
- Routing logic: the pipeline checks your query for price-intent keywords
  ("price", "quote", "market cap", "ATH", and similar). If matched, it uses
  the tools route; otherwise it uses the RAG route (retrieve -> rerank -> answer).

10) QUICK START
---------------
1. Initialize pipeline (if manual init is enabled).
2. Paste your OpenAI API key (not saved).
3. Add files and/or add URLs.
4. Click Build Index.
5. Ask questions. E.g. "what is Ethereum vs Solana?", "What is bitcoin strength and weakness?"
6. For prices, try queries like "ETH price", "SOL quote", "XRP price in USD".

NOTES
-----
- This tool is for research and education only. It is not financial advice.
- For best results, use focused, well-structured documents and reputable URLs.



### Installation and Execution (for Gradio UI)
---------------
1. **Create a new Python environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**

   For macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

   For Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```