"""Small, stdlib-first RAG assistant for a municipal public website."""
import json, math, os, re, sqlite3, sys, threading, time, urllib.parse, urllib.request, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent
DEFAULT_CORPUS = BASE / "corpus" / "pages.json"
STOP = set("a an and are as at be by for from has have how i in is it its of on or that the to was were what when where which who why will with your you can do does need about".split())

def load_config():
    path = BASE / "config.json"
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise RuntimeError("config.json is not valid JSON") from exc

CONFIG = load_config()
def corpus_path(): return Path(os.environ.get("CORPUS_PATH") or CONFIG.get("corpus_path") or DEFAULT_CORPUS)
def local_data_dir():
    if os.environ.get("DATA_DIR"): return Path(os.environ["DATA_DIR"])
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "civic-rag"
DATA_DIR, LOGDB = local_data_dir(), local_data_dir() / "conversations.db"
CONTACT_LABEL = os.environ.get("CONTACT_LABEL") or CONFIG.get("contact_label") or "the municipal office"
CONTACT_PHONE = os.environ.get("CONTACT_PHONE") or CONFIG.get("contact_phone") or ""

def tokens(text):
    return [word for word in re.findall(r"[a-z0-9']+", text.lower()) if word not in STOP and len(word) > 1]

class BM25:
    """Transparent lexical retrieval for small, inspectable corpora."""
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Titles are concise page labels and deserve more weight than repeated
        # navigation-like words inside a long body.
        self.docs = [tokens((item["title"] + " ") * 10 + item["text"]) for item in chunks]
        self.lens, self.avg = [len(doc) for doc in self.docs], sum(map(len, self.docs)) / max(1, len(self.docs))
        self.tf = [Counter(doc) for doc in self.docs]
        df = Counter()
        for doc in self.docs: df.update(set(doc))
        self.idf = {word: math.log(1 + (len(self.docs) - count + .5) / (count + .5)) for word, count in df.items()}
    def search(self, query, top=6):
        query_terms, scored = tokens(query), []
        for index, frequencies in enumerate(self.tf):
            score = sum(self.idf.get(word, 0) * (frequencies[word] * (self.k1 + 1)) /
                        (frequencies[word] + self.k1 * (1 - self.b + self.b * self.lens[index] / self.avg))
                        for word in query_terms if word in frequencies)
            if score: scored.append((score, index))
        # Evaluation and citations are page-level. Avoid letting many chunks from
        # one long page consume every retrieval slot.
        output, seen = [], set()
        for score, index in sorted(scored, reverse=True):
            chunk = self.chunks[index]
            if chunk["url"] in seen: continue
            seen.add(chunk["url"]); output.append((chunk, score))
            if len(output) == top: break
        return output

def load_corpus(path=None):
    path = Path(path or corpus_path())
    if not path.exists(): raise FileNotFoundError(f"No corpus at {path}. Run: py ingest.py https://your-site.example")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("pages") or not data.get("chunks"): raise ValueError("Corpus must contain pages and chunks")
    return data

def settings():
    return {"provider": os.getenv("LLM_PROVIDER", "ollama").lower(),
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            "anthropic_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "gemini_key": os.getenv("GEMINI_API_KEY", ""),
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}

def request_json(url, payload, headers=None, timeout=90):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items(): req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=timeout) as response: return json.load(response)

def complete(system, user, timeout=90):
    """Call ollama, Anthropic, or Gemini without a third-party SDK."""
    cfg, provider = settings(), settings()["provider"]
    if provider == "ollama":
        result = request_json(cfg["ollama_url"] + "/api/chat", {"model": cfg["ollama_model"], "stream": False,
            "think": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0}}, timeout=timeout)
        text = result.get("message", {}).get("content", "")
        # Some qwen3 builds omit the opening tag even when think=false.
        if "</think>" in text: text = text.split("</think>", 1)[1]
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.I | re.S).strip()
    if provider == "anthropic":
        if not cfg["anthropic_key"]: raise RuntimeError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        result = request_json("https://api.anthropic.com/v1/messages", {"model": cfg["anthropic_model"], "max_tokens": 450,
            "system": system, "messages": [{"role": "user", "content": user}]},
            {"x-api-key": cfg["anthropic_key"], "anthropic-version": "2023-06-01"}, timeout)
        return "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text").strip()
    if provider == "gemini":
        if not cfg["gemini_key"]: raise RuntimeError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(cfg["gemini_model"]) + ":generateContent?key=" + urllib.parse.quote(cfg["gemini_key"])
        result = request_json(url, {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 450}}, timeout=timeout)
        candidates = result.get("candidates", [])
        return "".join(part.get("text", "") for part in (candidates[0].get("content", {}).get("parts", []) if candidates else [])).strip()
    raise ValueError("LLM_PROVIDER must be ollama, anthropic, or gemini")

# Safety pre-filter only: personal medical or legal advice is declined before retrieval,
# whatever the website says. Everything else that the site doesn't cover has to be caught
# by the model abstaining (see is_abstention), which is what the eval measures.
OUT_OF_SCOPE = re.compile(r"\b(?:diagnos(?:e|is)|my symptoms?|prescription|dosage|my medication|medical advice|legal advice|should i sue)\b", re.I)
def is_out_of_scope(question): return bool(OUT_OF_SCOPE.search(question))

ABSTAIN = re.compile(r"\b(?:(?:do|does|did) not (?:say|mention|include|contain|cover|provide|address|list|specify)"
    r"|(?:doesn't|don't|didn't) (?:say|mention|include|contain|cover|provide|address|list|specify)"
    r"|not (?:mentioned|covered|included|addressed|specified) in"
    r"|no information|(?:could not|couldn't|can't|cannot) (?:find|answer|determine)"
    r"|(?:is|are) not (?:in|part of) the (?:sources|passages|provided)"
    r"|outside (?:the scope|what)|unable to (?:find|answer))\b", re.I)
def is_abstention(answer): return bool(ABSTAIN.search(answer or ""))
def refusal():
    return "I can only help with information published on this municipality's website." + (f" Please contact {CONTACT_LABEL} at {CONTACT_PHONE}." if CONTACT_PHONE else "")
SYSTEM = """You answer questions about a municipal website. Use only the supplied source passages. Do not infer rules, eligibility, legal, financial, medical, or permit outcomes. If the passages do not answer the question, say that plainly. Be concise and factual. The application attaches source links separately."""

def sources_for(hits):
    seen, output = set(), []
    for chunk, score in hits:
        if chunk["url"] not in seen:
            seen.add(chunk["url"]); output.append({"title": chunk["title"], "url": chunk["url"], "score": round(score, 2)})
        if len(output) == 3: break
    return output

def ask(question, bm25, top=6):
    """Retrieve, answer, and return structured page citations."""
    if is_out_of_scope(question): return {"answer": refusal(), "sources": [], "grounded": False, "refused": True}
    hits = bm25.search(question, top=top)
    if not hits: return {"answer": refusal(), "sources": [], "grounded": False, "refused": True}
    context = "\n\n".join(f"--- SOURCE {n}: {chunk['title']} ---\n{chunk['text'][:2200]}" for n, (chunk, _) in enumerate(hits, 1))
    answer = complete(SYSTEM, f"SOURCES:\n{context}\n\nQUESTION: {question}")
    if not answer: raise RuntimeError("The provider returned an empty answer")
    if is_abstention(answer):
        # The model read the passages and said they don't answer it. Don't attach
        # citations to a non-answer; that would look grounded when it isn't.
        return {"answer": answer, "sources": [], "grounded": False, "refused": False, "abstained": True}
    return {"answer": answer, "sources": sources_for(hits), "grounded": True, "refused": False, "abstained": False}

PII_PATTERNS = [(re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD REDACTED]"), (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[PHONE REDACTED]"),
    (re.compile(r"\b\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Ct|Court|Blvd|Way|Cir|Circle)\b"), "[ADDRESS REDACTED]"),
    (re.compile(r"\b(?:acct|account|policy|permit|case|invoice)\b[^\d\n]{0,15}\d{4,}", re.I), "[ACCOUNT REDACTED]"), (re.compile(r"(?<!\d)\d{6,}(?!\d)"), "[NUMBER REDACTED]")]
def scrub_pii(text):
    for pattern, replacement in PII_PATTERNS: text = pattern.sub(replacement, text)
    return text

def init_log():
    DATA_DIR.mkdir(parents=True, exist_ok=True); db = sqlite3.connect(LOGDB)
    db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, asked_at TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, sources TEXT NOT NULL, grounded INTEGER NOT NULL, channel TEXT NOT NULL)")
    db.commit(); return db
def log_turn(db, question, result, channel):
    db.execute("INSERT INTO conversations VALUES (?,?,?,?,?,?,?)", (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(timespec="seconds"), scrub_pii(question), scrub_pii(result["answer"]), json.dumps(result["sources"]), int(result["grounded"]), channel)); db.commit()

RATE, RATE_LOCK = {}, threading.Lock()
def serve(port=8787):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    corpus, db, widget = load_corpus(), init_log(), (BASE / "widget.html").read_bytes()
    bm25 = BM25(corpus["chunks"])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send_payload(self, status, content_type, payload):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def do_GET(self): self.send_payload(200, "text/html; charset=utf-8", widget) if self.path in ("/", "/index.html", "/widget.html") else self.send_payload(404, "text/plain", b"not found")
        def do_POST(self):
            if self.path != "/ask": return self.send_payload(404, "text/plain", b"not found")
            try: question = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("q", "")).strip()[:600]
            except (ValueError, OSError): return self.send_payload(400, "application/json", b'{"error":"bad json"}')
            if not question: return self.send_payload(400, "application/json", b'{"error":"empty question"}')
            now, ip = time.time(), self.client_address[0]
            with RATE_LOCK:
                recent = [stamp for stamp in RATE.get(ip, []) if now - stamp < 3600]
                if len(recent) >= 60 or sum(now - stamp < 60 for stamp in recent) >= 15: return self.send_payload(429, "application/json", b'{"error":"rate limited"}')
                RATE[ip] = recent + [now]
            try: result = ask(question, bm25)
            except Exception: result = {"answer": "I could not complete that request. Please try again later.", "sources": [], "grounded": False, "refused": False}
            log_turn(db, question, result, "web"); self.send_payload(200, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode())
    host, port = ("0.0.0.0", int(os.environ["PORT"])) if os.environ.get("PORT") else ("127.0.0.1", port)
    print(f"Civic RAG listening on http://{host}:{port}"); HTTPServer((host, port), Handler).serve_forever()

if __name__ == "__main__":
    if "--serve" in sys.argv: serve()
    else:
        question = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--"))
        if not question: print(__doc__); raise SystemExit(0)
        result = ask(question, BM25(load_corpus()["chunks"])); db = init_log(); log_turn(db, question, result, "cli"); print(result["answer"])
        for source in result["sources"]: print(f"[source] {source['title']}")
