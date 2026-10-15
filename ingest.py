"""Crawl a public site and create a small, inspectable BM25 corpus.

Usage: py ingest.py https://your-municipality.example
"""
import json, re, sys, time, urllib.parse, urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE, OUT = Path(__file__).parent, Path(__file__).parent / "corpus" / "pages.json"
SKIP = re.compile(r"(wp-content|wp-admin|\.pdf$|\.jpg$|\.png$|\.docx?$|#|mailto:|tel:)", re.I)

class Extractor(HTMLParser):
    DROP = {"script", "style", "nav", "footer", "svg", "noscript", "form"}
    def __init__(self):
        super().__init__(); self.parts, self.links, self.skip, self.title, self.in_title = [], [], 0, "", False
    def handle_starttag(self, tag, attrs):
        if tag in self.DROP: self.skip += 1
        if tag == "title": self.in_title = True
        if tag == "a": self.links.extend(value for key, value in attrs if key == "href" and value)
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "br"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in self.DROP and self.skip: self.skip -= 1
        if tag == "title": self.in_title = False
    def handle_data(self, data):
        if self.in_title: self.title += data.strip()
        elif not self.skip and data.strip(): self.parts.append(data.strip() + " ")
    def text(self):
        return "\n".join(line.strip() for line in re.sub(r"[ \t]+", " ", "".join(self.parts)).split("\n") if len(line.strip()) > 2)

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "civic-rag-example/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore") if "html" in response.headers.get("Content-Type", "").lower() else None

def chunk(text, title, url, size=1100, overlap=150):
    output, current = [], ""
    for paragraph in (item for item in text.split("\n") if item.strip()):
        if len(current) + len(paragraph) + 1 > size and current:
            output.append(current.strip()); current = current[-overlap:] + "\n" + paragraph
        else: current += "\n" + paragraph
    if current.strip(): output.append(current.strip())
    return [{"title": title, "url": url, "text": item} for item in output if len(item) > 120]

def crawl(root, max_pages=60):
    root = root.rstrip("/") + "/"; seen, queue, pages = set(), [root], []
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen or SKIP.search(url): continue
        seen.add(url)
        try: html = fetch(url)
        except Exception as exc: print(f"skip {url}: {exc}"); continue
        if not html: continue
        parser = Extractor(); parser.feed(html); text = parser.text()
        if len(text) < 200: continue
        title = re.sub(r"\s+", " ", parser.title).strip() or urllib.parse.urlparse(url).path.strip("/") or "Home"
        pages.append({"url": url, "title": title[:120], "text": text})
        for href in parser.links:
            candidate = urllib.parse.urljoin(url, href).split("#", 1)[0]
            if candidate.startswith(root) and candidate not in seen and not SKIP.search(candidate): queue.append(candidate)
        time.sleep(.25)
    return pages

def main():
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"): raise SystemExit("Usage: py ingest.py https://your-municipality.example")
    root = sys.argv[1]; pages = crawl(root); chunks = [part for page in pages for part in chunk(page["text"], page["title"], page["url"])]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "root": root, "pages": pages, "chunks": chunks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(pages)} pages and {len(chunks)} chunks to {OUT}")
if __name__ == "__main__": main()
