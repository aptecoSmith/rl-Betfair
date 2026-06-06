"""Web scraping front door: generic fetch + a dedicated ReadyAPI (SmartBear) provider.

Pure helpers (html_to_text, extract_links, looks_js_heavy) are unit-tested. The default network
fetch uses stdlib urllib (zero-dependency); scraped pages are cached under .runtime/scrape/ and the
URL is registered (reference-not-copy) for the agent to ingest. Same-domain link extraction lets the
ReadyAPI provider walk a docs section.

JS-rendered docs sites (SPA shells that ship little static HTML) need a real browser. That is an
opt-in power-up via Playwright - see fetch_rendered()/--fetch-mode and scripts/requirements-scrape.txt.
The core never imports playwright unless a page actually needs it.
"""
from __future__ import annotations

import html as _html
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki_tool as wt  # noqa: E402

PROVIDERS = {
    "readyapi": "https://support.smartbear.com/readyapi/docs/",
    "generic": "",
}
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_ROOT_SHELL = re.compile(r"""<div[^>]+id=["'](?:root|app|__next)["']""", re.I)

PLAYWRIGHT_INSTALL_HELP = (
    "Browser rendering is an optional power-up. Install it with "
    "`pip install -r scripts/requirements-scrape.txt` then `python -m playwright install chromium`, "
    "or rerun with `--fetch-mode http` to scrape the static HTML only."
)


def html_to_text(html: str) -> str:
    """Strip a web page down to readable text. Pure - no network."""
    html = _SCRIPT_STYLE.sub(" ", html)
    html = re.sub(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", html, flags=re.I)
    text = _TAG.sub("", html)
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def page_title(html: str) -> str:
    m = _TITLE.search(html)
    return _html.unescape(m.group(1).strip()) if m else ""


def looks_js_heavy(html: str) -> bool:
    """Does this page need a browser to render meaningful content? Pure - no network."""
    lowered = html.lower()
    if "please enable javascript" in lowered or "enable javascript to" in lowered:
        return True
    # SPA shell: a mount node (#root/#app/#__next) and almost no readable text without JS.
    if _ROOT_SHELL.search(html) and len(html_to_text(html)) < 200:
        return True
    return False


def extract_links(html: str, base_url: str, same_domain=True):
    """Absolute links found in the page. Pure - no network."""
    base_host = urlparse(base_url).netloc
    out, seen = [], set()
    for href in _HREF.findall(html):
        if href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        if same_domain and urlparse(absolute).netloc != base_host:
            continue
        seen.add(absolute)
        out.append(absolute.split("#")[0])
    return out


def _slug(url: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")
    return s[:80] or "page"


def fetch(url: str, timeout=20) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "llm-wiki-v2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (intended fetch)
        charset = r.headers.get_content_charset() or "utf-8"
        return r.read().decode(charset, errors="replace")


def _playwright_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("playwright") is not None


def fetch_playwright(url: str, timeout: int = 120000) -> str:
    """Network. Render the page in a headless browser so the JS-built DOM is captured.

    Optional: needs the `playwright` package (see scripts/requirements-scrape.txt). Lazy-imported so
    the core stays zero-dependency.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # ImportError, or a broken install
        raise RuntimeError(PLAYWRIGHT_INSTALL_HELP) from exc
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent="llm-wiki-v2/1.0")
            page.goto(url, wait_until="networkidle", timeout=timeout)
            return page.content()
        finally:
            browser.close()


def fetch_rendered(url: str, fetch_mode: str = "auto", timeout: int = 20):
    """Return (html, mode_used). Fetch strategy:

    - http: stdlib urllib only, never launches a browser.
    - playwright: always render in a headless browser.
    - auto (default): fetch over http, and escalate to playwright *only* if the page looks
      JS-rendered AND playwright is installed. If it looks JS-rendered and playwright is missing,
      raise with install guidance rather than caching an empty SPA shell.
    """
    if fetch_mode == "playwright":
        return fetch_playwright(url), "playwright"
    html = fetch(url, timeout=timeout)
    if fetch_mode == "auto" and looks_js_heavy(html):
        if _playwright_available():
            return fetch_playwright(url), "playwright"
        raise RuntimeError(f"{url} looks JavaScript-rendered (little static text). {PLAYWRIGHT_INSTALL_HELP}")
    return html, "http"


def scrape_to_markdown(url: str, fetch_mode: str = "auto"):
    """Network. Fetch -> cache markdown under .runtime/scrape/ -> register the URL. Returns cache path."""
    html, mode_used = fetch_rendered(url, fetch_mode=fetch_mode)
    title = page_title(html) or url
    text = html_to_text(html)
    cache_dir = wt.ROOT / ".runtime" / "scrape"
    cache_dir.mkdir(parents=True, exist_ok=True)
    md = f"# {title}\n\nSource: {url}\nFetch-mode: {mode_used}\n\n{text}\n"
    path = cache_dir / f"{_slug(url)}.md"
    path.write_text(md, encoding="utf-8")
    wt.register_source(url, is_file=False, title=title, content_type="url")
    return path


def provider_url(provider: str, path: str = "") -> str:
    base = PROVIDERS.get(provider, "")
    return urljoin(base, path) if base else path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="scrape")
    ap.add_argument("url")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="generic")
    ap.add_argument("--fetch-mode", choices=["auto", "http", "playwright"], default="auto",
                    help="auto: http, escalating to a headless browser only if the page is "
                         "JS-rendered and playwright is installed; http: static HTML only; "
                         "playwright: always render in a browser")
    ap.add_argument("--crawl", action="store_true", help="also list same-domain links found")
    args = ap.parse_args(argv)
    url = provider_url(args.provider, args.url) if args.provider != "generic" else args.url
    try:
        path = scrape_to_markdown(url, fetch_mode=args.fetch_mode)
    except Exception as e:  # network errors, missing playwright, etc.
        print(f"fetch failed: {e}", file=sys.stderr)
        return 3
    print(f"cached {path}")
    if args.crawl:
        html, _ = fetch_rendered(url, fetch_mode=args.fetch_mode)
        for link in extract_links(html, url)[:50]:
            print(link)
    return 0


if __name__ == "__main__":
    sys.exit(main())
