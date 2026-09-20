"""Document parsing — Milestone 3.

Bytes → ``ParsedDocument`` (title + ordered ``TextUnit`` list carrying
page/section metadata). All parsing is local and deterministic; HTML is
treated as untrusted data (scripts/styles are stripped, never executed).

Supported ``document_type`` values (canonical): pdf, markdown, txt, html.
Aliases md/htm are accepted and normalized. Anything else raises
``ParsingError`` — never silently swallowed (M3 requirement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO

CANONICAL_TYPES = ("pdf", "markdown", "txt", "html")
_TYPE_ALIASES = {"md": "markdown", "htm": "html", "text": "txt"}


class ParsingError(ValueError):
    """Raised for unsupported types, undecodable bytes, or corrupt files."""


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    title: str | None
    document_type: str
    units: tuple[TextUnit, ...] = field(default_factory=tuple)

    @property
    def normalized_text(self) -> str:
        return "\n".join(u.text for u in self.units if u.text.strip())


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

# Smallest explicit resource guard (see ADR-017): bound input size and
# strip NUL bytes, which PostgreSQL TEXT/JSON reject outright (a corrupt
# or hostile file must fail closed with ParsingError, never crash a query).
MAX_DOCUMENT_BYTES = 10_000_000


def normalize_document_type(document_type: str) -> str:
    key = document_type.strip().lower().lstrip(".")
    key = _TYPE_ALIASES.get(key, key)
    if key not in CANONICAL_TYPES:
        raise ParsingError(f"unsupported document_type: {document_type!r}")
    return key


def parse_bytes(raw: bytes, *, document_type: str, source: str = "") -> ParsedDocument:
    """Parse raw file bytes into a ``ParsedDocument``.

    ``source`` is used only for error messages. Empty-but-valid inputs
    (blank TXT/MD/HTML, text-less PDF) yield zero units — the ingestion
    layer persists those as chunk-less documents. Corrupt inputs raise.
    Inputs over ``MAX_DOCUMENT_BYTES`` are rejected; NUL bytes are
    stripped because PostgreSQL cannot store them.
    """
    doc_type = normalize_document_type(document_type)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ParsingError(f"document exceeds {MAX_DOCUMENT_BYTES} bytes ({len(raw)} given)")
    doc_type = normalize_document_type(document_type)
    if doc_type == "txt":
        return _parse_txt(raw)
    if doc_type == "markdown":
        return _parse_markdown(raw)
    if doc_type == "html":
        return _parse_html(raw)
    return _parse_pdf(raw, source=source)


def _decode_text(raw: bytes, *, kind: str) -> str:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParsingError(f"cannot decode {kind} bytes as UTF-8: {exc}") from exc
    return text.replace("\x00", "")


def _parse_txt(raw: bytes) -> ParsedDocument:
    text = _decode_text(raw, kind="TXT")
    units = tuple(TextUnit(text=line) for line in text.splitlines() if line.strip())
    return ParsedDocument(title=None, document_type="txt", units=units)


def _parse_markdown(raw: bytes) -> ParsedDocument:
    text = _decode_text(raw, kind="Markdown")
    units: list[TextUnit] = []
    title: str | None = None
    section: str | None = None
    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line.strip())
        if match:
            section = match.group(2).strip() or None
            if match.group(1) == "#" and title is None and section:
                title = section
            continue  # headings mark sections; body lines carry the text
        if line.strip():
            units.append(TextUnit(text=line.strip(), section=section))
    return ParsedDocument(title=title, document_type="markdown", units=tuple(units))


def _parse_html(raw: bytes) -> ParsedDocument:
    from bs4 import BeautifulSoup

    text = _decode_text(raw, kind="HTML")
    try:
        soup = BeautifulSoup(text, "html.parser")
    except Exception as exc:
        raise ParsingError(f"cannot parse HTML: {exc}") from exc
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    title: str | None = None
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    units: list[TextUnit] = []
    section: str | None = None
    body = soup.body if soup.body else soup
    for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]):
        content = tag.get_text(separator=" ", strip=True)
        if not content:
            continue
        if tag.name.startswith("h"):
            section = content
            if tag.name == "h1" and title is None:
                title = content
            continue
        units.append(TextUnit(text=content, section=section))
    return ParsedDocument(title=title, document_type="html", units=tuple(units))


def _parse_pdf(raw: bytes, *, source: str = "") -> ParsedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    if not raw.strip():
        raise ParsingError(f"empty PDF bytes{f' for {source}' if source else ''}")
    try:
        reader = PdfReader(BytesIO(raw))
    except (PdfReadError, ValueError, EOFError) as exc:
        raise ParsingError(f"cannot parse PDF{f' for {source}' if source else ''}: {exc}") from exc
    except Exception as exc:  # pypdf raises varied errors on corrupt input
        raise ParsingError(f"cannot parse PDF{f' for {source}' if source else ''}: {exc}") from exc
    title: str | None = None
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            title = str(meta.title).replace("\x00", "") or None
    except Exception:
        title = None
    units: list[TextUnit] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").replace("\x00", "")
        except Exception as exc:
            raise ParsingError(f"cannot extract text from PDF page {page_number}: {exc}") from exc
        for line in text.splitlines():
            if line.strip():
                units.append(TextUnit(text=line.strip(), page=page_number))
    return ParsedDocument(title=title, document_type="pdf", units=tuple(units))
