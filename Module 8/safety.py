"""Input-safety helpers, carried over from Capstone Checkpoint 7.1 (Lab 7.1 security middleware).

Every user turn (question, clarification answer) goes through sanitize_user_text() before it
reaches the semantic cache, the retriever or a prompt, and every prompt puts its parts inside
XML tags with escaped content, so neither the user nor a PDF can forge the tag structure.
"""
import re

# A pasted BEGIN...END EMAIL BLOCK is untrusted wholesale, so remove the entire block,
# including the body. Stripping only the header lines leaves the fake body ("...aliens...")
# in the user turn, where it pollutes retrieval and distracts the answer.
_EMAIL_BLOCK_RE = re.compile(r"(?is)begin email block.*?(?:end email block|\Z)")
# [^\S\n]* = whitespace EXCEPT newline. A bare \s* here would swallow the newline after
# "END EMAIL BLOCK" and delete the user's real question on the next line with it.
# A colon is required after a header keyword. With an optional colon, any legitimate
# line whose first word merely starts with to/from/date/subject ("To summarize, ...",
# "Today, ...", "Dates aside, ...") would be deleted wholesale. Block markers need no colon.
_EMAIL_HEADER_RE = re.compile(
    r"(?im)^[^\S\n]*(?:(?:from|to|subject|date)[^\S\n]*:|begin email block|end email block)[^\n]*$"
)
_INJECTION_RE = re.compile(
    r"(?i)\b(ignore|forget|disregard|override)\b[^.\n]*\b(instruction|instructions|rule|rules|context|prompt)\b"
    r"|\byou are now\b|\bact as\b|\bpretend (you|to)\b|\byour new (role|persona)\b|\bfrom now on\b"
)


def escape_xml(text: str) -> str:
    """Escape XML metacharacters so untrusted text cannot forge the prompt's tag structure."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attr(value) -> str:
    """escape_xml for a value placed inside a double-quoted tag attribute."""
    return escape_xml(str(value)).replace('"', "&quot;")


def sanitize_user_text(text: str) -> str:
    """Input middleware: Escape XML tags, strip injected e-mail blocks and neutralize the
    roleplay/ignore-instructions trigger phrases (The rest of an attack sentence may
    remain as inert text. The hardened XML prompt treats it as data, not instructions)

    The output must still contain the user's real question. A sanitizer that deletes the
    question "blocks the attack" but breaks the product. The result is already escaped, so
    it goes into a prompt as is.
    """
    text = escape_xml(text)
    text = _EMAIL_BLOCK_RE.sub("[removed injected e-mail block]", text)
    text = _EMAIL_HEADER_RE.sub("[removed injected header line]", text)
    text = _INJECTION_RE.sub("[removed instruction-injection]", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


REMOVED_MARKER = "[removed "


def was_sanitized(raw: str, sanitized: str) -> bool:
    """True when sanitize_user_text() removed something (a block, a header or a trigger phrase)."""
    return sanitized.count(REMOVED_MARKER) > raw.count(REMOVED_MARKER)


def xml_prompt(*sections: tuple[str, str]) -> str:
    """Wrap already-escaped parts in tags, e.g. xml_prompt(("documents", docs), ("user_question", q)).
    Empty parts are left out."""
    return "\n".join(f"<{tag}>\n{body}\n</{tag}>" for tag, body in sections if body)
