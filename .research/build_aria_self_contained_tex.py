import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = ROOT / "paper" / "ARIA_complete_research_paper.tex"
SOURCE_BIB = ROOT / "paper" / "references.bib"
OUTPUT_TEX = ROOT / "paper" / "ARIA_complete_IEEE_self_contained.tex"


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"' and depth == 0:
            quoted = not quoted
        elif not quoted:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == delimiter and depth == 0:
                parts.append(text[start:index].strip())
                start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def strip_wrapper(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == "{" and value[-1] == "}") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


def parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    cursor = 0
    while True:
        match = re.search(r"@([A-Za-z]+)\s*\{", text[cursor:])
        if not match:
            break
        entry_type = match.group(1).lower()
        opening = cursor + match.end() - 1
        depth = 0
        quoted = False
        escaped = False
        closing = None
        for index in range(opening, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                quoted = not quoted
            if quoted:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            raise ValueError("Unbalanced BibTeX entry")
        payload = text[opening + 1 : closing]
        key, fields_text = payload.split(",", 1)
        fields: dict[str, str] = {"ENTRYTYPE": entry_type}
        for part in split_top_level(fields_text):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            fields[name.strip().lower()] = strip_wrapper(value)
        entries[key.strip()] = fields
        cursor = closing + 1
    return entries


def ascii_latex(value: str) -> str:
    replacements = {
        "–": "--", "—": "---", "−": "-", "‐": "-", "‑": "-",
        "“": "``", "”": "''", "‘": "`", "’": "'",
        "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss", "ı": "i",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    output: list[str] = []
    for index, char in enumerate(value):
        if char in "&%#_" and (index == 0 or value[index - 1] != "\\"):
            output.append("\\" + char)
        else:
            output.append(char)
    return "".join(output).replace("  ", " ").strip()


def initials(given: str) -> str:
    result: list[str] = []
    for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", ascii_latex(given)):
        pieces = token.split("-")
        result.append("-".join(piece[0].upper() + "." for piece in pieces if piece))
    return " ".join(result)


def format_person(person: str) -> str:
    person = person.strip()
    if "," in person:
        family, given = [part.strip() for part in person.split(",", 1)]
        return f"{initials(given)} {ascii_latex(family)}".strip()
    tokens = person.split()
    if len(tokens) == 1:
        return ascii_latex(person)
    return f"{initials(' '.join(tokens[:-1]))} {ascii_latex(tokens[-1])}".strip()


def format_authors(raw: str) -> str:
    people = re.split(r"\s+and\s+", raw.strip())
    formatted = [format_person(person) for person in people]
    if len(formatted) > 6:
        return formatted[0] + " et al."
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return formatted[0] + " and " + formatted[1]
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def format_bibitem(key: str, fields: dict[str, str]) -> str:
    authors = format_authors(fields.get("author", "Unknown author"))
    title = ascii_latex(fields.get("title", "Untitled"))
    year = ascii_latex(fields.get("year", "n.d."))
    entry_type = fields["ENTRYTYPE"]
    pieces = [f"{authors}, ``{title},''"]
    if entry_type == "inproceedings":
        venue = ascii_latex(fields.get("booktitle", "Conference proceedings"))
        pieces.append(f"in \\emph{{{venue}}},")
    else:
        venue = ascii_latex(fields.get("journal", fields.get("publisher", "")))
        if venue:
            pieces.append(f"\\emph{{{venue}}},")
    if fields.get("volume"):
        pieces.append(f"vol. {ascii_latex(fields['volume'])},")
    if fields.get("number"):
        pieces.append(f"no. {ascii_latex(fields['number'])},")
    if fields.get("pages"):
        pieces.append(f"pp. {ascii_latex(fields['pages'])},")
    pieces.append(f"{year}.")
    doi = fields.get("doi", "").strip()
    url = fields.get("url", "").strip()
    eprint = fields.get("eprint", "").strip()
    if doi:
        safe_doi = ascii_latex(doi)
        pieces.append(f"doi: \\href{{https://doi.org/{safe_doi}}}{{{safe_doi}}}.")
    elif url:
        pieces.append(f"[Online]. Available: \\url{{{url}}}.")
    elif eprint:
        safe_eprint = ascii_latex(eprint)
        pieces.append(f"[Online]. Available: \\url{{https://arxiv.org/abs/{safe_eprint}}}.")
    return f"\\bibitem{{{key}}} " + " ".join(pieces)


def citation_order(tex: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        for key in (item.strip() for item in group.split(",")):
            if key and key not in seen:
                seen.add(key)
                order.append(key)
    return order


def main() -> None:
    tex = SOURCE_TEX.read_text(encoding="utf-8")
    entries = parse_bibtex(SOURCE_BIB.read_text(encoding="utf-8"))
    order = citation_order(tex)
    if len(entries) != 70 or len(order) != 70 or set(entries) != set(order):
        raise RuntimeError(
            f"Reference mismatch: entries={len(entries)}, cited={len(order)}, "
            f"missing={sorted(set(order) - set(entries))}, uncited={sorted(set(entries) - set(order))}"
        )
    bibliography = "\n".join(format_bibitem(key, entries[key]) for key in order)
    replacement = "\\begin{thebibliography}{99}\n" + bibliography + "\n\\end{thebibliography}"
    old = "\\bibliographystyle{IEEEtran}\n\\bibliography{references}"
    if old not in tex:
        raise RuntimeError("Expected bibliography block was not found")
    output = "% Self-contained IEEE manuscript. Compile twice with pdfLaTeX.\n" + tex.replace(old, replacement, 1)
    OUTPUT_TEX.write_text(output, encoding="ascii", newline="\n")
    print(OUTPUT_TEX)


if __name__ == "__main__":
    main()
