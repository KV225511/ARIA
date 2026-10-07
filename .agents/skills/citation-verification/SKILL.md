---
name: citation-verification
description: >
  Verify academic citations for existence and relevance before using them as
  evidence. Triggers whenever the assistant cites a paper, study, or scholarly
  source (DOI, PMID, arXiv ID, journal article, preprint) to support a claim,
  conclusion, or fact. Forbids citing from memory or generative guessing.
  Triggers: cite paper, citation, reference, according to study, literature
  review, bibliography, scholarly source, support with research, 引用文献,
  参考文献, 写综述.
---

# Citation Verification

## Goal

Eliminate **fabricated citations** and **misattributed citations**. Every claim backed by a scholarly source must satisfy three conditions:

1. The work **really exists**.
2. You have **actually read its abstract**.
3. Its content **genuinely supports** your claim.

If any one fails → **do not cite it**.

---

## Mandatory workflow (run for every citation)

### Step 1 — Verify existence

Confirm the identifier (DOI / PMID / arXiv ID) resolves to a real record.

Use whatever verification capability is available in the current environment, in this order of preference:

1. A dedicated citation/DOI validation tool, if present
2. A scholarly search/metadata API (Semantic Scholar, OpenAlex, CrossRef, PubMed, etc.)
3. Web fetch against `https://doi.org/{DOI}` or the publisher page as a fallback

**Forbidden**: guessing a DOI from author + topic + year. If you don't know the DOI, search first, then verify.

If the identifier does not resolve → **drop the citation**. Replace it or remove the supported claim.

### Step 2 — Confirm you read the abstract

Ask yourself: have I actually seen the abstract of this work in this conversation?

- Yes → continue to Step 3.
- No → **retrieve it now**, using whichever is available:
  - A paper-lookup tool that returns metadata + abstract
  - A scholarly search API result that includes the abstract field
  - As a last resort, fetch the DOI landing page and read the abstract there

**Never** rely on the title alone or on a single line from a search-result list.

### Step 3 — Judge relevance

After reading the abstract, decide honestly:

- Does it **directly address** the claim I am making?
- Or is it **topically near but conclusionally different / unrelated**?
- Or does it actually point the **opposite way** (and I'd be misusing it)?

| Relevance | Action |
|---|---|
| Directly supports | Cite it |
| Indirectly relevant | Cite it, but mark the link as indirect in the prose |
| Unrelated / contradictory | **Drop it** |

### Step 4 — Read the full text when needed

Pull the full text (or the relevant section) when **any** of the following holds:

- The abstract is not enough to back your specific wording (e.g. you quote a number, a method detail, a subgroup result).
- The user explicitly asks for evidence grounded in the body of the paper.
- The citation is a core evidence item in a review or meta-analysis.

Use whatever full-text retrieval path is available: a fulltext/Markdown conversion tool, an open-access PDF link followed by a fetch, or the publisher's HTML article. After reading, re-check: does the body really say what the abstract implied? Are the numbers and direction consistent?

---

## Output requirements

When presenting a citation to the user, each one must satisfy:

- Show the **verified identifier** (DOI, PMID, or arXiv ID).
- The link between the citation and the claim is one you confirmed by reading the abstract — not a literal title match.
- If you only read the abstract and not the body, **do not fabricate** body-level specifics (exact p-values, subgroup sample sizes, methodological details).

Recommended format:

```
[Author et al., Year, Title, Journal]
(DOI: 10.xxxx/xxxxx)
Short conclusion
The most core and relevant verbatim quote from the source (≤ 100 chars)
```
**Fallback when the abstract is unreachable** (paywalled, withdrawn, no API coverage, or the page genuinely returns nothing usable): **omit the verbatim quote line entirely** and replace it with `[no accessible abstract]`. Never fabricate a quote to fill the slot. If you additionally cannot verify the work's existence at all, drop the citation per Step 1.

---

## Red lines (any one is a serious failure)

- Writing a DOI from memory without verifying it
- Citing based on the title alone
- Keeping a citation after the lookup returned "not found"
- Using a "related but opposite-conclusion" paper as supporting evidence
- Fabricating body-level details when you only read the abstract

---

## One-line creed

> **No ID check, no cite. No abstract read, no cite. No topical match, no cite.**
