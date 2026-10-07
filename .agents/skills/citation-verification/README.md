# citation-verification

A Skill that forces an AI assistant to verify every academic citation it produces, instead of hallucinating DOIs or misattributing claims.

## Behavior chain

Whenever the assistant cites a paper to support a claim:

1. **Verify existence** — confirm the DOI / PMID / arXiv ID resolves
2. **Read the abstract** — never cite from the title alone
3. **Judge relevance** — drop if the paper doesn't actually support the claim
4. **Fetch full text when needed** — for specific numbers, methods, subgroup results

Output each citation as: `[Author et al., Year, Title, Journal] (DOI) + conclusion + verbatim quote (≤ 100 chars)`.

If the abstract is unreachable, mark `[no accessible abstract]` instead of fabricating a quote.

## Installation

| Platform | Path |
|---|---|
| OpenCode (user) | `~/.config/opencode/skills/citation-verification/` |
| OpenCode (project) | `<project>/.opencode/skills/citation-verification/` |
| Claude Code / Copilot CLI / Gemini CLI | per platform's skill convention |

## Recommended MCP tools

The skill is tool-agnostic, but pairing it with scholarly lookup tools gives best results:

- **academic-research** — DOI validation, multi-source search, OA PDF resolution
- **semantic-scholar** — paper metadata, full-text Markdown conversion
- **sciverse** — semantic search, full-text reading by byte offset
- **tavily** — web search fallback for DOI landing pages

## License

MIT.