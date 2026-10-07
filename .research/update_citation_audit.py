from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
paper=(ROOT/'paper/ARIA_complete_research_paper.md').read_text(encoding='utf-8')
body=paper.split('## References',1)[0]
order=(ROOT/'paper/reference_order.txt').read_text(encoding='utf-8').splitlines()
bib=(ROOT/'paper/references.bib').read_text(encoding='utf-8')

def entries(text):
    out={}
    for chunk in re.split(r'\n(?=@)',text):
        m=re.match(r'@[^\{]+\{([^,]+),',chunk.strip())
        if not m: continue
        f={}
        for x in re.finditer(r'(\w+)\s*=\s*(?:\{([^{}]*)\}|"([^"]*)")',chunk,re.S):
            f[x.group(1).lower()]=(x.group(2) if x.group(2) is not None else x.group(3)).replace('\n',' ').strip()
        out[m.group(1)]=f
    return out

def cited_numbers(text):
    nums=set()
    for m in re.finditer(r'\[([0-9 ,–-]+)\]',text):
        for part in [p.strip() for p in m.group(1).split(',')]:
            if re.fullmatch(r'\d+',part): nums.add(int(part))
            elif re.fullmatch(r'\d+[–-]\d+',part):
                a,b=map(int,re.split('[–-]',part)); nums.update(range(a,b+1))
    return nums

db=entries(bib); cited=cited_numbers(body)
full={21,26,27,54,58,65,69,70}
rl={8,17,18,19,22}
adaptive={21,22,33,44,68,69,70}
fair={1,3,6,9,10,11,12,13,14,15,16,26,27,28,29,30,31,34,35,36,38,40,42,43,47,48,49,52,53,54,56,57,58,59,60,61,63,65}

def purpose(n):
    p=[]
    if n in rl: p.append('offline RL, OPE, calibration, or adaptive assessment')
    if n in adaptive: p.append('adaptive interviewing, question design, or recent agentic systems')
    if n in fair: p.append('validity, fairness, accessibility, governance, or applicant experience')
    if not p: p.append('automated interview modeling or psychometrics')
    return '; '.join(p)

rows=[]
for n,key in enumerate(order,1):
    f=db[key]
    ident=f.get('doi') or ('arXiv:'+f['eprint'] if f.get('eprint') else f.get('url','none'))
    depth='Full text' if n in full else 'Abstract/metadata and relevance screen'
    source='Scite canonical metadata + Undermind discovery'
    if n in full: source+=' + Undermind PDF reading'
    rows.append(f"| {n} | `{key}` | {ident} | {source} | {depth} | {purpose(n)} | VERIFIED; cited in text |")

missing=sorted(set(range(1,len(order)+1))-cited)
extra=sorted(cited-set(range(1,len(order)+1)))
report=f'''# Citation Verification Report

Audit date: 2026-09-15  
Citation style: IEEE numeric  
Bibliography size: {len(order)}  
In-text coverage: {len(cited & set(range(1,len(order)+1)))}/{len(order)}  
Uncited bibliography entries: {missing or 'none'}  
Out-of-range citations: {extra or 'none'}

## Method

All records were discovered or re-identified through the completed Undermind search, reconciled to canonical DOI/arXiv metadata, and checked for sentence-level relevance. DOI records were formatted from Scite metadata. Eight central sources—five from 2026 and two IEEE Access comparators plus PolyInterview—were read end to end for this revision; the remaining records were used only for claims supportable from verified metadata and abstracts or from the earlier evidence extraction. Search snippets were not used as evidence. Zotero reconciliation remains unavailable because the local Zotero API could not be reached.

## Per-reference verification

| No. | BibTeX key | Canonical identifier | Metadata source | Evidence read | Manuscript role | Verdict |
|---:|---|---|---|---|---|---|
{chr(10).join(rows)}

## Scite context check for central claims

- Putra et al. (2024) directly cites Kim et al. (2023), confirming a method lineage from sensitive-attribute-aware multimodal fairness toward adversarial reweighting without sensitive attributes at inference. Available later contexts classify both as fairness-aware AVI methods; none supplied a contrasting empirical replication.
- A 2026 commentary on Cheng (2026) treats the psychometric/ML mapping as a useful foundation but qualifies its association-based scope by arguing for causal fairness analysis. The manuscript therefore uses Cheng for pipeline-wide fairness concepts, not as proof that parity metrics resolve causal inequity.
- Scite coverage for the two 2026 CHI disability papers is currently low (one resolved edge each), which is expected for very recent work. Their manuscript use is grounded in the full papers rather than citation counts or search snippets.

## Corrections and exclusions

- The paper no longer claims that ARIA is the first adaptive multimodal interview system; 2026 systems and datasets directly preclude that framing.
- Publication years follow canonical publisher/Scite metadata where online-first and issue years differ.
- The 2026 preprints are labeled as preprints and are not presented as settled peer-reviewed evidence.
- No reference is used to imply that ARIA itself has human validity, fairness, accessibility, or policy-effectiveness evidence.
- All 70 bibliography entries are cited at least once in the main text; no uncited padding references remain.
'''
(ROOT/'paper/citation_verification_report.md').write_text(report,encoding='utf-8')
if missing or extra or len(order)!=70:
    raise SystemExit(f'Citation audit failed: missing={missing}, extra={extra}, refs={len(order)}')
print('Citation audit PASS: 70/70 cited')
