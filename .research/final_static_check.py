from pathlib import Path
import ast, hashlib, json, re

root=Path(__file__).resolve().parents[1]
md=(root/'paper/ARIA_complete_research_paper.md').read_text(encoding='utf-8')
tex=(root/'paper/ARIA_complete_research_paper.tex').read_text(encoding='utf-8')
bib=(root/'paper/references.bib').read_text(encoding='utf-8')
body=md.split('## References',1)[0]
nums=set()
for m in re.finditer(r'\[([0-9 ,–-]+)\]',body):
    for part in (p.strip() for p in m.group(1).split(',')):
        if part.isdigit(): nums.add(int(part))
        elif re.fullmatch(r'\d+[–-]\d+',part):
            a,b=map(int,re.split('[–-]',part)); nums.update(range(a,b+1))
headings=['## Abstract','## I. Introduction','## II. Related Work / Literature Review','## III. Proposed Model / Methodology','## IV. Results and Analysis','## V. Conclusion','## References']
bibkeys=re.findall(r'^@[^\{]+\{([^,]+),',bib,re.M)
citedkeys={k for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')}
assert len(bibkeys)==70
assert nums==set(range(1,71))
assert all(h in md for h in headings)
assert set(bibkeys)==citedkeys
assert tex.count('{')==tex.count('}')
assert r'\documentclass[journal]{IEEEtran}' in tex
assert r'\bibliographystyle{IEEEtran}' in tex
for name in ['revise_ieee_paper.py','update_citation_audit.py','markdown_to_latex.py','build_artifacts.py','make_paper_figure.py']:
    ast.parse((root/'.research'/name).read_text(encoding='utf-8'))
state=json.loads((root/'.paper/manuscript_state.json').read_text(encoding='utf-8'))
paths={'main-md':'paper/ARIA_complete_research_paper.md','main-docx':'paper/ARIA_complete_research_paper.docx','main-pdf':'paper/ARIA_complete_research_paper.pdf','main-tex':'paper/ARIA_complete_research_paper.tex','claims':'.paper/claims.yml','references':'paper/references.bib'}
for key,path in paths.items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==state['release']['checked_hashes'][key]
assert not re.search(r'(?i)TODO|TBD|PLACEHOLDER|lorem ipsum',md)
for section in ['II. Related Work / Literature Review','III. Proposed Model / Methodology','IV. Results and Analysis']:
    start=md.index('## '+section); end=md.find('\n## ',start+4); text=md[start:end if end!=-1 else None]
    assert len(set(re.findall(r'\[(21|26|27|54|68|69|70)\]',text)))>=3
print('STATIC PASS: sections=7 refs=70 cited=70 latex_keys=70 braces=balanced 2026_sources>=3_per_required_section scripts=parse state=json')
