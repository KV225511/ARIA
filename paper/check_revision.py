from pathlib import Path
import re
import json
import hashlib
import pdfplumber

root=Path(__file__).resolve().parents[1]
tex=(root/'paper/ARIA_revised_journal_draft.tex').read_text(encoding='utf-8')
citations=set(re.findall(r'\\cite\{([^}]+)\}',tex))
bibliography=set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
assert len(citations)==11 and citations==bibliography
references=set(re.findall(r'\\ref\{([^}]+)\}',tex))
labels=re.findall(r'\\label\{([^}]+)\}',tex)
assert references<=set(labels)
assert len(labels)==len(set(labels))
assert len(re.findall(r'\\begin\{figure\}',tex))==5
assert len(re.findall(r'\\begin\{(?:table|longtable)\}',tex))==9
assert len(re.findall(r'\\begin\{equation\}',tex))==6
pdfpath=root/'output/pdf/ARIA_revised_journal_draft.pdf'
with pdfplumber.open(pdfpath) as pdf:
    assert len(pdf.pages)>=14
    extracted='\n'.join(p.extract_text() for p in pdf.pages)
    assert '\\' not in extracted
    assert not any(c['x0']<40 or c['x1']>p.width-40 for p in pdf.pages for c in p.chars)
    for expected in ['0.9121','0.9168','0.6030','0.7516','0.8015','0.7976','0.6887','0.4618','0.1055','0.4667','0.5333','32.16']:
        assert expected in extracted,expected
    assert 'SOTA ranking.' in extracted
report={'status':'PASS_WITH_DOCUMENTED_LIMITS','pdf_pages':len(pdf.pages),'citations':11,'figures':5,'tables':9,'equations':6,
        'source_sha256':hashlib.sha256((root/'paper/ARIA_revised_journal_draft.tex').read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256(pdfpath.read_bytes()).hexdigest(),
        'limits':['Native TeX compilation unavailable','ScienceDirect reference format and length not verified','Historical baselines trace to summaries, not recovered prediction artifacts']}
(root/'paper/ARIA_revision_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
