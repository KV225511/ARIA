from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
src=root/'paper/ARIA_complete_research_paper.md'
dst=root/'paper/ARIA_complete_research_paper.tex'
lines=src.read_text(encoding='utf-8').splitlines()
keys={i:k for i,k in enumerate((root/'paper/reference_order.txt').read_text(encoding='utf-8').splitlines(),1)}

def tex(s):
    s=re.sub(r'\*\*([^*]+)\*\*',r'\\textbf{\1}',s)
    s=re.sub(r'`([^`]+)`',lambda m:'\\texttt{'+escape(m.group(1))+'}',s)
    s=re.sub(r'\[(\d+(?:[–-]\d+)?(?:,\d+)*)\]',cite,s)
    return escape(s,protect=True)

def cite(m):
    nums=[]
    for part in re.split('[,]',m.group(1)):
        if '–' in part or '-' in part:
            a,b=map(int,re.split('[–-]',part)); nums.extend(range(a,b+1))
        else: nums.append(int(part))
    if any(n not in keys for n in nums):
        return m.group(0)
    return '\\cite{'+','.join(keys[n] for n in nums)+'}'

def escape(s,protect=False):
    tokens={}
    if protect:
        pats=[r'\\cite\{[^}]+\}',r'\\textbf\{[^}]+\}',r'\\texttt\{[^}]+\}']
        for pat in pats:
            for m in list(re.finditer(pat,s)):
                tok=f'@@TOK{len(tokens)}@@'; tokens[tok]=m.group(0); s=s.replace(m.group(0),tok,1)
    repl={'&':'\\&','%':'\\%','$':'\\$','#':'\\#','_':'\\_','{':'\\{','}':'\\}','~':'\\textasciitilde{}','^':'\\textasciicircum{}','≥':'$\\geq$','≤':'$\\leq$','×':'$\\times$','→':'$\\rightarrow$'}
    for a,b in repl.items(): s=s.replace(a,b)
    s=s.replace('–','--').replace('—','---').replace('’',"'").replace('“','``').replace('”',"''")
    for tok,val in tokens.items(): s=s.replace(tok,val)
    return s

preamble=r'''\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{cite}
\definecolor{ariaBlue}{HTML}{17365D}
\hypersetup{colorlinks=true,linkcolor=ariaBlue,citecolor=ariaBlue,urlcolor=ariaBlue}
\title{\textcolor{ariaBlue}{ARIA: An Auditable Offline Reinforcement-Learning Architecture for Adaptive Mock Interviews}}
\author{Raghav Sejpal (23BAI0095), Krissh Verma (23BAI0098)\\\\Supervisor: Dhivyaa C R}
\begin{document}
\maketitle
'''
out=[preamble]
i=0; in_abstract=False
while i<len(lines):
    s=lines[i].strip()
    if i<6: i+=1; continue
    if s=='## References': break
    if not s: out.append(''); i+=1; continue
    if s.startswith('|') and i+1<len(lines) and re.match(r'^\|?\s*:?-+',lines[i+1].strip()):
        rows=[[c.strip() for c in s.strip('|').split('|')]]; i+=2
        while i<len(lines) and lines[i].strip().startswith('|'):
            rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i+=1
        n=max(len(r) for r in rows); spec='>{\\raggedright\\arraybackslash}X'*n
        out.extend(['\\begin{table}[htbp]','\\centering','\\small',f'\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}','\\toprule'])
        out.append(' & '.join(tex(c) for c in rows[0])+' \\\\ \\midrule')
        for r in rows[1:]: out.append(' & '.join(tex(c) for c in r+['']*(n-len(r)))+' \\\\')
        out.extend(['\\bottomrule','\\end{tabularx}','\\end{table}']); continue
    m=re.match(r'^(#{2,3})\s+(.*)',s)
    if m:
        level=len(m.group(1)); title=m.group(2)
        if title=='Abstract': out.append('\\begin{abstract}'); in_abstract=True
        else:
            if in_abstract: out.append('\\end{abstract}'); in_abstract=False
            title=re.sub(r'^[IVX]+\.\s*','',title)
            out.append(('\\section{' if level==2 else '\\subsection{')+tex(title)+'}')
        i+=1; continue
    im=re.match(r'^!\[(.*?)\]\((.*?)\)$',s)
    if im:
        cap=im.group(1); path=im.group(2).replace('\\','/')
        label='fig:'+re.sub(r'[^a-z0-9]+','-',Path(path).stem.lower()).strip('-')
        out.extend(['\\begin{figure}[htbp]','\\centering',f'\\includegraphics[width=\\linewidth]{{{escape(path)}}}',f'\\caption{{{tex(cap)}}}',f'\\label{{{label}}}','\\end{figure}']); i+=1; continue
    if s.startswith('**Keywords:**'):
        out.append('\\begin{IEEEkeywords}'+tex(s.split(':',1)[1].strip().lstrip('*').strip())+'\\end{IEEEkeywords}'); i+=1; continue
    if re.match(r'^[-*]\s+',s):
        items=[]
        while i<len(lines) and re.match(r'^[-*]\s+',lines[i].strip()): items.append(re.sub(r'^[-*]\s+','',lines[i].strip())); i+=1
        out.append('\\begin{itemize}'); out.extend('\\item '+tex(x) for x in items); out.append('\\end{itemize}'); continue
    if re.match(r'^\d+\.\s+',s):
        items=[]
        while i<len(lines) and re.match(r'^\d+\.\s+',lines[i].strip()): items.append(re.sub(r'^\d+\.\s+','',lines[i].strip())); i+=1
        out.append('\\begin{enumerate}'); out.extend('\\item '+tex(x) for x in items); out.append('\\end{enumerate}'); continue
    out.append(tex(s)+'\n'); i+=1
if in_abstract: out.append('\\end{abstract}')
out.extend(['\\bibliographystyle{IEEEtran}','\\bibliography{references}','\\end{document}',''])
dst.write_text('\n'.join(out),encoding='utf-8')
print(dst)
