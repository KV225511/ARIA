"""Render the journal draft to PDF independently of the unavailable native TeX compiler.

Prose, tables, references and captions are extracted from the authoritative .tex.
Vector figures and equation typography are equivalent fallback renderings.
"""
from pathlib import Path
import re
import html
import json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               KeepTogether, Flowable, CondPageBreak)
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
pdf=OUT/'ARIA_revised_journal_draft.pdf'
raw=(ROOT/'paper/ARIA_revised_journal_draft.tex').read_text(encoding='utf-8')
for name,file in [('Serif','times.ttf'),('SerifBold','timesbd.ttf'),('SerifItalic','timesi.ttf'),('Sans','arial.ttf'),('SansBold','arialbd.ttf'),('Math','seguisym.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='SerifItalic',boldItalic='SerifBold')
W=A4[0]-112
blue=colors.HexColor('#235789'); gray=colors.HexColor('#526170'); teal=colors.HexColor('#267A73')
styles={
 'body':ParagraphStyle('Body',fontName='Serif',fontSize=10.5,leading=14,alignment=TA_JUSTIFY,spaceAfter=6),
 'h1':ParagraphStyle('H1',fontName='SansBold',fontSize=13,leading=17,spaceBefore=15,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('H2',fontName='SansBold',fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=6,keepWithNext=True),
 'title':ParagraphStyle('Title',fontName='SerifBold',fontSize=19,leading=23,alignment=TA_CENTER,spaceAfter=12),
 'center':ParagraphStyle('Center',fontName='Serif',fontSize=10,leading=14,alignment=TA_CENTER,spaceAfter=8),
 'caption':ParagraphStyle('Caption',fontName='Serif',fontSize=9,leading=11.5,spaceBefore=5,spaceAfter=7),
 'cell':ParagraphStyle('Cell',fontName='Serif',fontSize=8.5,leading=10.5),
 'note':ParagraphStyle('Note',fontName='Serif',fontSize=8.5,leading=11,spaceAfter=8),
 'eq':ParagraphStyle('Eq',fontName='Serif',fontSize=11,leading=17,alignment=TA_CENTER,spaceBefore=7,spaceAfter=9),
}
cites={k:str(i+1) for i,k in enumerate(re.findall(r'\\bibitem\{([^}]+)\}',raw))}
refs={}
for kind in ['table','figure']:
    blocks=re.findall(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}',raw,re.S)
    for i,b in enumerate(blocks):
        lab=re.search(r'\\label\{([^}]+)\}',b)
        if lab:refs[lab.group(1)]=str(i+1)
refs['tab:rapid']='8'; refs['sec:rapid']='A'; refs['sec:provenance']='B'
# Longtable precedes the final ordinary table in document order.
refs['tab:rapid']='8'

def clean(s):
    s=s.replace('\\%','%').replace('\\_','_').replace('\\&','&')
    s=re.sub(r'\\cite\{([^}]+)\}',lambda m:'['+', '.join(cites[x] for x in m.group(1).split(','))+']',s)
    s=re.sub(r'\\ref\{([^}]+)\}',lambda m:refs[m.group(1)],s)
    s=re.sub(r'\\url\{([^}]+)\}',r'\1',s)
    s=re.sub(r'\\text(?:bf|tt|it)\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\emph\{([^{}]*)\}',r'\1',s)
    s=s.replace('~',' ').replace('``','“').replace("''",'”').replace('---','—').replace('--','–')
    subs={r'\uparrow':'↑',r'\downarrow':'↓',r'\gamma':'γ',r'\tau':'τ',r'\beta':'β',r'\mu':'μ',r'\sigma':'σ',r'\pi':'π',r'\phi':'φ',r'\ell':'ℓ',r'\rho':'ρ',r'\widehat R':'R̂',r'\min':'min',r'\mid':'|',r'\log':'log',r'\exp':'exp',r'\sim':'~'}
    for a,b in subs.items():s=s.replace(a,b)
    s=re.sub(r'\$([^$]+)\$',lambda m:m.group(1),s)
    s=s.replace('\\noindent','').replace('\\small','').replace('\\normalsize','')
    s=s.replace('\\centering','').replace('\\par','').replace('\\footnotesize','')
    s=re.sub(r'\\label\{[^}]*\}','',s)
    s=s.replace('\\qquad',' ').replace('\\quad',' ')
    s=re.sub(r'\s+',' ',s).strip()
    return s

plain=[]
def glyph_fallback(s):
    supported=pdfmetrics.getFont('Serif').face.charToGlyph
    return ''.join(ch if ord(ch) in supported else '<font name="Math">'+ch+'</font>' for ch in s)

def p(s,style='body'):
    s=clean(s);plain.append(s)
    # Safe basic subscripts in inline mathematical notation.
    escaped=html.escape(s)
    escaped=re.sub(r'([A-Za-z])_\{([^{}]+)\}',r'\1<sub>\2</sub>',escaped)
    escaped=re.sub(r'([A-Za-z])_([A-Za-z0-9])',r'\1<sub>\2</sub>',escaped)
    escaped=escaped.replace('10^{-4}','10<super>−4</super>')
    return Paragraph(glyph_fallback(escaped),styles[style])

class VectorFigure(Flowable):
    def __init__(self,kind):
        Flowable.__init__(self);self.kind=kind;self.width=W;self.height={'architecture':252,'evidence':180,'split':136,'fusion':180,'confusion':172}[kind]
    def draw(self):
        c=self.canv
        def label(x,y,t,size=8,font='Sans',col=gray):
            c.setFillColor(col);c.setFont(font,size);c.drawCentredString(x,y,t)
        def box(x,y,w,h,lines,dashed=False):
            c.setStrokeColor(colors.HexColor('#956B23') if dashed else blue)
            c.setFillColor(colors.HexColor('#F5F8FA'));c.setLineWidth(.8)
            if dashed:c.setDash(3,2)
            c.roundRect(x,y,w,h,4,fill=1,stroke=1);c.setDash()
            for i,t in enumerate(lines):label(x+w/2,y+h/2+(len(lines)-1)*5.5-i*11,t)
        def arrow(points,dashed=False):
            c.setStrokeColor(gray);c.setLineWidth(.9)
            if dashed:c.setDash(3,2)
            path=c.beginPath();path.moveTo(*points[0])
            for xy in points[1:]:path.lineTo(*xy)
            c.drawPath(path);c.setDash()
            x,y=points[-1];px,py=points[-2]
            import math
            a=math.atan2(y-py,x-px);path=c.beginPath();path.moveTo(x,y)
            path.lineTo(x-5*math.cos(a-.45),y-5*math.sin(a-.45));path.lineTo(x-5*math.cos(a+.45),y-5*math.sin(a+.45));path.close()
            c.setFillColor(gray);c.drawPath(path,fill=1,stroke=0)
        gap=15;bw=(W-32-2*gap)/3;xs=[16+i*(bw+gap) for i in range(3)]
        if self.kind=='architecture':
            ys=[181,107,33];h=56
            items=[['Resume and job description','Skill context'],['Grounded question','Topic and prompt'],['Candidate response','Text, audio, video'],['33-dimensional state','Beliefs, coverage, history','Channel availability'],['Calibrated skill beliefs','Semantic likelihood','Evidence weighting'],['Evidence interface','Semantic score, reliability','Availability'],['Offline IQL checkpoint','Eight action logits'],['Valid-action mask','Next-action selection'],['Live integration gap','Fixed evidence','Three-action cycle']]
            for j,lines in enumerate(items):box(xs[j%3],ys[j//3],bw,h,lines,j==8)
            for a,b in [(0,1),(1,2)]:arrow([(xs[a]+bw,209),(xs[b],209)])
            arrow([(xs[2]+bw/2,181),(xs[2]+bw/2,163)])
            arrow([(xs[2],135),(xs[1]+bw,135)]);arrow([(xs[1],135),(xs[0]+bw,135)])
            arrow([(xs[0]+bw/2,107),(xs[0]+bw/2,89)])
            arrow([(xs[0]+bw,61),(xs[1],61)])
            arrow([(xs[1]+bw/2,33),(xs[1]+bw/2,16),(5,16),(5,247),(xs[1]+bw/2,247),(xs[1]+bw/2,237)],True)
        elif self.kind=='evidence':
            top=[['Component benchmarks','Five public-data tasks'],['Historical policy study','300 simulated episodes','Five non-IQL policies'],['Current offline release','600 synthetic episodes','Frozen v7 configuration']]
            bottom=[['Module-level results','Text and fusion baselines'],['Policy trade-offs','Turns, entropy, reward','Simulated skill accuracy'],['Locked belief comparison','91 episodes','Separate validation OPE']]
            for i in range(3):
                box(xs[i],107,bw,62,top[i]);box(xs[i],15,bw,62,bottom[i]);arrow([(xs[i]+bw/2,107),(xs[i]+bw/2,77)])
        elif self.kind=='split':
            box(16,92,W-32,35,['600 episodes / 9,018 transitions / 33 identity components'])
            for i,lines in enumerate([['Training: 422 episodes','Calibration and IQL fitting'],['Validation: 87 episodes','Selection and diagnostic OPE'],['Locked: 91 episodes','6 components; beliefs only']]):
                box(xs[i],10,bw,56,lines);arrow([(xs[i]+bw/2,92),(xs[i]+bw/2,66)])
        elif self.kind=='fusion':
            x0,y0=55,32;cw=W-90;ch=127
            c.setFont('Sans',8)
            for i in range(6):
                y=y0+ch*i/5;c.setStrokeColor(colors.HexColor('#DDDDDD'));c.line(x0,y,x0+cw,y);label(x0-20,y-3,f'{i/5:.1f}')
            for i,(name,v) in enumerate([('Text',.8015),('Audio',.6098),('Vision',.5808),('Fusion',.7976)]):
                x=x0+cw*(i+.5)/4;c.setFillColor(blue);c.rect(x-20,y0,40,ch*v,fill=1,stroke=0);label(x,y0+ch*v+7,f'{v:.4f}');label(x,17,name)
            c.saveState();c.translate(15,100);c.rotate(90);label(0,0,'Accuracy');c.restoreState()
        elif self.kind=='confusion':
            x0,y0=165,20;cw=65;ch=34;mat=[[29,6,0],[0,31,2],[0,0,23]]
            label(x0+cw*1.5,157,'Predicted class',9,'SansBold')
            for i,name in enumerate(['Beginner','Intermediate','Expert']):label(x0+cw*(i+.5),138,name)
            for r,row in enumerate(mat):
                y=y0+(2-r)*ch;label(x0-58,y+12,['Beginner','Intermediate','Expert'][r])
                for col,n in enumerate(row):
                    shade=.1+.5*n/31;c.setFillColor(colors.Color(1-shade*.75,1-shade*.5,1-shade*.3));c.rect(x0+col*cw,y,cw-2,ch-2,fill=1,stroke=0);label(x0+col*cw+cw/2,y+12,str(n),10)
            c.saveState();c.translate(55,70);c.rotate(90);label(0,0,'True class',9,'SansBold');c.restoreState()

story=[p('ARIA: A Belief-Aware Architecture for Adaptive Mock Interviews with Multimodal Baseline Evaluation','title'),
       p('Raghav Samir Sejpal and Krissh Verma','center'),
       p('School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India','center')]
body=raw.split('\\begin{abstract}',1)[1].split('\\end{document}',1)[0]
abstract,body=body.split('\\end{abstract}',1)
story.extend([p('Abstract','h2'),p(abstract)])
token=re.compile(r'(\\begin\{table\}.*?\\end\{table\}|\\begin\{figure\}.*?\\end\{figure\}|\\begin\{longtable\}.*?\\end\{longtable\}|\\begin\{equation\}.*?\\end\{equation\}|\\section\{[^}]+\}|\\subsection\{[^}]+\}|\\begin\{thebibliography\}.*?\\end\{thebibliography\}|\\appendix)',re.S)
parts=token.split(body);sec=sub=0;app=False;tn=fn=eq=0
equations=[
 'J(π) = E<sub>τ∼π</sub>[ Σ<sub>t=0</sub><super>T−1</super> γ<super>t</super>r<sub>t</sub> ],   γ = 0.99',
 'ℓ<sub>c</sub>(x<sub>t</sub>) = −½[(x<sub>t</sub> − μ<sub>c</sub>)/σ<sub>c</sub>]<super>2</super> − log σ<sub>c</sub>',
 'b<sub>k,t+1</sub>(c) ∝ b<sub>k,t</sub>(c) exp{w<sub>t</sub>ℓ<sub>c</sub>(x<sub>t</sub>)}',
 'L<sub>V</sub> = E<sub>(s,a)∼D</sub>[|τ − 1(u &lt; 0)|u<super>2</super>],   u = min<sub>j</sub> Q̄<sub>j</sub>(s,a) − V(s)',
 'L<sub>π</sub> = −E<sub>(s,a)∼D</sub>[min{e<super>βA(s,a)</super>, 100} log π(a|s)],   β = 3',
 'ρ<sub>i</sub> = min{π(a<sub>i</sub>|s<sub>i</sub>)/μ(a<sub>i</sub>|s<sub>i</sub>), 20}<br/>R̂ = Σ<sub>i</sub>ρ<sub>i</sub>r<sub>i</sub> / Σ<sub>i</sub>ρ<sub>i</sub>,   ESS = (Σ<sub>i</sub>ρ<sub>i</sub>)<super>2</super> / Σ<sub>i</sub>ρ<sub>i</sub><super>2</super>'
]

def table_block(block,long=False):
    global tn
    tn+=1
    cap=re.search(r'\\caption\{(.*?)\}',block,re.S).group(1)
    if long:
        content=block.split('\\endhead',1)[1].split('\\endfoot',1)[1].split('\\end{longtable}',1)[0]
        data=[['Domain','Architecture','Accuracy','Macro-F1','Time (s)']]
        lines=content.splitlines()
    else:
        content=block.split('\\toprule',1)[1].split('\\bottomrule',1)[0]
        data=[];lines=content.splitlines()
    for line in lines:
        line=line.strip().replace('\\midrule','').strip()
        if '&' in line:
            line=line.split('\\\\',1)[0];data.append([clean(x) for x in line.split('&')])
    n=len(data[0]);assert all(len(r)==n for r in data),(tn,data)
    if n==6:widths=[W*.31]+[W*.138]*5
    elif n==5:widths=[W*.50]+[W*.125]*4 if not long else [W*.15,W*.43,W*.14,W*.14,W*.14]
    elif n==4:widths=[W*.18,W*.27,W*.25,W*.30] if tn==1 else [W*.49,W*.17,W*.17,W*.17]
    elif n==3:widths=[W*.25,W*.36,W*.39]
    else:widths=[W/n]*n
    cells=[[Paragraph(glyph_fallback(html.escape(x)),styles['cell']) for x in row] for row in data]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.8,blue),('LINEBELOW',(0,0),(-1,0),.5,blue),('LINEBELOW',(0,-1),(-1,-1),.7,blue),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ECF2F6')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    cap_p=p(f'Table {tn}. '+cap,'caption')
    if long:
        cap_p.keepWithNext=True
        story.extend([cap_p,t,Spacer(1,6)])
    else:story.append(KeepTogether([cap_p,t,Spacer(1,6)]))
    if '\\source{' in block:
        note=block.split('\\source{',1)[1].rsplit('}',2)[0]
        # Remove the enclosing source brace, keeping nested texttt braces.
        depth=1;out=[]
        for ch in block.split('\\source{',1)[1]:
            if ch=='{':depth+=1
            elif ch=='}':
                depth-=1
                if depth==0:break
            out.append(ch)
        story.append(p('Source and scope: '+''.join(out),'note'))

for part in parts:
    if not part.strip():continue
    if part.startswith('\\appendix'):app=True;sec=0;continue
    if part.startswith('\\section'):
        sec+=1;sub=0;n=chr(64+sec) if app else str(sec)
        story.append(CondPageBreak(180 if app else 100))
        story.append(p(n+'. '+re.search(r'\{([^}]+)\}',part).group(1),'h1'))
    elif part.startswith('\\subsection'):
        sub+=1;story.append(p(f'{sec}.{sub}. '+re.search(r'\{([^}]+)\}',part).group(1),'h2'))
    elif part.startswith('\\begin{table}'):table_block(part)
    elif part.startswith('\\begin{longtable}'):table_block(part,True)
    elif part.startswith('\\begin{figure}'):
        fn+=1;cap=re.search(r'\\caption\{(.*?)\}\s*\\label',part,re.S).group(1)
        kind=['architecture','evidence','split','fusion','confusion'][fn-1]
        story.append(KeepTogether([Spacer(1,6),VectorFigure(kind),p(f'Figure {fn}. '+cap,'caption')]))
    elif part.startswith('\\begin{equation}'):
        story.append(Paragraph(glyph_fallback(equations[eq]+f'   ({eq+1})'),styles['eq']));eq+=1
    elif part.startswith('\\begin{thebibliography}'):
        story.append(p('References','h1'))
        for key,entry in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',part,re.S):
            story.append(p('['+cites[key]+'] '+entry,'note'))
    else:
        part=re.sub(r'\\label\{[^}]*\}','',part)
        part=re.sub(r'^%.*$','',part,flags=re.M)
        for para in re.split(r'\n\s*\n',part):
            if clean(para):story.append(p(para))

def footer(c,d):
    c.saveState();c.setStrokeColor(colors.HexColor('#CCCCCC'));c.line(56,40,A4[0]-56,40)
    c.setFont('Sans',7);c.setFillColor(gray);c.drawString(56,28,'ARIA | Revised research draft | 6 October 2026');c.drawRightString(A4[0]-56,28,str(d.page));c.restoreState()
doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=56,rightMargin=56,topMargin=48,bottomMargin=54,title='ARIA: Revised journal draft',author='Raghav Samir Sejpal; Krissh Verma')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'paper/ARIA_revised_reading_text.txt').write_text('\n\n'.join(plain),encoding='utf-8')
print(json.dumps({'pdf':str(pdf),'figures':fn,'tables':tn,'equations':eq,'text_words':len(' '.join(plain).split())}))
