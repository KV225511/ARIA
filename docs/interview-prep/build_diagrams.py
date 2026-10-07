"""Generate editable Excalidraw scenes and matching readable PNG previews."""
import json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
SCALE = 2
INK = '#233044'
COLORS = {'live':'#dbeafe','offline':'#e7ddfa','planned':'#fff0c2','neutral':'#edf1f5','good':'#d9f3e5'}
FONT = Path('C:/Windows/Fonts/segoeui.ttf')
BOLD = Path('C:/Windows/Fonts/segoeuib.ttf')

class Board:
    def __init__(self, name, title, subtitle, height=670):
        self.name, self.width, self.height = name, 1120, height
        self.elements = []
        self.im = Image.new('RGB', (self.width*SCALE, height*SCALE), 'white')
        self.draw = ImageDraw.Draw(self.im)
        self.text(30, 20, title, 29, bold=True)
        self.text(30, 63, subtitle, 18)

    def base(self, typ, x, y, w, h, **extra):
        i = len(self.elements)+1
        el = dict(id=f'{self.name}-{i}',type=typ,x=x,y=y,width=w,height=h,
            angle=0,strokeColor=INK,backgroundColor='transparent',fillStyle='solid',
            strokeWidth=2,strokeStyle='solid',roughness=1,opacity=100,
            groupIds=[],frameId=None,roundness=None,seed=i*7919,version=1,
            versionNonce=i*17,isDeleted=False,boundElements=None,updated=1,link=None,locked=False)
        el.update(extra)
        self.elements.append(el)
        return el

    def text(self, x, y, text, size=20, bold=False):
        font=ImageFont.truetype(str(BOLD if bold else FONT), size*SCALE)
        lines=text.split('\n')
        w=max(self.draw.textlength(line,font=font)/SCALE for line in lines)
        h=len(lines)*size*1.35
        assert x+w <= self.width-15, (self.name, text, x+w)
        assert y+h <= self.height-8, (self.name, text, y+h)
        self.base('text',x,y,w,h,text=text,fontSize=size,fontFamily=2,
                  textAlign='left',verticalAlign='top',containerId=None,
                  originalText=text,autoResize=True,lineHeight=1.35)
        for k,line in enumerate(lines):
            self.draw.text((x*SCALE,(y+k*size*1.35)*SCALE),line,font=font,fill=INK)

    def box(self,x,y,w,h,title,body='',kind='live'):
        start = len(self.elements)
        h = max(h, 46 + len(body.split('\n'))*17*1.35 + 12) if body else h
        self.base('rectangle',x,y,w,h,backgroundColor=COLORS[kind],roundness={'type':3})
        self.draw.rounded_rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE),
            radius=12*SCALE,fill=COLORS[kind],outline=INK,width=2*SCALE)
        self.text(x+16,y+12,title,21,True)
        if body:self.text(x+16,y+46,body,17)
        for element in self.elements[start:]:
            element['groupIds'] = [f'{self.name}-group-{start}']

    def arrow(self,points,label=None):
        x,y=points[0]
        self.base('arrow',x,y,max(p[0] for p in points)-min(p[0] for p in points),
            max(p[1] for p in points)-min(p[1] for p in points),
            points=[[a-x,b-y] for a,b in points],startBinding=None,endBinding=None,
            startArrowhead=None,endArrowhead='arrow',elbowed=False)
        self.draw.line([(a*SCALE,b*SCALE) for a,b in points],fill=INK,width=2*SCALE)
        px,py=points[-2];ex,ey=points[-1];a=math.atan2(ey-py,ex-px)
        wings=[(ex-12*math.cos(a+d),ey-12*math.sin(a+d)) for d in (-.45,.45)]
        self.draw.line([(wings[0][0]*SCALE,wings[0][1]*SCALE),(ex*SCALE,ey*SCALE),
            (wings[1][0]*SCALE,wings[1][1]*SCALE)],fill=INK,width=2*SCALE)

    def save(self):
        scene={'type':'excalidraw','version':2,'source':'ARIA interview study pack',
            'elements':self.elements,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}}
        (OUT/f'{self.name}.excalidraw').write_text(json.dumps(scene,indent=2),encoding='utf-8')
        self.im.save(OUT/f'{self.name}.png')

b=Board('01-system-map','ARIA: three parts of one research prototype',
    'Blue = live application    Purple = offline ML    Yellow = modules awaiting live integration')
b.box(30,115,310,160,'Candidate interface','React + Vite\nPDF setup, text / audio answers\nTranscript, camera preview\nBrowser speech synthesis')
b.box(400,115,330,160,'Live orchestrator','FastAPI + WebSocket\nGrounded ontology + questions\nPer-session history and beliefs\nFixed interview action cycle')
b.box(790,115,300,160,'Local services','PyMuPDF: PDF text\nffmpeg: audio conversion\nfaster-whisper: transcription\nOllama: question wording')
b.arrow([(340,185),(400,185)]);b.arrow([(730,185),(790,185)])
b.box(30,330,510,165,'Offline evidence and learning','Synthetic interviews -> identity-safe splits\nCalibration -> deterministic replay -> IQL\nValidated inference runtime + fresh-rollout code\nCheckpoint is not loaded by app.py',kind='offline')
b.box(580,330,510,165,'Wider multimodal architecture','Vision, prosody and 72-feature fusion\nCognitive / integrity / fairness modules\nEvaluation reports and SQLite feedback\nThese are not orchestrated in each live turn',kind='planned')
b.text(35,545,'Core separation',23,True)
b.text(35,585,'Ontology: which skill?   Belief: what do we know?   Policy: which strategy?\nLLM: how should the next question be worded?',22)
b.save()

b=Board('02-live-turn','Trace one real interview turn',
    'Read app.py alongside this diagram. This shows current behavior, including its placeholders.',740)
b.box(30,110,320,130,'1. Start session','POST two PDFs\nExtract text; adapt skill graph\nReturn session_id')
b.box(410,110,320,130,'2. Connect socket','/ws/interview/{session_id}\nChoose first target skill\nGenerate first question')
b.box(790,110,300,130,'3. Candidate answers','Typed text, or recorded audio\nWebM -> WAV -> transcript\nAppend question + answer')
b.arrow([(350,175),(410,175)]);b.arrow([(730,175),(790,175)])
b.arrow([(940,240),(940,300)])
b.box(790,300,300,145,'4. Update belief','Current semantic score = 0.9\nCurrent cognitive load = low\nBehavior argument = 0.9\nActual answer is not graded',kind='planned')
b.box(410,300,320,145,'5. Select strategy','Fixed repeating cycle:\nprobe_foundation\nincrease_difficulty\nswitch_topic',kind='planned')
b.box(30,300,320,145,'6. Select target skill','Follow graph prerequisites\nor advanced skills as needed\nRank by selection priority,\nevidence count, then skill ID')
b.arrow([(790,365),(730,365)]);b.arrow([(410,365),(350,365)])
b.arrow([(190,445),(190,505)])
b.box(30,505,510,130,'7. Generate and validate wording','Ollama receives target + JD grounding + history\nCheck grounding / duplicates; retry up to 3 times\nFail if all attempts are invalid')
b.box(600,505,490,130,'8. Deliver question','Send one aria_question WebSocket message\nReact updates transcript and speaks question\nThen wait for the next candidate answer')
b.arrow([(540,570),(600,570)])
b.text(30,678,'WebSocket transport is live; this path sends a complete question after validation.',21)
b.save()

b=Board('03-belief-and-policy','Belief estimates knowledge; policy chooses the next move',
    'Conceptual offline loop. The live app substitutes constant evidence and a fixed strategy cycle.',720)
b.box(30,115,300,145,'Observe evidence','Answer receives semantic score\nReliability comes from confidence\nRepeated evidence is discounted\nBehavior does not set likelihood')
b.box(410,115,310,145,'Update skill belief','P(Beginner, Mid, Expert)\nPosterior proportional to:\nprior x likelihood ^ weight\nCap effective evidence per skill')
b.box(800,115,290,145,'Aggregate + abstain','Pool visited-skill beliefs\nRequire confidence, evidence\nand sufficient skill coverage\nOtherwise: insufficient evidence')
b.arrow([(330,185),(410,185)]);b.arrow([(720,185),(800,185)])
b.box(30,330,300,130,'Ask the next question','Select a skill in the graph\nLLM turns strategy into wording\nCandidate provides new evidence',kind='good')
b.box(410,330,310,130,'Choose legal action','IQL policy: 8 action logits\nMask prohibited conclusion\nNormalize legal probabilities\nChoose an action',kind='offline')
b.box(800,330,290,130,'Build policy state','33 fixed features\nBeliefs + entropy + coverage\nHistory + signal availability\nAction mask stored separately',kind='offline')
b.arrow([(945,260),(945,330)]);b.arrow([(800,395),(720,395)]);b.arrow([(410,395),(330,395)])
b.arrow([(30,395),(15,395),(15,185),(30,185)])
b.box(30,515,1060,145,'Example: knowledge and uncertainty are different','Illustrative belief [0.33, 0.33, 0.34] = uncertain; [0.05, 0.20, 0.75] = stronger Expert belief.\nA good SQL answer should update SQL evidence, not prove competence in every skill.\nLow uncertainty alone is not enough: require coverage and evaluate correctness.',kind='neutral')
b.save()

b=Board('04-offline-pipeline','How ARIA learns without experimenting on live candidates',
    'Keep evidence creation, parameter fitting, checkpoint selection and final evaluation separate.',730)
b.box(30,115,310,145,'1. Generate raw evidence','Resume / JD pairing plan\nDistinct candidate and evaluator\nStore scores, actions, identities\nand provenance',kind='offline')
b.box(405,115,310,145,'2. Split by identity','Group connected resume / JD IDs\nKeep each group in one split\nTrain / validation / locked test\nPrevents identity leakage',kind='offline')
b.box(780,115,310,145,'3. Calibrate + replay','Fit under versioned protocol\nRebuild beliefs, 33-D states,\nrewards and masks from evidence\nKeep source evidence immutable',kind='offline')
b.arrow([(340,185),(405,185)]);b.arrow([(715,185),(780,185)])
b.arrow([(935,260),(935,330)])
b.box(780,330,310,140,'4. Train IQL','Two Q networks + value network\nAdvantage-weighted policy fit\nSelect using development data\nSave checkpoint + contract hashes',kind='offline')
b.box(405,330,310,140,'5. Stored-belief test','Saved v7 report: 91 interviews\nAccuracy / micro-F1 = 91.2%\nSynthetic classification result\nNot learned-policy performance',kind='good')
b.box(30,330,310,140,'6. Policy evaluation','Offline diagnostic exists\nFresh-rollout executor exists\nNew actions need new responses\nThen compare with baselines',kind='planned')
b.arrow([(780,400),(715,400)]);b.arrow([(405,400),(340,400)])
b.box(30,530,1060,130,'The interview distinction to remember','Replaying old answers can evaluate a belief model on stored evidence.\nIt cannot show how candidates would answer different questions chosen by a new policy.\nA saved metric is evidence for its stated experiment, not for the whole deployed system.',kind='neutral')
b.save()
print('Created four Excalidraw scenes and four PNG previews.')

# One editable canvas containing all four boards in reading order.
combined = []
for name, dx, dy in (
    ('01-system-map', 0, 0),
    ('02-live-turn', 1240, 0),
    ('03-belief-and-policy', 0, 860),
    ('04-offline-pipeline', 1240, 860),
):
    scene = json.loads((OUT/f'{name}.excalidraw').read_text(encoding='utf-8'))
    for element in scene['elements']:
        element['x'] += dx
        element['y'] += dy
        combined.append(element)
assert len({element['id'] for element in combined}) == len(combined)
assert all(element['width'] >= 0 and element['height'] >= 0 for element in combined)
document = {
    'type': 'excalidraw', 'version': 2, 'source': 'ARIA interview study pack',
    'elements': combined,
    'appState': {'viewBackgroundColor': '#ffffff', 'gridSize': None}, 'files': {},
}
target = OUT/'ARIA-all-diagrams.excalidraw'
target.write_text(json.dumps(document, indent=2), encoding='utf-8')
assert len(json.loads(target.read_text(encoding='utf-8'))['elements']) == len(combined)
print(f'Combined all four boards: {len(combined)} editable elements.')
