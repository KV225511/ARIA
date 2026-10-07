"""Create a native editable Excalidraw scene and deterministic vector/PNG previews.

No image-generation model or raster image element is used. All scene objects are
rectangles, text and arrows. Run with Python + Pillow.
"""
from pathlib import Path
import json
import math
import html
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
W, H = 2440, 1870
INK, MUTED = '#172b42', '#52677b'
COLORS = {'live':('#087e8b','#e8f6f5'), 'offline':('#7056a6','#f1edf8'),
          'module':('#b77a25','#fff5e5'), 'neutral':('#64748b','#f2f5f8')}
FONT = 'C:/Windows/Fonts/arial.ttf'
elements, nodes, svg = [], {}, []
im=Image.new('RGB',(W,H),'#ffffff');draw=ImageDraw.Draw(im)

def base(kind,x,y,w,h,**kw):
    i=len(elements)+1
    e=dict(id=f'aria-architecture-{i:03d}',type=kind,x=x,y=y,width=w,height=h,
           angle=0,strokeColor=INK,backgroundColor='transparent',fillStyle='solid',
           strokeWidth=1.5,strokeStyle='solid',roughness=0,opacity=100,groupIds=[],
           frameId=None,roundness=None,seed=7919*i,version=1,versionNonce=17*i,
           isDeleted=False,boundElements=None,updated=1791331200000,created=None,
           link=None,locked=False)
    e.update(kw);elements.append(e);return e

def rect(x,y,w,h,fill,stroke='#dce5ec',radius=16,**kw):
    e=base('rectangle',x,y,w,h,backgroundColor=fill,strokeColor=stroke,
           roundness={'type':3} if radius else None,**kw)
    draw.rounded_rectangle((x,y,x+w,y+h),radius=radius,fill=fill,outline=stroke,width=2)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    return e

def text(x,y,value,size=20,color=INK,group=None,max_width=None):
    font=ImageFont.truetype(FONT,size)
    lines=value.split('\n');width=max(draw.textlength(s,font=font) for s in lines)
    height=len(lines)*size*1.25
    if max_width is not None: assert width<=max_width,(value,width,max_width)
    assert x+width<=W-22 and y+height<=H-12,(value,x+width,y+height)
    e=base('text',x,y,width,height,strokeColor=color,text=value,originalText=value,
           fontSize=size,fontFamily=2,lineHeight=1.25,textAlign='left',
           verticalAlign='top',containerId=None,autoResize=True)
    if group:e['groupIds']=[group]
    for i,line in enumerate(lines):
        ly=y+i*size*1.25
        draw.text((x,ly),line,font=font,fill=color)
        svg.append(f'<text x="{x}" y="{ly+size*.91}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="{color}">{html.escape(line)}</text>')
    return e

def panel(x,y,w,h,title,subtitle=''):
    rect(x,y,w,h,'#f8fafc','#dce5ec',22)
    text(x+26,y+18,title,25)
    if subtitle:text(x+26,y+52,subtitle,17,MUTED)

def box(key,x,y,w,h,title,body,kind='module',badge=None):
    stroke,fill=COLORS[kind];group='group-'+key
    r=rect(x,y,w,h,fill,stroke);r['groupIds']=[group]
    r['customData']={'module':key,'implementationStatus':badge or kind}
    nodes[key]=r
    text(x+18,y+13,badge or {'live':'LIVE','offline':'OFFLINE','module':'STANDALONE','neutral':'INTERFACE'}[kind],12,stroke,group,max_width=w-36)
    text(x+18,y+36,title,23,INK,group,max_width=w-36)
    text(x+18,y+73,body,18,MUTED,group,max_width=w-36)
    return r

def arrow(points,kind='neutral',dashed=False,start=None,end=None):
    color=COLORS[kind][0]
    x,y=points[0]
    e=base('arrow',x,y,max(a for a,b in points)-min(a for a,b in points),
           max(b for a,b in points)-min(b for a,b in points),
           strokeColor=color,strokeWidth=2,strokeStyle='dashed' if dashed else 'solid',
           points=[[a-x,b-y] for a,b in points],startBinding=None,endBinding=None,
           startArrowhead=None,endArrowhead='arrow',elbowed=False)
    for key,which in [(start,'startBinding'),(end,'endBinding')]:
        if key:
            node=nodes[key]
            e[which]={'elementId':node['id'],'focus':0,'gap':8}
            if node['boundElements'] is None:node['boundElements']=[]
            node['boundElements'].append({'id':e['id'],'type':'arrow'})
    for a,b in zip(points,points[1:]):
        length=math.dist(a,b)
        if dashed:
            for d in range(0,int(length),15):
                t=d/length;u=min(d+8,length)/length
                draw.line((a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,
                           a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u),fill=color,width=2)
        else:draw.line([a,b],fill=color,width=2)
    px,py=points[-2];ex,ey=points[-1];ang=math.atan2(ey-py,ex-px)
    wings=[(ex-10*math.cos(ang+d),ey-10*math.sin(ang+d)) for d in (-.5,.5)]
    draw.line([wings[0],(ex,ey),wings[1]],fill=color,width=2)
    dash=' stroke-dasharray="8 7"' if dashed else ''
    svg.append(f'<polyline points="'+ ' '.join(f'{a},{b}' for a,b in points)+f'" fill="none" stroke="{color}" stroke-width="2"{dash}/><polyline points="{wings[0][0]},{wings[0][1]} {ex},{ey} {wings[1][0]},{wings[1][1]}" fill="none" stroke="{color}" stroke-width="2"/>')

# Header and self-contained reading key.
text(48,26,'ARIA  /  COMPLETE PROJECT ARCHITECTURE',37)
text(50,77,'Autonomous Reinforcement-Based Interview Agent  |  15 modules  |  repository implementation map',21,MUTED)
for x,kind,label in [(50,'live','LIVE: used by the web app'),(465,'offline','OFFLINE: learning / evaluation'),(940,'module','STANDALONE: module exists')]:
    rect(x,118,15,15,COLORS[kind][0],COLORS[kind][0],3)
    text(x+25,114,label,17,MUTED)
arrow([(1460,126),(1535,126)]);text(1550,114,'Implemented path',17,MUTED)
arrow([(1840,126),(1915,126)],dashed=True);text(1930,114,'Integration to be connected',17,MUTED)

# 1: The actual browser/API route, not the intended learned-policy route.
panel(40,170,2360,275,'01   LIVE INTERVIEW','Read left to right; the upper return line closes the conversation loop.')
box('ui',70,275,350,135,'Candidate workspace','React + Vite: browser interface\nResume + job description; answers','live')
box('api',470,275,350,135,'FastAPI orchestrator','Session state + WebSocket stream\nffmpeg + M1: recorded audio to text','live')
box('m5',870,275,325,135,'M5  Skill ontology','Skill graph + prerequisite links\nGrounded in role and resume','live')
box('m6live',1245,275,325,135,'M6  Live belief update','Per-skill probability estimates\nCurrent evidence scores: 0.9','live')
box('m8',1620,275,350,135,'M8  Grounded LLM','Large language model via Ollama\nTurns skill + action into a question','live')
box('delivery',2020,275,350,135,'Question delivery','Transcript + browser speech\nTTS = text-to-speech','live')
for a,b in [('ui','api'),('api','m5'),('m5','m6live'),('m6live','m8'),('m8','delivery')]:
    n,m=nodes[a],nodes[b];arrow([(n['x']+n['width']+7,342),(m['x']-8,342)],'live',start=a,end=b)
arrow([(2195,267),(2195,247),(245,247),(245,267)],'live')
text(908,226,'next question / next answer',16,COLORS['live'][0])

# 2: Complete module map, with implemented components and proposed orchestration.
panel(40,480,2360,715,'02   EVIDENCE TO INTERVIEW ACTION','Complete module design; amber connections are proposed orchestration, while purple paths run offline.')
text(76,566,'PERCEPTION',16,MUTED);text(500,566,'SIGNAL FUSION',16,MUTED)
text(907,566,'AUXILIARY CONTEXT',16,MUTED);text(1335,566,'BELIEF + POLICY STATE',16,MUTED);text(1930,566,'ACTION SELECTION',16,MUTED)
box('m1',70,625,350,135,'M1  STT + semantics','Speech-to-text; answer scoring\nSTT live; semantic grading separate','module','MIXED: LIVE / STANDALONE')
box('m2',70,800,350,135,'M2  Vision engine','Face, gaze + expression cues\nLive camera is preview-only')
box('m3',70,975,350,135,'M3  Prosody','Pitch, timing + voice energy\nMeasures how an answer is spoken')
box('m4',495,800,330,150,'M4  Multimodal fusion','Combines text, audio + vision\n72-D = 72 normalized features')
box('m10',900,625,350,135,'M10  Cognitive load','Rule-based stress / knowledge cues\nContext flags, not a diagnosis')
box('m11',900,800,350,135,'M11  Anti-gaming','Gaze, latency + template checks\nFlags possible assistance')
box('m12',900,975,350,135,'M12  Incongruence','Mismatch between answer quality\nand vocal / behavioral confidence')
box('m6',1330,685,420,165,'M6  Bayesian skill belief','Updates probabilities with evidence\nBeginner / intermediate / expert\nConfidence weights evidence','offline','OFFLINE: CALIBRATED PATH')
box('state',1330,945,420,150,'33-D policy state','Beliefs + coverage + recent history\nD = number of input features\nDistinct from the 72-D fusion vector','offline')
box('m7',1920,685,450,165,'M7  IQL interview policy','Implicit Q-Learning: learns from logs\nAction mask: blocks invalid choices\nChooses 1 of 8 interview actions','offline')
box('actions',1920,945,450,165,'Eight action choices','Raise / lower difficulty; follow up\nSwitch topic; probe foundations\nBehavioral; situational; conclude','offline','ACTION SPACE')

# Horizontal/vertical routes run in the gutters, never across card text.
arrow([(645,418),(645,461),(25,461),(25,692),(62,692)],'live',end='m1')
for a,yy in [('m1',730),('m2',866),('m3',1040)]:
    arrow([(428,yy),(460,yy),(460,875),(487,875)],'module',True,end='m4')
for b,yy in [('m10',692),('m11',867),('m12',1042)]:
    arrow([(833,875),(860,875),(860,yy),(892,yy)],'module',True,end=b)
arrow([(428,672),(450,672),(450,605),(1540,605),(1540,677)],'module',True,end='m6')
text(612,609,'semantic score + reliability',16,COLORS['module'][0])
for a,yy in [('m10',692),('m11',867),('m12',1042)]:
    arrow([(1258,yy),(1288,yy),(1288,1030),(1322,1030)],'module',True,end='state')
arrow([(1540,858),(1540,937)],'offline',start='m6',end='state')
text(1555,885,'summarize',16,COLORS['offline'][0])
arrow([(1758,1020),(1820,1020),(1820,770),(1912,770)],'offline',start='state',end='m7')
arrow([(2145,858),(2145,937)],'offline',start='m7',end='actions')
arrow([(2120,677),(2120,535),(1795,535),(1795,418)],'offline',True,end='m8')
text(1900,507,'selected strategy',16,COLORS['offline'][0])
text(1334,1133,'Competency likelihood uses semantic evidence.\nBehavioral cues are not validated measures of ability.',18,MUTED)

# 3: All remaining modules, separated from the currently wired browser path.
panel(40,1230,2360,230,'03   REPORTING, GOVERNANCE + OUTPUT','Services exist; the complete post-interview flow is not connected to the live API.')
box('m13',70,1335,530,115,'M13  Fairness auditor','Checks policy actions against speech patterns','module')
box('m14',650,1335,530,115,'M14  Evaluation report','Skill summary + language-model narrative','module')
box('m15',1230,1335,530,115,'M15  Feedback store','SQLite database: trajectories + later outcomes','module')
box('m9',1810,1335,560,115,'M9  Speech / avatar interface','Local speech prototype; avatar is a placeholder','module')
arrow([(2240,1118),(2240,1209),(1495,1209),(1495,1327)],'module',True,end='m15')
arrow([(1322,735),(1305,735),(1305,1218),(915,1218),(915,1327)],'module',True,end='m14')
text(979,1196,'skill profile',16,MUTED)
arrow([(1222,1392),(1205,1392),(1205,1318),(335,1318),(335,1327)],'module',True,start='m15',end='m13')
arrow([(608,1392),(642,1392)],'module',True,start='m13',end='m14')
arrow([(2195,418),(2390,418),(2390,1318),(2090,1318),(2090,1327)],'module',True,end='m9')
text(1770,1184,'trajectory = sequence of turns',16,MUTED)

# 4: Offline lifecycle; test belief results do not serve as a learned-policy test.
panel(40,1500,2360,270,'04   OFFLINE LEARNING + EVALUATION','Frozen synthetic data; controlled replay and comparison. No automatic retraining from feedback.')
box('data',70,1595,405,140,'Evidence generation','Distinct candidate + evaluator LLMs\nSave responses, scores and actions','offline')
box('split',525,1595,405,140,'Identity-safe split','Linked resumes / roles stay together\nTraining / validation / locked test','offline')
box('replay',980,1595,405,140,'Calibration + replay','Fit belief parameters; rebuild states,\nrewards and masks from saved data','offline')
box('train',1435,1595,405,140,'Policy learning','IQL + behavior-cloning baseline\nCQL + DT-inspired comparators','offline')
box('eval',1890,1595,480,140,'Separate evaluation tracks','Locked test: stored belief accuracy\nDevelopment: matched policy rollouts','offline')
for a,b in [('data','split'),('split','replay'),('replay','train'),('train','eval')]:
    n,m=nodes[a],nodes[b];arrow([(n['x']+n['width']+7,1665),(m['x']-8,1665)],'offline',start=a,end=b)
arrow([(1638,1587),(1638,1480),(2420,1480),(2420,767),(2378,767)],'offline',end='m7')
text(1940,1458,'trained checkpoint = saved policy weights',16,COLORS['offline'][0])
text(50,1800,'CQL = Conservative Q-Learning  |  DT = Decision Transformer-inspired  |  LLM = large language model',18,MUTED)
text(50,1830,'CURRENT LIVE SHORTCUT: fixed evidence + three-action cycle. Learned policy and full multimodal loop still need live integration.',18,COLORS['live'][0])

# Validate scene structure and references. Every module is present in the canvas.
ids={e['id'] for e in elements};assert len(ids)==len(elements)
assert {f'm{i}' for i in range(1,16)}<=set(nodes)
for e in elements:
    assert e['width']>=0 and e['height']>=0
    assert e['type'] in {'rectangle','text','arrow'}
    for key in ['startBinding','endBinding']:
        if e.get(key):assert e[key]['elementId'] in ids
scene={'type':'excalidraw','version':2,'source':'https://excalidraw.com',
       'elements':elements,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'ARIA_complete_architecture.excalidraw').write_text(json.dumps(scene,indent=2),encoding='utf-8')
clipboard={'type':'excalidraw/clipboard','elements':elements,'files':{}}
(OUT/'ARIA_complete_architecture.clipboard.json').write_text(json.dumps(clipboard,indent=2),encoding='utf-8')
(OUT/'ARIA_complete_architecture.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>'+''.join(svg)+'</svg>',encoding='utf-8')
im.save(OUT/'ARIA_complete_architecture.png')
print(json.dumps({'elements':len(elements),'modules':15,'size':[W,H],'native_image_elements':0,'output':str(OUT)}))
