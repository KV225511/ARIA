from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
W,H=1800,980
im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
font=ImageFont.truetype('arial.ttf',26); bold=ImageFont.truetype('arialbd.ttf',34); small=ImageFont.truetype('arial.ttf',21)
def box(x,y,w,h,text,color,dash=False):
    if dash:
        for xx in range(x,x+w,24): d.line((xx,y,min(xx+14,x+w),y),fill=color,width=4); d.line((xx,y+h,min(xx+14,x+w),y+h),fill=color,width=4)
        for yy in range(y,y+h,24): d.line((x,yy,x,min(yy+14,y+h)),fill=color,width=4); d.line((x+w,yy,x+w,min(yy+14,y+h)),fill=color,width=4)
    else: d.rounded_rectangle((x,y,x+w,y+h),radius=15,outline=color,width=4)
    lines=text.split('\n'); total=len(lines)*32
    for i,line in enumerate(lines):
        bb=d.textbbox((0,0),line,font=font); d.text((x+(w-(bb[2]-bb[0]))/2,y+(h-total)/2+i*32),line,font=font,fill='#17202a')
def arrow(x1,y1,x2,y2,color='#52606d'):
    d.line((x1,y1,x2,y2),fill=color,width=5); d.polygon([(x2,y2),(x2-18,y2-11),(x2-18,y2+11)],fill=color)
d.text((45,30),'Evaluated offline path',font=bold,fill='#123b5d')
box(45,105,285,125,'600 synthetic episodes\n9,018 transitions','#1f77b4'); box(430,105,270,125,'belief-v3\ncalibration','#1f77b4'); box(800,105,315,125,'aria-state-v4\n33 features, 8 actions','#1f77b4'); box(1215,105,285,125,'IQL checkpoint\nversioned + hashed','#1f77b4')
arrow(330,168,430,168); arrow(700,168,800,168); arrow(1115,168,1215,168)
box(430,315,270,125,'Locked belief test\nn=91; not policy eval','#2ca02c'); box(1215,315,285,125,'Validation WIS\ndiagnostic only','#2ca02c')
d.line((565,230,565,315),fill='#52606d',width=5); d.polygon([(565,315),(554,296),(576,296)],fill='#52606d'); d.line((1358,230,1358,315),fill='#52606d',width=5); d.polygon([(1358,315),(1347,296),(1369,296)],fill='#52606d')
d.text((45,535),'Current live application',font=bold,fill='#7a3e00')
box(45,610,410,125,'WebSocket + transcript UI\nfixed evidence scores','#d97706'); box(555,610,410,125,'three-action cycle\nno checkpoint loading','#d97706'); arrow(455,673,555,673,'#d97706')
d.text((1100,535),'Evidence still required',font=bold,fill='#7b1e1e')
box(1100,610,550,100,'fresh learned-policy rollouts','#b42318',True); box(1100,760,550,100,'real-person validity, fairness, access','#b42318',True)
d.text((380,900),'Solid boxes: inspected implementation or artifact     Dashed boxes: unsupported claim requirements',font=small,fill='#52606d')
out=Path('paper/figures'); out.mkdir(parents=True,exist_ok=True); im.save(out/'aria_evidence_boundary.png')

# Paired v6-v7 metric comparison. Values are read from the immutable locked-test report.
metrics=[('Accuracy',.7912087912,.9120879121,True),('Macro-F1',.8010035717,.9167658730,True),('Bal. acc.',.8178932179,.9226551227,True),('Min. rec.',.5142857143,.8285714286,True),('ECE',.0808055325,.0870087026,False),('Brier',.2450570351,.1395080408,False),('Ord. MAE',.2087912088,.0879120879,False)]
cw,ch=1800,980
chart=Image.new('RGB',(cw,ch),'white'); cd=ImageDraw.Draw(chart)
cd.text((55,30),'Paired locked-test belief metrics: v6 vs. v7',font=bold,fill='#17365D')
cd.text((55,78),'Same 91 synthetic episodes; this is a belief-estimator comparison, not a policy evaluation.',font=small,fill='#52606D')
left,top,right,bottom=250,150,1700,850
for j in range(6):
    y=bottom-j*(bottom-top)/5
    cd.line((left,y,right,y),fill='#D9E2E8',width=2)
    cd.text((145,y-12),f'{j/5:.1f}',font=small,fill='#52606D')
group=(right-left)/len(metrics); bw=55
for i,(label,v6,v7,higher) in enumerate(metrics):
    x=left+i*group+35
    for k,(val,col,name) in enumerate([(v6,'#9AA6B2','v6'),(v7,'#1F77B4','v7')]):
        h=val*(bottom-top); xx=x+k*(bw+12)
        cd.rounded_rectangle((xx,bottom-h,xx+bw,bottom),radius=5,fill=col)
        cd.text((xx-2,bottom-h-30),f'{val:.3f}',font=small,fill='#17202A')
    bb=cd.textbbox((0,0),label,font=small); cd.text((x+(2*bw+12-(bb[2]-bb[0]))/2,bottom+18),label,font=small,fill='#17202A')
cd.rectangle((1320,55,1360,80),fill='#9AA6B2'); cd.text((1370,54),'v6',font=small,fill='#17202A')
cd.rectangle((1450,55,1490,80),fill='#1F77B4'); cd.text((1500,54),'v7',font=small,fill='#17202A')
cd.text((475,920),'Higher is better for the first four metrics; lower is better for ECE, Brier, and ordinal MAE.',font=small,fill='#52606D')
chart.save(out/'belief_metric_comparison.png')

# Locked v7 confusion matrix: rows=true, columns=predicted.
cm=[[29,6,0],[0,31,2],[0,0,23]]; labels=['Low','Medium','High']
heat=Image.new('RGB',(1200,980),'white'); hd=ImageDraw.Draw(heat)
hd.text((55,30),'Locked v7 synthetic belief confusion matrix (n=91)',font=bold,fill='#17365D')
hd.text((55,78),'Rows are true labels; columns are predicted labels.',font=small,fill='#52606D')
sx,sy,cell=320,230,190
for i,row in enumerate(cm):
    hd.text((110,sy+i*cell+75),labels[i],font=font,fill='#17202A')
    for j,val in enumerate(row):
        intensity=int(245-175*(val/31)); color=(intensity,intensity+10,min(255,intensity+35))
        hd.rectangle((sx+j*cell,sy+i*cell,sx+(j+1)*cell,sy+(i+1)*cell),fill=color,outline='#FFFFFF',width=5)
        f=bold if val else font; bb=hd.textbbox((0,0),str(val),font=f)
        hd.text((sx+j*cell+(cell-(bb[2]-bb[0]))/2,sy+i*cell+(cell-(bb[3]-bb[1]))/2),str(val),font=f,fill='#102A43')
for j,label in enumerate(labels):
    bb=hd.textbbox((0,0),label,font=font); hd.text((sx+j*cell+(cell-(bb[2]-bb[0]))/2,sy-48),label,font=font,fill='#17202A')
hd.text((500,135),'Predicted label',font=font,fill='#17365D')
hd.text((55,760),'All eight errors are adjacent-class errors: six low→medium and two medium→high.',font=font,fill='#52606D')
hd.text((55,815),'Artifact: aria-locked-test-evaluation-v1; producer: aria-belief-calibration-v7.',font=small,fill='#52606D')
heat.save(out/'belief_confusion_matrix.png')
