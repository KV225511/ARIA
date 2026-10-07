from pathlib import Path
from PIL import Image, ImageDraw
import sys, math
folder=Path(sys.argv[1]); output=Path(sys.argv[2]); files=sorted(folder.glob('*.png'))
thumbs=[]
for f in files:
    im=Image.open(f).convert('RGB'); im.thumbnail((360,510)); thumbs.append((f.name,im.copy()))
cols=3; rows=math.ceil(len(thumbs)/cols); sheet=Image.new('RGB',(cols*390,rows*550),'#d9dde2'); d=ImageDraw.Draw(sheet)
for idx,(name,im) in enumerate(thumbs):
    x=(idx%cols)*390+15; y=(idx//cols)*550+25; sheet.paste(im,(x,y)); d.text((x,y-20),name,fill='black')
output.parent.mkdir(parents=True,exist_ok=True); sheet.save(output)
