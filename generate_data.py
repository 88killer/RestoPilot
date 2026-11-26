from pathlib import Path
import json, zipfile, hashlib, io, sys, numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent/'AgenticIR'))
from region_ir.synthesize import synthesize
base=Path(__file__).resolve().parent
sources=sorted(x for x in (base/'原图').rglob('*') if x.is_file() and x.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp'))
if not sources: raise SystemExit('请先把原图放进 原图 文件夹。')
if (base/'生成数据').exists(): raise SystemExit('生成数据 文件夹已存在；请先移走或重命名，避免覆盖。')
out=base/'生成数据';out.mkdir(exist_ok=True)
scenarios=['compound_pair_a','compound_pair_b','compound_triple','multi_region','single','clean']
kinds=['noise','blur','jpeg']; pairs=[('noise','blur'),('noise','jpeg'),('blur','jpeg')]
def operation(k,i):
 if k=='noise':return {'kind':k,'sigma':[15,25,35][i%3]}
 if k=='blur':return {'kind':k,'length':[7,15,23][i%3],'angle':[0,45,90,135][(i//3)%4]}
 return {'kind':k,'quality':[20,30,40][i%3]}
def mask(i,half=None):
 m=np.zeros((512,512),dtype=np.float32);rng=np.random.default_rng(19000+i); h,w=[(160,160),(240,240),(320,320)][i%3];
 if half is not None:w=min(w,220); x=int(rng.integers(0,257-w))+half*256
 else:x=int(rng.integers(0,513-w))
 y=int(rng.integers(0,513-h));m[y:y+h,x:x+w]=1;return m
archives={}
try:
 for i,src in enumerate(sources):
  raw=src.read_bytes(); source_sha=hashlib.sha256(raw).hexdigest(); sample_id=f'C{i+1:04d}'
  im=Image.open(io.BytesIO(raw)).convert('RGB')
  if min(im.size)<512: raise ValueError(f'图片尺寸小于 512: {src}')
  x=(im.width-512)//2;y=(im.height-512)//2;crop=im.crop((x,y,x+512,y+512));clean=np.asarray(crop,dtype=np.float32)/255
  folder=out/sample_id;folder.mkdir(exist_ok=True);crop.save(folder/'clean.png')
  for j,scenario in enumerate(scenarios):
   seed=2026112400+i*10+j; a=mask(i); b=mask(i+2,0); c=mask(i+5,1)
   if j<2:
    ks=pairs[(i+j)%3];ks=ks if (i+j)%2==0 else ks[::-1];ops=[operation(k,i+j) for k in ks]; ms=[a,a]
   elif j==2:
    ks=kinds[i%3:]+kinds[:i%3];ops=[operation(k,i) for k in ks];ms=[a]*3
   elif j==3:
    ks=pairs[i%3];ops=[operation(k,i) for k in ks];ms=[b,c]
   elif j==4:ops=[operation(kinds[i%3],i)];ms=[a]
   else:ops=[];ms=[]
   image,labels=synthesize(clean,ms,ops,seed);case=folder/scenario;case.mkdir(exist_ok=True)
   Image.fromarray(np.rint(image*255).astype('uint8')).save(case/'input.png');np.savez_compressed(case/'masks.npz',masks=labels)
   meta={'source_file':str(src.relative_to(base/'原图')),'source_id':sample_id,'source_sha256':source_sha,'scenario':scenario,'seed':seed,'crop':[x,y,512,512],'operations':ops,'mask_layout':'two disjoint rectangles' if j==3 else 'shared rectangle' if ops else 'none','clean':'../clean.png','input':'input.png','masks':'masks.npz','jpeg_subsampling':0}
   (case/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
  print(f'{i+1}/{len(sources)} {sample_id}',flush=True)
finally:
 for z in archives.values():z.close()
