from pathlib import Path
from PIL import Image
import json,numpy as np
root=Path(__file__).resolve().parent; data=root/'生成数据'; out=root/'实例清单.jsonl'; rows=[]
for file in sorted(data.glob('C*/*/manifest.json')):
 meta=json.loads(file.read_text()); case=file.parent; clean=(case/meta['clean']).resolve(); inp=case/'input.png'; masks=case/'masks.npz'
 with np.load(masks,allow_pickle=False) as a:
  assert a.files==['masks'] and a['masks'].shape==(512,512,3)
 assert Image.open(clean).size==Image.open(inp).size==(512,512)
 rows.append({**meta,'clean_path':str(clean.relative_to(data)),'input_path':str(inp.relative_to(data)),'masks_path':str(masks.relative_to(data))})
assert rows and len({(x['source_id'],x['scenario']) for x in rows})==len(rows)
out.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows));print('Validated',len(rows),'instances:',out)
