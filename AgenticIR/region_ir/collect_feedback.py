"""Derive validation acceptance labels from saved candidates, without rerunning tools."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import load_image


def collect(data_manifest,results):
    root=Path(data_manifest).parent
    samples={(r['source_id'],r['scenario']):r for r in
             [json.loads(s) for s in Path(data_manifest).read_text().splitlines()]}
    comparisons=[]
    for line in Path(results).read_text().splitlines():
        row=json.loads(line);sample=samples[(row['source_id'],row['scenario'])]
        if sample['split']!='val':raise ValueError('Calibration may only use validation results')
        clean=load_image(root/sample['folder']/'clean.png')
        case=Path(row['run_dir']);trace=case/'trace.jsonl'
        if not trace.exists():continue
        for event in [json.loads(s) for s in trace.read_text().splitlines()]:
            values=event.get('feedback')
            if not values or event.get('error'):continue
            prefix=f'r{event["region_id"]}_s{event["step"]}'
            before=load_image(case/f'{prefix}_before.png');after=load_image(case/f'{prefix}_candidate.png')
            # Paired full-image MSE judges actual benefit, including outside-region damage.
            gain=float(((before-clean)**2).mean()-((after-clean)**2).mean())
            comparisons.append(dict(split='val',source_id=sample['source_id'],scenario=sample['scenario'],
                                    beneficial=gain>0,reference_mse_gain=gain,
                                    values={k:values[k] for k in ['quality_gain','residual_gain','color_shift','normal_change']}))
    return comparisons


def main():
    p=argparse.ArgumentParser();p.add_argument('data_manifest');p.add_argument('results');p.add_argument('output');args=p.parse_args()
    rows=collect(args.data_manifest,args.results)
    if not rows:raise ValueError('No joint-feedback candidate records to calibrate')
    with Path(args.output).open('x',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')


if __name__=='__main__':main()
