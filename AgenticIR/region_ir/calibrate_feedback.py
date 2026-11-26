"""Select feedback thresholds from validation-only candidate comparisons."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
from .feedback import FeedbackConfig,decide


def calibrate(rows):
    if not rows or any(row.get('split')!='val' for row in rows):
        raise ValueError('Feedback calibration requires validation-only rows')
    if any('beneficial' not in row for row in rows): raise ValueError('Missing reference-based beneficial label')
    positives=sum(bool(r['beneficial']) for r in rows)
    if positives==0 or positives==len(rows): raise ValueError('Need beneficial and harmful candidates')
    best=None
    for quality_gain,tie,residual,color,normal in itertools.product([0.,.5,1.],[0.,.1,.5],[0.,.005,.02],[.03,.1],[.001,.01]):
        config=FeedbackConfig(quality_gain,tie,residual,0.,color,normal)
        predictions=[decide(row['values'],config) for row in rows]
        tpr=sum(p and r['beneficial'] for p,r in zip(predictions,rows))/positives
        tnr=sum(not p and not r['beneficial'] for p,r in zip(predictions,rows))/(len(rows)-positives)
        score=(tpr+tnr)/2
        if best is None or score>best[0]: best=(score,config)
    return dict(thresholds=asdict(best[1]),validation_balanced_accuracy=best[0],
                validation_sources=sorted({r['source_id'] for r in rows}),samples=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('comparisons');p.add_argument('output');args=p.parse_args()
    rows=[json.loads(s) for s in Path(args.comparisons).read_text().splitlines()]
    result=calibrate(rows)
    with Path(args.output).open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)


if __name__=='__main__': main()
