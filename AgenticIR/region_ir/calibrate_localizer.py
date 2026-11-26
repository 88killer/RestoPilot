"""Fit presence/severity temperature and thresholds using validation sources only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import split_for
from .arrays import load_sample_array

TEMPERATURES = (.5, .75, 1., 1.25, 1.5, 2., 3.)


def temperature_probabilities(probabilities, temperatures):
    p = np.clip(probabilities,1e-6,1-1e-6)
    return 1/(1+np.exp(-np.log(p/(1-p))/np.asarray(temperatures)))


def temperature_severity(distributions, temperatures):
    logits=np.log(np.clip(distributions,1e-8,1))/np.asarray(temperatures)[...,None]
    logits-=logits.max(axis=-1,keepdims=True)
    scores=np.exp(logits)
    return scores/scores.sum(axis=-1,keepdims=True)


def fit_calibration(probabilities, labels, distributions=None, severity=None):
    p=np.asarray(probabilities); labels=np.asarray(labels)
    if p.ndim!=2 or p.shape[1]!=3 or labels.shape!=p.shape or not len(p):
        raise ValueError('Expected nonempty Nx3 probabilities and labels')
    if not np.isfinite(p).all() or p.min()<0 or p.max()>1 or not np.isfinite(labels).all() or labels.min()<0 or labels.max()>1:
        raise ValueError('Invalid probabilities/labels')
    temperatures=[]; thresholds=[]
    for kind in range(3):
        target=labels[:,kind]
        losses=[]
        for temperature in TEMPERATURES:
            probability=temperature_probabilities(p[:,kind],temperature)
            losses.append(float(-(target*np.log(probability)+(1-target)*np.log(1-probability)).mean()))
        temperature=TEMPERATURES[int(np.argmin(losses))]
        temperatures.append(temperature)
        probability=temperature_probabilities(p[:,kind],temperature)
        truth=target>=.5
        candidates=np.linspace(.2,.8,13)
        dice=[2*np.sum((probability>=threshold)&truth)/max(1,np.sum(probability>=threshold)+truth.sum()) for threshold in candidates]
        # No positive samples cannot calibrate sensitivity: retain conservative default.
        thresholds.append(float(candidates[int(np.argmax(dice))]) if truth.any() else .5)
    severity_temperatures=[1.,1.,1.]
    severity_support=[0,0,0]
    if distributions is not None and severity is not None:
        distributions=np.asarray(distributions); severity=np.asarray(severity)
        if distributions.shape!=(*p.shape,5) or severity.shape!=p.shape:
            raise ValueError('Invalid severity calibration shapes')
        for kind in range(3):
            valid=severity[:,kind]>=0
            severity_support[kind]=int(valid.sum())
            if not valid.any(): continue
            scores=distributions[valid,kind]; truth=severity[valid,kind].astype(int)
            losses=[]
            for temperature in TEMPERATURES:
                calibrated=temperature_severity(scores,temperature)
                losses.append(float(-np.log(calibrated[np.arange(len(truth)),truth]).mean()))
            severity_temperatures[kind]=TEMPERATURES[int(np.argmin(losses))]
    return dict(temperatures=temperatures,thresholds=thresholds,severity_temperatures=severity_temperatures,
                confidence_threshold=.2,normal_threshold=.3,severity_support=severity_support,
                note='Temperature/segmentation thresholds fitted to validation; confidence/normal thresholds remain explicit defaults')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('data'); p.add_argument('checkpoint'); p.add_argument('output')
    p.add_argument('--device',default='cuda'); p.add_argument('--pixels-per-image',type=int,default=4096)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.pixels_per_image<1: raise ValueError('pixels-per-image must be positive')
    from .core import load_image
    from .localizer import Predictor
    from .severity import severity_labels
    root=Path(args.data)
    records=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    records=[r for r in records if r['split']=='val']
    if not records or any(split_for(r['source_id'])!='val' for r in records):
        raise ValueError('Calibration requires held-out validation sources 0701..0800')
    predictor=Predictor(args.checkpoint,args.device)
    probabilities=[]; labels=[]; distributions=[]; severities=[]
    rng=np.random.default_rng(args.seed)
    for row in records:
        folder=root/row['folder']
        state=predictor.state(load_image(folder/'input.png'))
        target=load_sample_array(folder, 'masks')
        severity=load_sample_array(folder, 'severity', required=False)
        if severity is None: severity=severity_labels(target,row['operations'])
        indices=rng.choice(target.shape[0]*target.shape[1],min(args.pixels_per_image,target.shape[0]*target.shape[1]),replace=False)
        probabilities.append(state['probabilities'].reshape(-1,3)[indices]); labels.append(target.reshape(-1,3)[indices])
        distributions.append(state['severity_distribution'].reshape(-1,3,5)[indices]); severities.append(severity.reshape(-1,3)[indices])
    config=fit_calibration(np.concatenate(probabilities),np.concatenate(labels),
                           np.concatenate(distributions) if predictor.model.has_severity else None,
                           np.concatenate(severities) if predictor.model.has_severity else None)
    config.update(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  validation_sources=sorted({r['source_id'] for r in records}),seed=args.seed,pixels_per_image=args.pixels_per_image)
    with Path(args.output).open('x',encoding='utf-8') as stream: json.dump(config,stream,indent=2)


if __name__=='__main__': main()
