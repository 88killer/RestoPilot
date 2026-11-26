"""Train-only scaling, source-disjoint validation and gain/cost multi-output MLP."""
import argparse
import json
from pathlib import Path
import numpy as np
from .policy import ACTIONS, STATE_SIZE


def training_coverage(x, vx):
    if x.shape[1] != STATE_SIZE + len(ACTIONS):
        raise ValueError('Expected current action-feature schema')
    action_labels = x[:, -len(ACTIONS):]
    validation_actions = vx[:, -len(ACTIONS):]
    for labels in (action_labels, validation_actions):
        if not np.isin(labels, [0, 1]).all() or not (labels.sum(axis=1) == 1).all():
            raise ValueError('Expected one-hot action labels')
    counts = action_labels.sum(axis=0).astype(np.int64)
    supported = np.asarray(ACTIONS)[counts > 0]
    unseen = np.asarray(ACTIONS)[(validation_actions.sum(axis=0) > 0) & (counts == 0)]
    if len(unseen):
        raise ValueError('Validation actions absent from training: ' + ', '.join(unseen))
    return counts, supported


def train(x, y, vx, vy, seed=42, epochs=500, lr=.001):
    """Preserves the original scalar-target API; v2 uses two gain/cost targets."""
    x,y,vx,vy = [np.asarray(a,dtype=np.float64) for a in (x,y,vx,vy)]
    if x.ndim != 2 or vx.ndim != 2 or x.shape[1] != vx.shape[1] or len(x) != len(y) or len(vx) != len(vy) or not len(x) or not len(vx):
        raise ValueError('Invalid supervision shapes')
    if y.ndim not in (1,2) or vy.ndim != y.ndim or y.shape[1:] != vy.shape[1:]:
        raise ValueError('Invalid target shapes')
    if not all(np.isfinite(a).all() for a in (x,y,vx,vy)) or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('Non-finite supervision or invalid optimizer settings')
    scalar = y.ndim == 1
    if scalar: y,vy = y[:,None],vy[:,None]
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0); scale = np.maximum(x.std(axis=0),.01)
    x = (x-mean)/scale; vx = (vx-mean)/scale
    target_scale = np.maximum(y.std(axis=0),1e-4)
    y = y/target_scale; vy = vy/target_scale
    w1 = rng.normal(0,.1,(x.shape[1],32)); b1 = np.zeros(32)
    w2 = rng.normal(0,.1,(32,y.shape[1])); b2 = np.zeros(y.shape[1])
    best = float('inf'); result = None
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0,len(x),128):
            indices = order[start:start+128]
            batch,target = x[indices],y[indices]
            pre = np.einsum('bi,ij->bj',batch,w1)+b1
            hidden = np.maximum(pre,0)
            error = 2*(np.einsum('bi,ij->bj',hidden,w2)+b2-target)/(len(batch)*y.shape[1])
            dh = np.einsum('bi,ji->bj',error,w2)*(pre>0)
            gradients = [np.einsum('bi,bj->ij',batch,dh),dh.sum(axis=0),
                         np.einsum('bi,bj->ij',hidden,error),error.sum(axis=0)]
            norm = np.sqrt(sum(float(np.square(g).sum()) for g in gradients))
            factor = min(1.,5/max(norm,1e-12))
            for parameter,gradient in zip((w1,b1,w2,b2),gradients):
                parameter -= lr*factor*gradient
        pred = np.einsum('bi,ij->bj',np.maximum(np.einsum('bi,ij->bj',vx,w1)+b1,0),w2)+b2
        loss = float(np.mean((pred-vy)**2))
        if not np.isfinite(loss): raise ValueError('Training diverged')
        if loss < best:
            best = loss
            out_w,out_b = w2.copy()*target_scale,b2.copy()*target_scale
            result = dict(mean=mean,scale=scale,w1=w1.copy(),b1=b1.copy(),
                          w2=out_w[:,0] if scalar else out_w,b2=out_b[0] if scalar else out_b)
            raw_mse = float(np.mean(((pred-vy)*target_scale)**2))
    return result,raw_mse


def main():
    p = argparse.ArgumentParser(); p.add_argument('train'); p.add_argument('val'); p.add_argument('output')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--lr',type=float,default=.001)
    args = p.parse_args()
    with np.load(args.train,allow_pickle=False) as a,np.load(args.val,allow_pickle=False) as b:
        if str(a['split']) != 'train' or str(b['split']) != 'val': raise ValueError('Expected train/val labels')
        if set(a['source_ids']) & set(b['source_ids']): raise ValueError('Source leakage')
        if 'targets' not in a or 'targets' not in b:
            raise ValueError('Regenerate v2 full-path gain/cost labels before training')
        for key in ('feature_version','normalization_hash','cost_weight','no_history','no_cost','perception_signature','perception_json'):
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):
                raise ValueError(f'Train/val settings differ: {key}')
        counts, supported = training_coverage(a['x'], b['x'])
        model,loss = train(a['x'],a['targets'],b['x'],b['targets'],args.seed,args.epochs,args.lr)
        metadata = {key:a[key].copy() for key in ('feature_version','normalization_hash','cost_weight','no_history','no_cost','perception_signature','perception_json')}
        metadata.update(supported_actions=supported, action_counts=counts)
        source_counts = dict(train=len(set(a['source_ids'])),val=len(set(b['source_ids'])))
    output = Path(args.output); output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'ranker.npz',**model,**metadata)
    (output/'training.json').write_text(json.dumps(dict(seed=args.seed,validation_mse=loss,epochs=args.epochs,
                                                       targets=['normalized_gain','normalized_cost'],source_counts=source_counts,
                                                       config={key:value.item() if value.ndim == 0 else value.tolist() for key,value in metadata.items()}),indent=2))


if __name__ == '__main__': main()
