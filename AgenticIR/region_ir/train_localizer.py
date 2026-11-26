import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .core import load_image
from .arrays import load_sample_array
from .data import split_for
from .localizer import Localizer, segmentation_loss, severity_loss, deployment_logits, initialize_from_checkpoint
from .severity import severity_labels, BIN_EDGES, LEVEL_NAMES


class Samples(Dataset):
    def __init__(self, root, records, training, clean_probability=0.):
        if not np.isfinite(clean_probability) or not 0 <= clean_probability <= 1:
            raise ValueError('Clean probability must be in [0,1]')
        if not training and clean_probability:
            raise ValueError('Clean augmentation is training-only')
        self.root, self.records, self.training = root, records, training
        self.clean_probability, self.clean_count = clean_probability, 0
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        directory = self.root/record['folder']
        image = load_image(directory/'input.png')
        labels = load_sample_array(directory, 'masks')
        if labels.shape != image.shape or not np.isfinite(labels).all() or labels.min()<0 or labels.max()>1:
            raise ValueError('Invalid label/image shapes or range')
        severity = load_sample_array(directory, 'severity', required=False)
        if severity is None: severity = severity_labels(labels,record['operations'])
        if severity.shape!=labels.shape or not np.isin(severity,[-1,0,1,2,3,4]).all():
            raise ValueError('Invalid severity labels')
        if self.training and self.clean_probability > 0 and random.random() < self.clean_probability:
            image = load_image(directory/'clean.png')
            if image.shape != labels.shape: raise ValueError('Clean/input dimensions differ')
            labels = np.zeros_like(labels)
            severity = np.full_like(severity,-1)
            self.clean_count += 1
        if self.training and random.random()<.5:
            image,labels,severity = image[:,::-1],labels[:,::-1],severity[:,::-1]
        return (torch.from_numpy(image.copy()).permute(2,0,1),
                torch.from_numpy(labels.copy()).permute(2,0,1),
                torch.from_numpy(severity.copy()).permute(2,0,1).long())


def validate_initial_checkpoint(state, width, normalization, train_sources, input_features='rgb'):
    """A development warm start must not introduce training/validation source leakage."""
    if state.get('width') != width or state.get('normalization','none') != normalization:
        raise ValueError('Initial checkpoint architecture differs from requested model')
    source_features=state.get('input_features','rgb')
    if source_features != input_features and (source_features,input_features) != ('rgb','rgb_hp'):
        raise ValueError('Initial checkpoint input features differ')
    if not state.get('train_sources'):
        raise ValueError('Initial checkpoint requires training-source provenance')
    canonical = lambda values: {str(int(value)).zfill(4) for value in values}
    outside = canonical(state['train_sources']) - canonical(train_sources)
    if outside:
        raise ValueError('Initial checkpoint was trained on sources outside current training split: ' + ', '.join(sorted(outside)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data'); p.add_argument('output')
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--input-features', choices=['rgb','rgb_hp'], default='rgb', help='Optional local residual inputs; RGB warm starts are zero-expanded')
    p.add_argument('--clean-probability', type=float, default=0., help='Training-only clean negative examples; validation remains degraded')
    p.add_argument('--initial-checkpoint', help='Development warm start; optimizer restarts and parent provenance is recorded')
    p.add_argument('--validate-every', type=int, default=1, help='Validate first, last and every N epochs')
    p.add_argument('--validation-mode', choices=['tiled','full'], default='tiled', help='Use deployment tiling for checkpoint selection; full reproduces legacy validation')
    p.add_argument('--tile-batch-size', type=int, default=4)
    p.add_argument('--normalization', choices=['none','group'], default='none', help='Architecture variant recorded in the checkpoint')
    p.add_argument('--width',type=int,default=32); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda'); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--severity-weight',type=float,default=.25)
    p.add_argument('--positive-weight', type=float, nargs=3, help='Optional BCE weights for noise/blur/JPEG; derive from training pixels only')
    args=p.parse_args()
    if args.validate_every<1 or args.tile_batch_size<1 or args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.severity_weight<=0:
        raise ValueError('epochs, batch size, lr and severity weight must be positive')
    if not np.isfinite(args.clean_probability) or not 0 <= args.clean_probability <= 1:
        raise ValueError('Clean probability must be in [0,1]')
    if args.positive_weight is not None and not all(np.isfinite(v) and v > 0 for v in args.positive_weight):
        raise ValueError('Presence weights must be finite and positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root=Path(args.data)
    manifest=root/'samples.jsonl'
    records=[json.loads(s) for s in manifest.read_text().splitlines()]
    for record in records:
        if record['split']!=split_for(record['source_id']): raise ValueError('Source split does not match DIV2K protocol')
    train=[r for r in records if r['split']=='train']; val=[r for r in records if r['split']=='val']
    if not train or not val: raise ValueError('Both train and val source images required')
    if {r['source_id'] for r in train}&{r['source_id'] for r in val}: raise ValueError('Source leakage')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    shuffle_generator=torch.Generator().manual_seed(args.seed)
    loaders=[DataLoader(Samples(root,rows,i==0,args.clean_probability if i==0 else 0.),batch_size=args.batch_size if i==0 else 1,shuffle=i==0,num_workers=0,generator=shuffle_generator if i==0 else None)
             for i,rows in enumerate([train,val])]
    model=Localizer(args.width,severity=True,normalization=args.normalization,input_features=args.input_features).to(args.device)
    initial_sha=None; initial_conversion=None
    if args.initial_checkpoint:
        initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
        validate_initial_checkpoint(initial,args.width,args.normalization,{r['source_id'] for r in train},args.input_features)
        initial_conversion=initialize_from_checkpoint(model,initial)
        initial_sha=hashlib.sha256(Path(args.initial_checkpoint).read_bytes()).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf')
    provenance=dict(train_sources=sorted({r['source_id'] for r in train}),val_sources=sorted({r['source_id'] for r in val}),
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),arguments=vars(args),
                    initial_checkpoint_sha256=initial_sha,initial_conversion=initial_conversion,validation_mode=args.validation_mode,shuffle_generator_seed=args.seed,
                    severity_names=LEVEL_NAMES,severity_edges=BIN_EDGES)
    (output/'config.json').write_text(json.dumps(provenance,indent=2))
    for epoch in range(args.epochs):
        metrics={}
        for phase,loader in zip(['train','val'],loaders):
            if phase == 'val' and not (epoch == 0 or epoch == args.epochs-1 or (epoch+1) % args.validate_every == 0):
                metrics['val']=None
                continue
            loader.dataset.clean_count = 0
            model.train(phase=='train'); total=0.; count=0
            intersection=np.zeros(3); union=np.zeros(3); predicted=np.zeros(3); actual=np.zeros(3)
            confusion=np.zeros((3,5,5),dtype=np.int64); false_normal=normal_count=0
            for image,labels,severity in loader:
                image,labels,severity=image.to(args.device),labels.to(args.device),severity.to(args.device)
                with torch.set_grad_enabled(phase=='train'):
                    if phase == 'val' and args.validation_mode == 'tiled':
                        if len(image) != 1: raise ValueError('Tiled validation expects batch size one')
                        source = image[0].permute(1,2,0).detach().cpu().numpy()
                        logits,levels=deployment_logits(model,source,args.device,tile_batch_size=args.tile_batch_size)
                    else:
                        logits,levels=model.forward_state(image)
                    loss=segmentation_loss(logits,labels,args.positive_weight)+args.severity_weight*severity_loss(levels,severity)
                    if not torch.isfinite(loss): raise ValueError('Non-finite localization loss')
                    if phase=='train':
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                total+=float(loss.detach())*len(image); count+=len(image)
                pred=logits.detach().sigmoid()>=.5; target=labels>=.5
                intersection+=(pred&target).sum((0,2,3)).cpu().numpy()
                union+=(pred|target).sum((0,2,3)).cpu().numpy()
                predicted+=pred.sum((0,2,3)).cpu().numpy(); actual+=target.sum((0,2,3)).cpu().numpy()
                true_normal=labels.max(dim=1).values==0
                false_normal+=int((pred.any(dim=1)&true_normal).sum()); normal_count+=int(true_normal.sum())
                level_pred=levels.detach().argmax(dim=2)
                for kind in range(3):
                    truth=severity[:,kind]; valid=truth>=0
                    encoded=truth[valid]*5+level_pred[:,kind][valid]
                    confusion[kind]+=torch.bincount(encoded,minlength=25).reshape(5,5).cpu().numpy()
            metrics[phase]=dict(loss=total/count,clean_samples=loader.dataset.clean_count,
                iou=[float(i/u) if u else None for i,u in zip(intersection,union)],
                dice=[float(2*i/(p+a)) if p+a else None for i,p,a in zip(intersection,predicted,actual)],
                normal_false_positive_rate=false_normal/normal_count if normal_count else None,
                severity_confusion=confusion.tolist(),
                severity_accuracy=[float(c.trace()/c.sum()) if c.sum() else None for c in confusion])
        if metrics['val'] is not None and metrics['val']['loss']<best:
            best=metrics['val']['loss']
            torch.save(dict(model=model.state_dict(),width=args.width,seed=args.seed,epoch=epoch,severity=True,normalization=args.normalization,input_features=args.input_features,
                            **provenance),output/'best.pt')
        with (output/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=epoch,**metrics))+'\n')
        print(epoch,metrics,flush=True)


if __name__=='__main__': main()
