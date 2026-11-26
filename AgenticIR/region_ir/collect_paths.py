"""Offline training/validation path search; clean reference stays in this module."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import subprocess
import numpy as np
from .arrays import load_sample_array, sample_array_path
from scipy import ndimage
from .core import CommandTool, build_regions, candidate_paths, execute_region, load_image
from .provenance import perception_payload,payload_sha256,canonical_json
from .policy import prediction_state, state_features, candidate_tools, FEATURE_VERSION
from .path_supervision import fit_normalization, path_labels, validate_normalization


def reward_components(before, after, clean, initial, mask, elapsed, true_normal=None):
    inside = mask >= .99
    if not inside.any(): inside = mask > 0
    if not inside.any(): raise ValueError('Empty objective region')
    normal = mask == 0 if true_normal is None else true_normal
    support = mask > 0
    boundary = ndimage.binary_dilation(support,iterations=5)^ndimage.binary_erosion(support,iterations=5)
    gain = float(((before-clean)**2)[inside].mean()-((after-clean)**2)[inside].mean())
    # Incremental changes telescope across the full path, rather than charging prior damage again.
    normal_damage = float((np.abs(after-initial)-np.abs(before-initial))[normal].mean()) if normal.any() else 0.
    edge_damage = float((np.abs(after-clean)-np.abs(before-clean))[boundary].mean()) if boundary.any() else 0.
    return [gain,float(elapsed),normal_damage,edge_damage]


def utility(before,after,clean,mask,elapsed,cost_weight,normal_weight,boundary_weight):
    """Legacy raw-unit diagnostic utility; training v2 uses fitted normalization."""
    gain,cost,normal,edge = reward_components(before,after,clean,before,mask,elapsed)
    return gain-cost_weight*cost-normal_weight*normal-boundary_weight*edge


def collect_tree(image, clean, region, predictor, tools, source_id, split, max_steps=2,
                 true_normal=None, threshold=.5, no_history=False, context=32):
    if split not in ('train','val'):
        raise ValueError('Path supervision is forbidden on test sources')
    initial_state = prediction_state(predictor,image)
    selected = candidate_tools(image,region,initial_state,tools,threshold)
    cache = {(): image}
    states = {(): initial_state}
    nodes, edges = [], []
    for path in candidate_paths(selected,max_steps):
        if not path: continue
        prefix = path[:-1]
        if prefix not in cache: continue
        before = cache[prefix]
        start = perf_counter()
        error = None
        try:
            after = execute_region(before,region,tools[path[-1]],context=context)
            seconds = perf_counter()-start
            components = reward_components(before,after,clean,image,region.mask,seconds,true_normal)
            cache[path] = after
            states[path] = prediction_state(predictor,after)
        except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
            seconds = perf_counter()-start
            components = [0., seconds, 0., 0.]
            error = f'{type(exc).__name__}: {exc}'
        edges.append(dict(parent=list(prefix),action=path[-1],components=components,failed=error is not None,error=error))
    for prefix, current in cache.items():
        state = state_features(current,region,states[prefix],prefix,12-len(prefix),max_steps-len(prefix),no_history)
        nodes.append(dict(path=list(prefix),state=state.tolist()))
    return dict(source_id=source_id,split=split,region_id=region.id,max_steps=max_steps,
                selected_tools=selected,nodes=nodes,edges=edges,feature_version=FEATURE_VERSION)


def select_region_candidates(candidates, limit, covered):
    """Select from prediction-only descriptors; clean images/rewards are not inputs."""
    remaining = [item for item in candidates if item['tools']]
    chosen = []
    while remaining and len(chosen) < limit:
        index = max(range(len(remaining)), key=lambda i: (
            len(set(remaining[i]['tools']) - covered), len(remaining[i]['tools']),
            remaining[i]['area']))
        item = remaining.pop(index)
        chosen.append(item)
        covered.update(item['tools'])
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data'); p.add_argument('output'); p.add_argument('--localizer',required=True)
    p.add_argument('--calibration')
    p.add_argument('--tools',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--split',choices=['train','val'],default='train')
    p.add_argument('--limit',type=int,default=100,help='Maximum original images, not degraded variants')
    p.add_argument('--region-selection', choices=['priority','coverage'], default='priority', help='Development sampling using predicted tool coverage only')
    p.add_argument('--regions-per-source',type=int,default=2)
    p.add_argument('--max-steps',type=int,choices=[1,2,3],default=2)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--context',type=int,choices=[16,32,64],default=32)
    p.add_argument('--normalization',help='Frozen training normalization JSON; required for validation')
    p.add_argument('--cost-weight',type=float,default=.1)
    p.add_argument('--normal-weight',type=float,default=.1); p.add_argument('--boundary-weight',type=float,default=.1)
    p.add_argument('--no-history',action='store_true'); p.add_argument('--no-cost',action='store_true')
    args = p.parse_args()
    if args.limit < 1 or args.regions_per_source < 1: p.error('Limits must be positive')
    if args.split == 'val' and not args.normalization: p.error('Validation requires frozen --normalization from training')
    root = Path(args.data)
    records = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    rows = [r for r in records if r['split'] == args.split]
    source_ids = list(dict.fromkeys(r['source_id'] for r in rows))[:args.limit]
    rows = [r for r in rows if r['source_id'] in source_ids]
    # Spend the two-region/source development budget on order-sensitive scenarios first.
    priority = {'compound': 0, 'three_regions': 1, 'two_regions': 2}
    rows.sort(key=lambda r: (r['source_id'], priority.get(r['scenario'], 3), r['scenario']))
    if not rows: raise ValueError('No samples in requested split')
    # Validate the canonical source partition even if the caller manually edited the manifest.
    from .data import split_for
    if any(split_for(r['source_id']) != args.split for r in rows):
        raise ValueError('Manifest split disagrees with original source ids')
    config = json.loads(Path(args.tools).read_text(encoding='utf-8'))
    tools = {k:CommandTool(**v) for k,v in config.items()}
    from .localizer import Predictor
    predictor = Predictor(args.localizer,args.device,calibration=args.calibration)
    perception = perception_payload(predictor.provenance,args.threshold)
    perception_signature = payload_sha256(perception)
    output = Path(args.output); output.mkdir(parents=True,exist_ok=False)
    chosen_regions = {}
    if args.region_selection == 'coverage':
        covered = set()
        for source_id in source_ids:
            candidates = []
            for row in rows:
                if row['source_id'] != source_id:
                    continue
                image = load_image(root / row['folder'] / 'input.png')
                state = prediction_state(predictor, image)
                for region in build_regions(state['probabilities'], threshold=args.threshold):
                    candidates.append(dict(key=(row['source_id'], row['scenario']), region_id=region.id,
                        tools=candidate_tools(image,region,state,tools,args.threshold),
                        area=int((region.mask > 0).sum())))
            for item in select_region_candidates(candidates, args.regions_per_source, covered):
                chosen_regions.setdefault(item['key'], []).append(item['region_id'])
        rows = [row for row in rows if (row['source_id'],row['scenario']) in chosen_regions]
    trees = []; region_count = dict.fromkeys(source_ids,0)
    with (output/'trees.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:
            if region_count[row['source_id']] >= args.regions_per_source: continue
            folder = root/row['folder']
            image,clean = load_image(folder/'input.png'),load_image(folder/'clean.png')
            if image.shape != clean.shape: raise ValueError('Clean/input dimensions differ')
            truth = load_sample_array(folder, 'masks')
            if truth.shape != image.shape: raise ValueError('Ground-truth mask dimensions differ')
            normal = truth.max(axis=2) == 0
            initial = prediction_state(predictor,image)
            regions = build_regions(initial['probabilities'],threshold=args.threshold)
            if args.region_selection == 'coverage':
                ids = chosen_regions[(row['source_id'],row['scenario'])]
                regions = [region for region in regions if region.id in ids]
            for region in regions:
                if region_count[row['source_id']] >= args.regions_per_source: break
                tree = collect_tree(image,clean,region,predictor,tools,row['source_id'],args.split,
                                    args.max_steps,normal,args.threshold,args.no_history,args.context)
                tree.update(scenario=row['scenario'],generation_order=row['operations'])
                trees.append(tree); region_count[row['source_id']] += 1
                stream.write(json.dumps(tree,allow_nan=False)+'\n'); stream.flush()
    if not trees: raise ValueError('No predicted regions; validate the localizer first')
    if args.normalization:
        normalization = json.loads(Path(args.normalization).read_text())
        validate_normalization(normalization)
        if args.split == 'val' and set(source_ids) & set(normalization['source_ids']):
            raise ValueError('Normalization source leakage')
    else:
        normalization = fit_normalization(trees,args.cost_weight,args.normal_weight,args.boundary_weight)
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2))
    xs,targets,ids = [],[],[]
    for tree in trees:
        x,y,_ = path_labels(tree,normalization,no_history=args.no_history,no_cost=args.no_cost)
        xs.extend(x); targets.extend(y); ids.extend([tree['source_id']]*len(x))
    fingerprint = hashlib.sha256(json.dumps(normalization,sort_keys=True).encode()).hexdigest()
    target = np.asarray(targets,dtype=np.float32)
    weight = 0. if args.no_cost else normalization['cost_weight']
    np.savez(output/'labels.npz',x=np.asarray(xs),targets=target,y=target[:,0]-weight*target[:,1],
             source_ids=np.asarray(ids),split=np.asarray(args.split),feature_version=FEATURE_VERSION,
             perception_signature=perception_signature,perception_json=canonical_json(perception),
             normalization_hash=fingerprint,cost_weight=weight,no_history=args.no_history,no_cost=args.no_cost)
    (output/'config.json').write_text(json.dumps(dict(**vars(args),source_count=len(source_ids),regions=len(trees),
                                                    predictor_calls=predictor.calls,normalization_hash=fingerprint,regions_by_source=region_count,perception=perception,perception_signature=perception_signature),indent=2))


if __name__ == '__main__': main()
