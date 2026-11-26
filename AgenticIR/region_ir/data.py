"""Frozen source-level six-scenario protocol, with balanced test parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import PIL
from PIL import features
import scipy
from scipy import ndimage
from .core import load_image, save_image
from .arrays import save_sample_array
from .synthesize import synthesize
from .severity import severity_labels, KINDS, BIN_EDGES, LEVEL_NAMES

SCENARIOS = ('global', 'two_regions', 'three_regions', 'normal_majority', 'small', 'compound')
PROTOCOL_VERSION = 'regional-six-v2'


def split_for(source_id):
    number = int(source_id)
    if not 1 <= number <= 900:
        raise ValueError('DIV2K source id must be in 0001..0900')
    return 'train' if number <= 700 else 'val' if number <= 800 else 'test'


def _order(ordinal, scenario):
    if scenario == 'three_regions':
        return [(ordinal + i) % 3 for i in range(3)]
    if scenario in ('two_regions', 'compound'):
        return [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)][ordinal % 6]
    return [ordinal % 3]


def sample_spec(source_id, scenario, seed=42):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    split = split_for(source_id)
    number = int(source_id)
    ordinal = number - {'train': 1, 'val': 701, 'test': 801}[split]
    order = _order(ordinal, scenario)
    operations = []
    for kind in order:
        # Canonical occurrence count makes subset generation identical to full generation.
        occurrence = sum(kind in _order(n, scenario) for n in range(ordinal))
        if kind == 0:
            values = [15, 25, 50] if split == 'test' else [5, 15, 25, 35, 50, 15, 25, 50]
            operation = dict(kind='noise', sigma=values[occurrence % len(values)])
        elif kind == 1:
            values = [7, 15, 23] if split == 'test' else [3, 7, 13, 19, 23, 7, 15, 23]
            operation = dict(kind='blur', length=values[occurrence % len(values)],
                             angle=[0, 45, 90, 135][(occurrence // len(values)) % 4])
        else:
            values = [10, 20, 30, 40] if split == 'test' else [95, 75, 55, 35, 10, 20, 30, 40]
            operation = dict(kind='jpeg', quality=values[occurrence % len(values)])
        operations.append(operation)
    # Blocked factors avoid directly encoding degradation class in mask shape.
    shape = ['rectangle', 'ellipse', 'irregular'][(ordinal // 3) % 3]
    boundary = ['hard', 'soft'][(ordinal // 9) % 2]
    area = 1. if scenario == 'global' else .25 if scenario == 'normal_majority' else (
        [.05, .10, .15][(ordinal // 3) % 3] if scenario == 'small' else [.25, .5, .75][(ordinal // 3) % 3])
    return dict(source_id=f'{number:04d}', split=split, scenario=scenario,
                seed=seed + number * 100 + SCENARIOS.index(scenario), operations=operations,
                mask=dict(shape=shape, boundary=boundary, target_area=area,
                          gaussian_sigma=1.5 if boundary == 'soft' else 0., truncate_below=1e-4))


def _shape_mask(height, width, spec, rng):
    yy, xx = np.mgrid[:height, :width]
    cy, cx = rng.uniform(.4, .6, 2)
    dy, dx = np.abs(yy / height - cy), np.abs(xx / width - cx)
    if spec['shape'] == 'rectangle':
        distance = np.maximum(dy, dx)
    elif spec['shape'] == 'ellipse':
        distance = dy**2 + dx**2
    else:
        distance = dy**2 + dx**2 + .01*np.sin(yy/height*25)*np.cos(xx/width*23)
    count = max(1, min(height * width, round(spec['target_area'] * height * width)))
    mask = np.zeros(height * width, dtype=np.float32)
    mask[np.argsort(distance.ravel(), kind='stable')[:count]] = 1
    mask = mask.reshape(height, width)
    if spec['boundary'] == 'soft':
        mask = ndimage.gaussian_filter(mask, spec['gaussian_sigma'])
        mask[mask < spec['truncate_below']] = 0
    return mask


def make_sample(clean, seed, scenario, spec=None):
    """Compatibility API; production callers pass a frozen sample_spec."""
    if spec is None:
        spec = sample_spec(801 + seed % 100, scenario, seed)
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    masks = []
    count = len(spec['operations'])
    shared = None
    for index in range(count):
        if scenario == 'global':
            mask = np.ones((h, w), dtype=np.float32)
        elif scenario in ('two_regions', 'three_regions'):
            x0, x1 = index*w//count, (index+1)*w//count
            mask = np.zeros((h, w), dtype=np.float32)
            mask[:, x0:x1] = _shape_mask(h, x1-x0, spec['mask'], rng)
        elif shared is not None:
            mask = shared.copy()
        else:
            mask = _shape_mask(h, w, spec['mask'], rng)
            shared = mask
        masks.append(mask)
    image, labels = synthesize(clean, masks, spec['operations'], seed)
    return image, labels, spec['operations']


def write_source(source, output, seed, protocol_sha, array_storage='npy', severity_storage='int8'):
    """Write the six frozen scenarios for one original and return manifest rows."""
    source, output = Path(source), Path(output)
    clean = load_image(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    split = split_for(source.stem)
    size = 256 if split == 'train' else 512
    if min(clean.shape[:2]) < size:
        raise ValueError(f'{source}: smaller than required crop {size}')
    records=[]
    for scenario in SCENARIOS:
        spec = sample_spec(source.stem, scenario, seed)
        rng = np.random.default_rng(spec['seed'])
        h, w = clean.shape[:2]
        y = int(rng.integers(h-size+1)) if split == 'train' else (h-size)//2
        x = int(rng.integers(w-size+1)) if split == 'train' else (w-size)//2
        crop = clean[y:y+size,x:x+size]
        degraded, labels, operations = make_sample(crop, spec['seed'], scenario, spec)
        folder = output/split/f'{int(source.stem):04d}_{scenario}'
        folder.mkdir(parents=True)
        save_image(folder/'input.png', degraded)
        save_image(folder/'clean.png', crop)
        save_sample_array(folder, 'masks', labels, array_storage)
        save_sample_array(folder, 'severity', severity_labels(labels, operations).astype(severity_storage), array_storage)
        records.append(dict(**spec, crop=[x,y,size,size], folder=folder.relative_to(output).as_posix(),
                            source_sha256=source_hash, protocol_sha256=protocol_sha,
                            jpeg_subsampling=0, jpeg_block_origin=[0,0],
                            affected_fraction=float((labels.max(axis=2)>0).mean()),
                            interior_fraction=float((labels.max(axis=2)>=.99).mean())))
    return records


def protocol_document(source_ids, seed=42, array_storage='npy', severity_storage='int8'):
    ids=[int(s) for s in source_ids]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate source ids')
    for sid in ids: split_for(sid)
    if array_storage not in ('npy','npz') or severity_storage not in ('int8','int64'):
        raise ValueError('Invalid lossless array storage')
    plans=[sample_spec(sid, scenario, seed) for sid in ids for scenario in SCENARIOS]
    return dict(version=PROTOCOL_VERSION, seed=seed, complete_test=set(range(801,901)).issubset(ids),
                array_storage=array_storage, test_inputs=sum(p['split']=='test' for p in plans), plans=plans,
                severity=dict(storage_dtype=severity_storage,names=LEVEL_NAMES, normalized_bin_edges=BIN_EDGES,
                              note='train/val add intermediate strengths to cover five proxy levels; test retains proposal strengths'),
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              pillow=PIL.__version__, jpeg_codec=features.version('jpg')),
                jpeg=dict(subsampling=0, block_origin=[0,0], input='full current crop before mask blending'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sources', help='DIV2K HR PNG directory; original names 0001.png..0900.png')
    p.add_argument('output')
    p.add_argument('--array-storage', choices=['npy','npz'], default='npy', help='Lossless array container; npz compresses masks and severity')
    p.add_argument('--severity-storage',choices=['int8','int64'],default='int8',help='Lossless storage for severity labels -1..4')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None, help='Development subset; retains canonical protocol')
    p.add_argument('--require-complete-test', action='store_true', help='Require all 0801..0900 test originals')
    args = p.parse_args()
    sources = sorted(Path(args.sources).glob('*.png'))
    if args.limit is not None:
        if args.limit < 1: raise ValueError('limit must be positive')
        sources = sources[:args.limit]
    if not sources: raise ValueError('No source PNG images found')
    ids = [int(path.stem) for path in sources]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate source ids')
    for source in sources: split_for(source.stem)
    complete_test = set(range(801,901)).issubset(ids)
    if args.require_complete_test and not complete_test:
        raise ValueError('Complete test protocol needs all source ids 0801..0900')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = protocol_document(ids, args.seed, args.array_storage, args.severity_storage)
    protocol_text = json.dumps(protocol, indent=2, ensure_ascii=False)
    (output/'protocol.json').write_text(protocol_text, encoding='utf-8')
    protocol_sha = hashlib.sha256(protocol_text.encode()).hexdigest()
    with (output/'samples.jsonl').open('w', encoding='utf-8') as stream:
        for source in sources:
            for record in write_source(source, output, args.seed, protocol_sha, args.array_storage, args.severity_storage):
                stream.write(json.dumps(record)+'\n')


if __name__ == '__main__': main()
