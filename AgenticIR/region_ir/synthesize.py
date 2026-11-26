"""Generate a reproducible diagnostic sample. Split source images before calling."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, features
from scipy import ndimage

from .core import load_image, save_image


def synthesize(clean, masks, operations, seed):
    current = clean.copy()
    rng = np.random.default_rng(seed)
    labels = np.zeros((*clean.shape[:2], 3), dtype=np.float32)
    kinds = ['noise', 'blur', 'jpeg']
    if len(masks) != len(operations):
        raise ValueError('Each operation requires one mask')
    for mask, operation in zip(masks, operations):
        kind = operation['kind']
        if kind == 'noise':
            degraded = np.clip(current + rng.normal(0, operation['sigma']/255, current.shape), 0, 1)
        elif kind == 'blur':
            length, angle = operation['length'], operation['angle']
            if length < 1 or length % 2 != 1:
                raise ValueError('Motion kernel length must be positive and odd')
            kernel = np.zeros((length, length))
            kernel[length//2] = 1
            kernel = np.maximum(ndimage.rotate(kernel, angle, reshape=False, order=1), 0)
            kernel /= kernel.sum()
            degraded = ndimage.convolve(current, kernel[:, :, None], mode='reflect')
        elif kind == 'jpeg':
            buffer = BytesIO()
            Image.fromarray(np.rint(current*255).astype('uint8')).save(buffer, 'JPEG', quality=operation['quality'], subsampling=0)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                degraded = np.asarray(decoded, dtype=np.float32)/255
        else:
            raise ValueError(kind)
        if mask.shape != clean.shape[:2] or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('Invalid degradation mask')
        current = current + mask[:, :, None] * (degraded-current)
        labels[:, :, kinds.index(kind)] = np.maximum(labels[:, :, kinds.index(kind)], mask)
    return np.clip(current, 0, 1), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clean')
    parser.add_argument('output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source-id', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    clean = load_image(args.clean)
    h, w = clean.shape[:2]
    masks = []
    for i in range(3):
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h//4:3*h//4, i*w//3:(i+1)*w//3] = 1
        masks.append(mask)
    operations = [dict(kind='noise', sigma=25), dict(kind='blur', length=15, angle=45), dict(kind='jpeg', quality=20)]
    degraded, labels = synthesize(clean, masks, operations, args.seed)
    save_image(output/'clean.png', clean)
    save_image(output/'input.png', degraded)
    np.save(output/'diagnostic_masks.npy', labels)
    metadata = dict(source_id=args.source_id, split=args.split, seed=args.seed, operations=operations,
                    mask='three disjoint rectangles, middle half height', jpeg_subsampling=0,
                    jpeg_block_origin=[0, 0], jpeg_codec=features.version('jpg'))
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
