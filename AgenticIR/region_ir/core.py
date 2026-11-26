from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage

KINDS = ('noise', 'blur', 'jpeg')


def image_array(image):
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
        raise ValueError('Expected finite HxWx3 RGB image')
    if a.min() < 0 or a.max() > 1:
        raise ValueError('RGB values must be in [0, 1]')
    return a


def save_image(path, image):
    Image.fromarray(np.rint(image_array(image) * 255).astype('uint8')).save(path)


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255


@dataclass
class Region:
    id: int
    mask: np.ndarray
    probabilities: tuple


def build_regions(probabilities, threshold=0.5, min_area=64, max_regions=4):
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 3 or p.shape[2] != 3 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Expected HxWx3 probabilities in [0, 1]')
    if not 0 < threshold <= 1 or min_area < 1 or max_regions < 1:
        raise ValueError('Invalid region thresholds')
    # Split connected regions by their multi-label signature, retaining overlaps.
    signatures = ((p >= threshold) * np.array([1, 2, 4])).sum(axis=2)
    components = []
    for signature in range(1, 8):
        labels, count = ndimage.label(signatures == signature)
        for index in range(1, count + 1):
            support = labels == index
            if support.sum() >= min_area:
                weight = np.where(support, p.max(axis=2), 0)
                components.append((support.sum(), weight, tuple(p[support].mean(axis=0))))
    components.sort(key=lambda x: x[0], reverse=True)
    return [Region(i, mask, probs) for i, (_, mask, probs) in enumerate(components[:max_regions])]


def candidate_paths(tools, max_steps=3):
    """Offline supervision/diagnostic only; never select with test ground truth."""
    return [()] + [p for n in range(1, min(max_steps, len(tools)) + 1)
                   for p in permutations(tools, n)]


class DirectoryTool:
    """Adapt original Tool(input_dir, output_dir, silent=True) without changing it."""
    def __init__(self, tool):
        self.tool = tool

    def __call__(self, crop):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input', root / 'output'
            source.mkdir()
            output.mkdir()
            save_image(source / 'input.png', crop)
            self.tool(source, output, silent=True)
            return load_image(output / 'output.png')


class CommandTool:
    """Separate model environment, Windows-safe argv; no shell interpolation."""
    def __init__(self, command, cwd=None, timeout=300):
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError('command must be an argv array')
        self.command, self.cwd, self.timeout = command, cwd, timeout
        self.last_metadata = None

    def __call__(self, crop):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'input.png', root / 'output.png'
            save_image(source, crop)
            self.last_metadata = None
            metadata = root / "metadata.json"
            argv = [s.replace('{input}', str(source)).replace('{output}', str(target)).replace('{metadata}', str(metadata)) for s in self.command]
            subprocess.run(argv, cwd=self.cwd, check=True, timeout=self.timeout)
            if metadata.exists():
                self.last_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            return load_image(target)


def execute_region(image, region, tool, context=32, window=8, audit_dir=None, soft=True):
    before = image_array(image)
    mask = np.asarray(region.mask, dtype=np.float32)
    if mask.shape != before.shape[:2] or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('Invalid region mask')
    if context < 0 or window < 1:
        raise ValueError('Invalid context/window')
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return before.copy()
    y0, y1 = max(0, yy.min() - context), min(before.shape[0], yy.max() + context + 1)
    x0, x1 = max(0, xx.min() - context), min(before.shape[1], xx.max() + context + 1)
    crop = before[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    padded = np.pad(crop, ((0, -h % window), (0, -w % window), (0, 0)), mode='reflect')
    restored = image_array(tool(padded.copy()))
    if restored.shape != padded.shape:
        raise ValueError('Tool changed image dimensions')
    if audit_dir is not None:
        audit = Path(audit_dir)
        audit.mkdir(parents=True, exist_ok=True)
        save_image(audit / "crop_input.png", padded)
        save_image(audit / "crop_output.png", restored)
        (audit / "crop.json").write_text(json.dumps(dict(box=[int(x0),int(y0),int(x1),int(y1)], unpadded_shape=[h,w], context=context, window=window)))
    weight = mask[y0:y1, x0:x1, None]
    if not soft:
        weight = (weight > 0).astype(np.float32)
    fused = crop + weight * (restored[:h, :w] - crop)
    result = before.copy()
    result[y0:y1, x0:x1] = np.where(weight > 0, fused, crop)
    return np.clip(result, 0, 1)


def run(image, regions, tools, planner, feedback, output_dir, budget=12, max_steps=2,
        context=32, soft=True, time_budget=None):
    """Planner/feedback see current image and predicted state, never clean references."""
    if budget < 0 or max_steps < 1:
        raise ValueError('Invalid budget/steps')
    current = image_array(image).copy()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_image(output / 'input.png', current)
    calls, history = 0, []
    run_start = perf_counter()
    termination = "completed"
    for region in regions:
        failed = set()
        for step in range(max_steps):
            if calls >= budget:
                termination = "call_budget"
                break
            if time_budget is not None and perf_counter()-run_start >= time_budget:
                termination = "time_budget"
                break
            planning_start = perf_counter()
            action = planner(current.copy(), region, tuple(history), budget - calls, frozenset(failed))
            record = dict(region_id=region.id, step=step, tool_id=action, calls=calls, planning_seconds=perf_counter()-planning_start)
            if action in ('stop', 'noop'):
                record.update(accepted=False, reason=action, seconds=0.)
                history.append(record)
                with (output / 'trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                if action == 'stop':
                    break
                continue
            if action not in tools or action in failed:
                raise ValueError('Planner selected unknown or previously failed tool')
            calls += 1  # Attempted and rejected tool calls consume budget too.
            start = perf_counter()
            stem = f'r{region.id}_s{step}'
            save_image(output / f'{stem}_before.png', current)
            error_text = None
            try:
                candidate = execute_region(current, region, tools[action], context=context, soft=soft, audit_dir=output/stem)
                save_image(output / f'{stem}_candidate.png', candidate)
                accepted, reason = feedback(current.copy(), candidate.copy(), region)
                if accepted:
                    current = candidate
                else:
                    failed.add(action)
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
                error_text = f'{type(error).__name__}: {error}'
                accepted, reason = False, error_text
                failed.add(action)
            record.update(accepted=bool(accepted), reason=str(reason), seconds=perf_counter()-start, calls=calls)
            record.update(error=error_text, remaining_budget=budget-calls,
                          feedback=getattr(feedback,"last",None), tool_metadata=getattr(tools[action],"last_metadata",None))
            history.append(record)
            save_image(output / f'{stem}_result.png', current)
            with (output / 'trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    save_image(output / 'output.png', current)
    (output / 'summary.json').write_text(json.dumps(dict(calls=calls, history=history, termination=termination, seconds=perf_counter()-run_start,
        failed_calls=sum(r.get("error") is not None for r in history)), ensure_ascii=False, indent=2), encoding='utf-8')
    return current, history
