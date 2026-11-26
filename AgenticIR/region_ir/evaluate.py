"""Reference-only evaluation. All RGB metrics use range [0, 1]."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from .core import image_array, load_image
from .arrays import load_array, load_sample_array

KINDS = ('noise', 'blur', 'jpeg')
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def ssim_map(reference, candidate):
    """Wang-style Gaussian population SSIM, averaged across RGB channels.

    The returned map includes image borders; callers must restrict it to complete
    11x11 windows. No rescaling, luminance conversion, or automatic downsampling.
    """
    reference, candidate = image_array(reference), image_array(candidate)
    if reference.shape != candidate.shape:
        raise ValueError('SSIM image dimensions differ')
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    radius = SSIM_WINDOW // 2
    def average(x):
        return ndimage.gaussian_filter(x, sigma=(SSIM_SIGMA, SSIM_SIGMA, 0),
                                       truncate=radius / SSIM_SIGMA, mode='reflect')
    ma, mb = average(a), average(b)
    va = np.maximum(average(a * a) - ma * ma, 0)
    vb = np.maximum(average(b * b) - mb * mb, 0)
    covariance = average(a * b) - ma * mb
    values = ((2 * ma * mb + .01**2) * (2 * covariance + .03**2)
              / ((ma * ma + mb * mb + .01**2) * (va + vb + .03**2)))
    return values.mean(axis=2)


def window_ssim(values, support):
    """Only centers whose entire 11x11 window lies inside the support are valid."""
    valid = ndimage.binary_erosion(support, structure=np.ones((SSIM_WINDOW, SSIM_WINDOW)),
                                  border_value=0)
    count = int(valid.sum())
    return (float(values[valid].mean()) if count else None), count


def boundary_band(masks, width=5):
    """Union of each class's internal/external bands, including class interfaces.

    Image edges are not imaginary degradation boundaries: all-zero and all-one
    masks have no band. Soft transitions use their nonzero support boundary.
    """
    if width < 1:
        raise ValueError('Boundary width must be positive')
    band = np.zeros(masks.shape[:2], dtype=bool)
    for index in range(3):
        support = masks[:, :, index] > 0
        dilated = ndimage.binary_dilation(support, iterations=width, border_value=0)
        eroded = ndimage.binary_erosion(support, iterations=width, border_value=1)
        band |= dilated ^ eroded
    return band


def segmentation_metrics(probabilities, masks, threshold=.5):
    probabilities = np.asarray(probabilities)
    masks = np.asarray(masks)
    if (masks.ndim != 3 or masks.shape[2] != 3 or probabilities.shape != masks.shape
            or not np.isfinite(probabilities).all() or probabilities.min() < 0
            or probabilities.max() > 1 or not np.isfinite(masks).all()
            or masks.min() < 0 or masks.max() > 1 or not 0 < threshold <= 1):
        raise ValueError('Expected matching HxWx3 probabilities/labels in [0, 1]')
    predicted, target = probabilities >= threshold, masks >= .5
    results = {}
    for index, name in enumerate(KINDS):
        a, b = predicted[:, :, index], target[:, :, index]
        intersection = int((a & b).sum())
        union, total = int((a | b).sum()), int(a.sum() + b.sum())
        results[name] = dict(iou=intersection / union if union else None,
                             dice=2 * intersection / total if total else None,
                             intersection=intersection, union=union,
                             target_pixels=int(b.sum()), predicted_pixels=int(a.sum()))
    normal = masks.max(axis=2) == 0
    count = int(normal.sum())
    false_positive = int((predicted.any(axis=2) & normal).sum())
    return dict(classes=results, normal_false_positive_rate=false_positive / count if count else None,
                normal_false_positive_pixels=false_positive, normal_pixels=count,
                probability_threshold=threshold, target_threshold=.5)


class LPIPSScorer:
    """Optional fixed LPIPS configuration; pass one instance to an evaluation run."""
    def __init__(self, device='cuda', net='alex'):
        try:
            import torch
            import lpips
        except ImportError as error:
            raise RuntimeError('LPIPS requested: install torch and lpips in the evaluation environment') from error
        if net not in ('alex', 'vgg', 'squeeze'):
            raise ValueError('Unsupported LPIPS backbone')
        self.torch, self.device = torch, device
        self.model = lpips.LPIPS(net=net, version='0.1').eval().to(device)
        self.configuration = dict(network=net, version='0.1', input_range=[-1, 1],
                                  minimum_side=64, resize=False)
        self.calls = 0

    def __call__(self, reference, candidate):
        reference, candidate = image_array(reference), image_array(candidate)
        if reference.shape != candidate.shape:
            raise ValueError('LPIPS image dimensions differ')
        if min(reference.shape[:2]) < 64:
            return None
        def tensor(a):
            return self.torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        with self.torch.inference_mode():
            value = float(self.model(tensor(reference), tensor(candidate)).item())
        self.calls += 1
        if not np.isfinite(value):
            raise ValueError('LPIPS returned a non-finite score')
        return value


def metrics(clean, before, after, masks, predictions=None, lpips=None):
    clean, before, after = map(image_array, [clean, before, after])
    masks = np.asarray(masks)
    if clean.shape != before.shape or clean.shape != after.shape or masks.shape != clean.shape:
        raise ValueError('Evaluation dimensions differ')
    if not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
        raise ValueError('Invalid masks')
    squared_error = (after.astype(np.float64) - clean.astype(np.float64)) ** 2
    def psnr(support):
        if not support.any():
            return None
        mse = float(squared_error[support].mean())
        return 'infinity' if mse == 0 else float(-10 * np.log10(mse))
    support = masks.max(axis=2) > 0
    normal = ~support
    boundary = boundary_band(masks)
    ssim_values = ssim_map(clean, after)
    full_ssim, full_count = window_ssim(ssim_values, np.ones(support.shape, dtype=bool))
    region_ssim, region_count = {}, {}
    for i, kind in enumerate(KINDS):
        region_ssim[kind], region_count[kind] = window_ssim(ssim_values, masks[:, :, i] >= .99)
    gradient_sum, gradient_count = 0., 0
    for axis in (0, 1):
        error = np.abs(np.diff(after.astype(np.float64), axis=axis)
                       - np.diff(clean.astype(np.float64), axis=axis))
        selected = (boundary[1:] | boundary[:-1]) if axis == 0 else (boundary[:, 1:] | boundary[:, :-1])
        gradient_sum += float(error[selected].sum())
        gradient_count += int(selected.sum()) * 3
    result = dict(
        psnr=psnr(np.ones(support.shape, dtype=bool)), ssim=full_ssim, ssim_valid_windows=full_count,
        region_psnr={kind: psnr(masks[:, :, i] >= .99) for i, kind in enumerate(KINDS)},
        region_ssim=region_ssim, region_ssim_valid_windows=region_count,
        normal_mae=float(np.abs(after - clean)[normal].mean()) if normal.any() else None,
        normal_change=float(np.abs(after - before)[normal].mean()) if normal.any() else None,
        normal_pixels=int(normal.sum()), boundary_pixels=int(boundary.sum()),
        boundary_mae=float(np.abs(after - clean)[boundary].mean()) if boundary.any() else None,
        boundary_gradient_mae=gradient_sum / gradient_count if gradient_count else None,
        boundary_gradient_values=gradient_count)
    if predictions is not None:
        result['segmentation'] = segmentation_metrics(predictions, masks)
    if lpips is not None:
        result['lpips'] = lpips(clean, after)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample'); parser.add_argument('run'); parser.add_argument('output')
    parser.add_argument('--predictions', help='Optional HxWx3 probability array to evaluate localization')
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    sample, run = Path(args.sample), Path(args.run)
    scorer = LPIPSScorer(args.device) if args.lpips else None
    predictions = load_array(args.predictions) if args.predictions else None
    result = metrics(load_image(sample / 'clean.png'), load_image(sample / 'input.png'),
                     load_image(run / 'output.png'), load_sample_array(sample, 'masks'),
                     predictions=predictions, lpips=scorer)
    result['metric_protocol'] = dict(rgb=True, data_range=1, ssim_window=SSIM_WINDOW, ssim_sigma=SSIM_SIGMA,
                                    ssim_population_covariance=True, region_interior_threshold=.99,
                                    boundary_width=5, lpips=scorer.configuration if scorer else None)
    result['execution'] = json.loads((run / 'summary.json').read_text())
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
