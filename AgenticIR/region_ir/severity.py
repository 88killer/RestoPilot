"""Synthetic strength targets, distinct from degradation-presence probability.

Five equal bins on normalized generator strength. These are proxy labels; they
must be calibrated against held-out images and restoration outputs before using
very-low..very-high as perceptual severity claims.
"""
import numpy as np

KINDS = ('noise', 'blur', 'jpeg')
LEVEL_NAMES = ('very low', 'low', 'medium', 'high', 'very high')
BIN_EDGES = (0.2, 0.4, 0.6, 0.8)


def normalized_strength(operation):
    kind = operation['kind']
    if kind == 'noise':
        strength = float(operation['sigma']) / 50
    elif kind == 'blur':
        strength = (float(operation['length']) - 1) / 22
    elif kind == 'jpeg':
        strength = (100 - float(operation['quality'])) / 90
    else:
        raise ValueError(f'Unknown degradation: {kind}')
    return float(np.clip(strength, 0, 1))


def severity_labels(masks, operations, interior_threshold=.99):
    """HxWx3, classes 0..4; -1 excludes normal pixels and soft transitions."""
    if masks.ndim != 3 or masks.shape[2] != 3 or not np.isfinite(masks).all():
        raise ValueError('Expected finite HxWx3 masks')
    result = np.full(masks.shape, -1, dtype=np.int64)
    seen = set()
    for operation in operations:
        kind = operation['kind']
        if kind in seen:
            raise ValueError('Repeated same-kind synthetic operations have undefined severity')
        seen.add(kind)
        index = KINDS.index(kind)
        level = int(np.searchsorted(BIN_EDGES, normalized_strength(operation), side='right'))
        result[..., index][masks[..., index] >= interior_threshold] = level
    return result


def confidence_state(probabilities, confidence_threshold=.2, normal_threshold=.3):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError('Expected HxWx3 probabilities')
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError('Invalid probabilities')
    # Least certain class controls confidence: 0.5 is unknown, not normal.
    confidence = (2 * np.abs(probabilities - .5)).min(axis=2)
    uncertain = confidence < confidence_threshold
    normal = (probabilities.max(axis=2) <= normal_threshold) & ~uncertain
    return dict(confidence=confidence, uncertain=uncertain, normal=normal)
