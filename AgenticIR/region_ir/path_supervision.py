"""Bounded offline path return labels. All normalization is fitted on training trees."""
from functools import lru_cache
import numpy as np
from .policy import ACTIONS, action_features

COMPONENTS = ('mse_gain', 'seconds', 'normal_damage', 'boundary_damage')


def fit_normalization(trees, cost_weight=.1, normal_weight=.1, boundary_weight=.1):
    if not trees or any(tree['split'] != 'train' for tree in trees):
        raise ValueError('Normalization may only be fitted on training sources')
    values = np.asarray([edge['components'] for tree in trees for edge in tree['edges'] if not edge['failed']], dtype=np.float64)
    if not len(values):
        raise ValueError('No successful training paths to fit normalization')
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('Invalid reward components')
    scales = np.maximum(np.quantile(np.abs(values), .95, axis=0), [1e-6, 1., 1e-6, 1e-6])
    return dict(version=2, components=list(COMPONENTS), scales=scales.tolist(),
                cost_weight=float(cost_weight), normal_weight=float(normal_weight), boundary_weight=float(boundary_weight),
                fitted_on='train', source_ids=sorted({tree['source_id'] for tree in trees}))


def validate_normalization(normalization):
    scales = np.asarray(normalization['scales'], dtype=float)
    if normalization.get('version') != 2 or normalization.get('fitted_on') != 'train' or scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Expected frozen training-only normalization v2')
    if any(not np.isfinite(normalization[k]) or normalization[k] < 0 for k in ('cost_weight','normal_weight','boundary_weight')):
        raise ValueError('Invalid reward weights')
    return scales


def path_labels(tree, normalization, max_budget=12, no_history=False, no_cost=False):
    """Return gain/cost targets for optimal bounded continuation, including empty path.

    Stop is absorbing. No-Op spends a decision step without spending a tool call.
    Only precomputed prefix outputs are traversed; never called during deployment.
    """
    scales = validate_normalization(normalization)
    nodes = {tuple(node['path']): np.asarray(node['state'], dtype=np.float32) for node in tree['nodes']}
    edges = {(tuple(edge['parent']), edge['action']): edge for edge in tree['edges']}
    max_steps = int(tree['max_steps'])
    cost_weight = 0. if no_cost else float(normalization['cost_weight'])
    nw, bw = normalization['normal_weight'], normalization['boundary_weight']

    def edge_value(edge):
        components = np.asarray(edge['components'], dtype=float)/scales
        gain = -1. if edge['failed'] else components[0]-nw*components[2]-bw*components[3]
        return float(gain), max(0., float(components[1]))

    @lru_cache(None)
    def best(prefix, horizon, budget):
        choices = [(0., 0.)]  # Empty continuation is always valid.
        if horizon > 0 and budget > 0:
            for (parent, action), edge in edges.items():
                if parent != prefix:
                    continue
                gain, cost = edge_value(edge)
                child = prefix+(action,)
                if not edge['failed'] and child in nodes:
                    next_gain, next_cost = best(child, horizon-1, budget-1)
                    gain += next_gain
                    cost += next_cost
                choices.append((gain, cost))
        return max(choices, key=lambda value: (value[0]-cost_weight*value[1], -value[1]))

    x, targets, metadata = [], [], []
    for prefix, node_state in nodes.items():
        # A No-Op state is the identical image with one fewer decision step and explicit history.
        for had_noop in (False, True):
            max_horizon = max_steps-len(prefix)-int(had_noop)
            if max_horizon <= 0:
                continue
            for horizon in range(1, max_horizon+1):
                for budget in range(1, max_budget+1):
                    state = node_state.copy()
                    state[15] = budget/12
                    state[-1] = horizon/3
                    if had_noop:
                        ordered = np.zeros((3,len(ACTIONS)), dtype=np.float32)
                        for i, action in enumerate((prefix+('noop',))[-3:]):
                            ordered[i,ACTIONS.index(action)] = 1
                        state[20:38] = ordered.ravel()
                    if no_history:
                        state[11:15] = 0
                        state[20:38] = 0
                    actions = ['stop'] + ([] if had_noop or horizon == 1 else ['noop'])
                    actions += [a for parent,a in edges if parent == prefix]
                    for action in actions:
                        gain, cost = 0., 0.
                        if action == 'noop':
                            gain, cost = best(prefix, horizon-1, budget)
                        elif action != 'stop':
                            edge = edges[(prefix,action)]
                            gain, cost = edge_value(edge)
                            if not edge['failed'] and prefix+(action,) in nodes:
                                future_gain, future_cost = best(prefix+(action,), horizon-1, budget-1)
                                gain += future_gain
                                cost += future_cost
                        x.append(action_features(state, action))
                        targets.append([gain, cost])
                        metadata.append(dict(prefix=list(prefix), action=action, horizon=horizon, budget=budget, had_noop=had_noop))
    return np.asarray(x,dtype=np.float32), np.asarray(targets,dtype=np.float32), metadata
