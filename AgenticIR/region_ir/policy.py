"""Shared inference features and bounded, history-aware regional tool selection."""
import numpy as np
from scipy import ndimage
from .provenance import perception_payload,payload_sha256
from .severity import confidence_state

TOOLS = ('swinir15', 'swinir50', 'restormer', 'fbcnn')
ACTIONS = TOOLS + ('stop', 'noop')
FEATURE_VERSION = 2
STATE_SIZE = 39


def prediction_state(predictor, image):
    state = predictor.state(image) if hasattr(predictor, 'state') else {'probabilities': predictor(image)}
    probabilities = np.asarray(state['probabilities'], dtype=np.float32)
    if probabilities.shape != image.shape or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Predictor must return finite HxWx3 probabilities in [0,1]')
    severity = np.asarray(state.get('severity', np.zeros_like(probabilities)), dtype=np.float32)
    confidence = np.asarray(state['confidence'] if 'confidence' in state else
                            confidence_state(probabilities)['confidence'], dtype=np.float32)
    if severity.shape != image.shape or confidence.shape != image.shape[:2]:
        raise ValueError('Invalid severity/confidence shape')
    if not np.isfinite(severity).all() or not np.isfinite(confidence).all() or np.any((severity < 0) | (severity > 1)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Severity/confidence must be finite in [0,1]')
    return dict(probabilities=probabilities, severity=severity, confidence=confidence,
                has_severity=bool(state.get('severity_available', 'severity' in state)))


def features(image, region, probabilities, history, remaining):
    """Legacy 16 features retained for compatibility with initial experiments."""
    support = region.mask > 0
    pixels = image[support]
    if not len(pixels):
        raise ValueError('Empty region')
    gray = image.mean(axis=2)
    highpass = gray-ndimage.gaussian_filter(gray, 1)
    noise = float(np.median(np.abs(highpass[support]))/.6745)
    used = [float(tool in history) for tool in TOOLS]
    return np.asarray(list(probabilities[support].mean(axis=0))+list(pixels.mean(axis=0))+
                      list(pixels.std(axis=0))+[float(support.mean()), noise]+used+[remaining/12], dtype=np.float32)


def state_features(image, region, state, history, remaining, steps_left, no_history=False):
    """Only current-image estimates/history/budget; no clean images or synthesis labels."""
    history = [] if no_history else list(history)
    basic = features(image, region, state['probabilities'], history, remaining)
    support = region.mask > 0
    severity = state['severity'][support].mean(axis=0)
    confidence = [state['confidence'][support].mean()]
    ordered = np.zeros((3, len(ACTIONS)), dtype=np.float32)
    for index, action in enumerate(history[-3:]):
        if action in ACTIONS:
            ordered[index, ACTIONS.index(action)] = 1
    result = np.concatenate([basic, severity, confidence, ordered.ravel(), [steps_left/3]])
    if result.shape != (STATE_SIZE,) or not np.isfinite(result).all():
        raise ValueError('Invalid policy features')
    return result.astype(np.float32)


def action_features(state, action):
    return np.concatenate([state, np.eye(len(ACTIONS), dtype=np.float32)[ACTIONS.index(action)]])


def candidate_tools(image, region, state, tools, threshold=.5):
    """Select at most one weight per task using input estimates, never synthetic sigma."""
    support = region.mask > 0
    probabilities = state['probabilities'][support].mean(axis=0)
    if state.get('has_severity', False):
        denoiser = 'swinir15' if state['severity'][support, 0].mean() < .5 else 'swinir50'
    else:
        denoiser = 'swinir15' if features(image, region, state['probabilities'], [], 12)[10] < 30/255 else 'swinir50'
    # Missing selected weights are a configuration error, not permission to use truth.
    chosen = [(0, denoiser), (1, 'restormer'), (2, 'fbcnn')]
    return [tool for index, tool in chosen if probabilities[index] >= threshold and tool in tools]


class Ranker:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as state:
            self.params = {key: state[key].copy() for key in state.files}
        for key in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2'):
            if key not in self.params or not np.isfinite(self.params[key]).all():
                raise ValueError(f'Invalid ranker parameter: {key}')
        if np.any(self.params['scale'] <= 0):
            raise ValueError('Invalid ranker scale')
        self.supported_actions = tuple(str(v) for v in self.params.get('supported_actions', ACTIONS))
        if not self.supported_actions or not set(self.supported_actions).issubset(ACTIONS) or 'stop' not in self.supported_actions:
            raise ValueError('Invalid trained action coverage')
        self.action_coverage_known = 'supported_actions' in self.params
        self.perception_signature = str(self.params['perception_signature']) if 'perception_signature' in self.params else None
        if self.perception_signature:
            import json
            if 'perception_json' not in self.params or payload_sha256(json.loads(str(self.params['perception_json']))) != self.perception_signature:
                raise ValueError('Corrupt ranker perception identity')
        self.feature_version = int(self.params.get('feature_version', 1))
        self.cost_weight = float(self.params.get('cost_weight', 0.))
        self.no_history = bool(self.params.get('no_history', False))
        self.no_cost = bool(self.params.get('no_cost', False))

    def require_coverage(self, actions=()):
        if not self.action_coverage_known:
            raise ValueError('Ranker has no training action coverage; retrain from path labels')
        missing = set(actions) - set(self.supported_actions)
        if missing:
            raise ValueError('Ranker training is missing actions: ' + ', '.join(sorted(missing)))

    def require_perception(self, provenance, threshold=.5):
        if self.perception_signature is None:
            raise ValueError('Ranker lacks perception provenance; recollect paths and retrain')
        if self.perception_signature != payload_sha256(perception_payload(provenance,threshold)):
            raise ValueError('Ranker perception differs: localizer, calibration, tiling or threshold mismatch')

    def predict(self, state, action):
        if action not in self.supported_actions:
            raise ValueError('Action absent from ranker training: ' + action)
        p = self.params
        x = action_features(state, action)
        if x.shape != p['mean'].shape:
            raise ValueError('Policy feature schema differs from checkpoint; retrain the ranker')
        x = (x-p['mean'])/p['scale']
        hidden = np.maximum(np.einsum('i,ij->j', x, p['w1'])+p['b1'], 0)
        if p['w2'].ndim == 1:
            value = float(np.einsum('i,i->', hidden, p['w2'])+p['b2'])
            return dict(gain=value, cost=0., utility=value)
        prediction = np.einsum('i,ij->j', hidden, p['w2'])+p['b2']
        gain, cost = float(prediction[0]), max(0., float(prediction[1]))
        return dict(gain=gain, cost=cost, utility=gain-self.cost_weight*cost)

    def score(self, state, action, no_cost=False, *, inference_cost_weight=None):
        result = self.predict(state, action)
        if no_cost:
            return result['gain']
        if inference_cost_weight is not None:
            return result['gain'] - inference_cost_weight * result['cost']
        return result['utility']


class Planner:
    def __init__(self, predictor, tools, ranker=None, no_history=False, no_cost=False,
                 force_initial_tool=False, max_steps=2, min_confidence=.5, threshold=.5,
                 *, require_first_attempt=False, inference_cost_weight=None):
        if max_steps not in (1, 2, 3) or not 0 <= min_confidence <= 1 or not 0 < threshold <= 1:
            raise ValueError('Invalid planner configuration')
        if isinstance(ranker, Ranker):
            ranker.require_coverage()
            if hasattr(predictor,'provenance'):
                ranker.require_perception(predictor.provenance,threshold)
        self.predictor, self.tools, self.ranker = predictor, tools, ranker
        self.no_history, self.no_cost = no_history, no_cost
        self.force_initial_tool, self.max_steps = force_initial_tool, max_steps
        self.min_confidence, self.threshold = min_confidence, threshold
        if ranker and ranker.feature_version >= 2:
            if ranker.no_history != no_history or ranker.no_cost != no_cost:
                raise ValueError('History/cost ablation must use a separately trained matching ranker')
        if inference_cost_weight is not None and (
            ranker is None or ranker.feature_version < 2 or no_cost
            or isinstance(inference_cost_weight, bool)
            or not isinstance(inference_cost_weight, (int, float))
            or not np.isfinite(inference_cost_weight) or inference_cost_weight < 0
        ):
            raise ValueError('Inference cost weight requires a gain/cost ranker and a finite nonnegative weight')
        self.inference_cost_weight = inference_cost_weight
        if require_first_attempt and (ranker is None or force_initial_tool):
            raise ValueError('Required first attempt needs a learned ranker without per-region forced repair')
        self.require_first_attempt = require_first_attempt
        self.last_override = None

    def __call__(self, image, region, history, remaining, failed):
        self.last_override = None
        local = [r for r in history if r['region_id'] == region.id]
        used = [r['tool_id'] for r in local]
        steps_left = self.max_steps-len(local)
        if remaining <= 0 or steps_left <= 0:
            return 'stop'
        prediction = prediction_state(self.predictor, image)
        selected = candidate_tools(image, region, prediction, self.tools, self.threshold)
        available = [tool for tool in selected if tool not in used and tool not in failed
                     and not (tool.startswith('swinir') and any(t.startswith('swinir') for t in used))]
        first_required = self.require_first_attempt and not history
        force = first_required or (self.force_initial_tool and not local)
        if first_required:
            available = [t for t in available if t in self.ranker.supported_actions]
        if force:
            # Force-repair ablation excludes both initial No-Op and Stop; choose any valid tool if perception is negative.
            available = available or [t for t in TOOLS if t in self.tools and t not in failed]
        confidence = float(prediction['confidence'][region.mask > 0].mean())
        if not force and confidence < self.min_confidence:
            return 'noop' if 'noop' not in used and steps_left > 1 else 'stop'
        if self.ranker:
            state = state_features(image, region, prediction, used, remaining, steps_left, self.no_history)
            if self.ranker.feature_version == 1:
                state = features(image, region, prediction['probabilities'], [] if self.no_history else used, remaining)
            choices = [tool for tool in available if tool in self.ranker.supported_actions]
            if force and not choices:
                raise ValueError('Forced repair has no tool supported by ranker training')
            if not force:
                choices.append('stop')
                if 'noop' not in used and steps_left > 1 and 'noop' in self.ranker.supported_actions:
                    choices.append('noop')
            if not choices:
                return 'stop'
            # Whenever stopping is allowed, its exact value is zero.
            scores = {
                action: (0. if action == 'stop' else self.ranker.score(
                    state, action, self.no_cost, inference_cost_weight=self.inference_cost_weight
                ))
                for action in choices
            }
            best = max(choices, key=lambda action: (scores[action], action == 'stop'))
            if first_required:
                self.last_override = {'rule': 'required_first_attempt'}
            return best
        for tool in ('restormer', 'swinir15', 'swinir50', 'fbcnn'):
            if tool in available:
                return tool
        return 'stop'


class ResidualFeedback:
    """Compatibility feedback; the experiment runner can supply joint MUSIQ feedback."""
    def __init__(self, predictor, min_gain=.005, max_color_shift=.1):
        self.predictor = predictor
        self.min_gain, self.max_color_shift = min_gain, max_color_shift

    def __call__(self, before, after, region):
        support = region.mask > 0
        old = self.predictor(before)[support].mean()
        new = self.predictor(after)[support].mean()
        shift = np.abs(after[support].mean(axis=0)-before[support].mean(axis=0)).max()
        accepted = old-new >= self.min_gain and shift <= self.max_color_shift
        return bool(accepted), f'residual_gain={old-new:.6f}; color_shift={shift:.6f}'
