"""Fixed MUSIQ plus degradation residual, color and predicted-normal safeguards."""
import hashlib
import json
from dataclasses import dataclass, asdict
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .core import image_array


class Musiq:
    def __init__(self,weights,device='cpu'):
        import pyiqa
        self.metric=pyiqa.create_metric('musiq',device=device,pretrained_model_path=str(weights))
        self.device=device; self.calls=0
        self.metadata=dict(metric='musiq',package_version=version('pyiqa'),
                           weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                           rgb_range=[0,1],device=device)
    def __call__(self,image):
        import torch
        tensor=torch.from_numpy(image_array(image).copy()).permute(2,0,1)[None].to(self.device)
        with torch.inference_mode(): result=float(self.metric(tensor).item())
        if not np.isfinite(result): raise ValueError('Non-finite MUSIQ score')
        self.calls+=1
        return result


@dataclass
class FeedbackConfig:
    min_quality_gain: float = 0.0
    quality_tie: float = 0.1
    min_residual_gain: float = 0.005
    max_residual_increase: float = 0.0
    max_color_shift: float = 0.1
    max_normal_change: float = 0.01
    normal_threshold: float = 0.1

    def validate(self):
        if not all(np.isfinite(v) and v>=0 for v in asdict(self).values()):
            raise ValueError('Feedback thresholds must be finite and nonnegative')
        if self.normal_threshold>1: raise ValueError('Invalid normal threshold')


def decide(values,config):
    config.validate()
    if not all(np.isfinite(v) for v in values.values()): return False
    quality_gain=values['quality_gain']; residual_gain=values['residual_gain']
    improves=(quality_gain>config.min_quality_gain and residual_gain>=-config.max_residual_increase) or (
        quality_gain>=-config.quality_tie and residual_gain>=config.min_residual_gain)
    return bool(improves and values['color_shift']<=config.max_color_shift and
                values['normal_change']<=config.max_normal_change)


class JointFeedback:
    def __init__(self,predictor,scorer,config=None):
        self.predictor,self.scorer=predictor,scorer
        self.config=config or FeedbackConfig(); self.config.validate()
        self.last=None; self.calls=0
    def __call__(self,before,after,region):
        before,after=image_array(before),image_array(after)
        if before.shape!=after.shape: raise ValueError('Feedback requires identical shapes/content locations')
        support=region.mask>0
        y,x=np.nonzero(support)
        if not len(y): return False,'empty region'
        bounds=(slice(y.min(),y.max()+1),slice(x.min(),x.max()+1))
        # Both candidates use exactly the same crop, no scores across different regions added.
        quality_before=self.scorer(before[bounds]); quality_after=self.scorer(after[bounds])
        old=self.predictor(before); new=self.predictor(after)
        normal=old.max(axis=2)<self.config.normal_threshold
        values=dict(quality_gain=quality_after-quality_before,
                    residual_gain=float((old[support]-new[support]).mean()),
                    color_shift=float(np.abs(after[support].mean(0)-before[support].mean(0)).max()),
                    normal_change=float(np.abs(after-before)[normal].mean()) if normal.any() else 0.)
        self.last=dict(**values,quality_before=quality_before,quality_after=quality_after)
        self.calls+=1
        return decide(values,self.config),json.dumps(self.last,sort_keys=True)


def load_feedback(path):
    if path is None: return FeedbackConfig()
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    config=FeedbackConfig(**data.get('thresholds',data)); config.validate()
    return config
