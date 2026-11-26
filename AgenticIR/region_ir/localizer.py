"""Four-scale U-Net with independent presence and conditional severity heads."""
import hashlib
from math import gcd
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .core import image_array
from .provenance import predictor_provenance
from .severity import confidence_state
from .calibrate_localizer import temperature_probabilities, temperature_severity


class Block(nn.Sequential):
    def __init__(self, inputs, outputs, normalization='none'):
        if normalization == 'none':
            layers = [nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                      nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU()]
        elif normalization == 'group':
            groups = gcd(8, outputs)
            layers = [nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(groups, outputs), nn.ReLU(),
                      nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(groups, outputs), nn.ReLU()]
        else:
            raise ValueError('Unsupported normalization')
        super().__init__(*layers)


class Localizer(nn.Module):
    def __init__(self, width=32, severity=False, normalization='none', input_features='rgb'):
        super().__init__()
        if width < 1: raise ValueError('width must be positive')
        if input_features not in ('rgb','rgb_hp'): raise ValueError('Invalid input features')
        self.input_features = input_features
        input_channels = 3 if input_features == 'rgb' else 9
        self.width = width
        self.has_severity = severity
        self.normalization = normalization
        self.encoders = nn.ModuleList([Block(input_channels,width,normalization), Block(width,width*2,normalization),
                                      Block(width*2,width*4,normalization), Block(width*4,width*8,normalization)])
        self.decoders = nn.ModuleList([Block(width*12,width*4,normalization), Block(width*6,width*2,normalization), Block(width*3,width,normalization)])
        self.head = nn.Conv2d(width,3,1)
        if severity:
            self.severity_head = nn.Conv2d(width,15,1)

    def _features(self, image):
        skips = []
        x = image
        if self.input_features == 'rgb_hp':
            local3 = F.avg_pool2d(F.pad(image,(1,1,1,1),mode='reflect'),3,stride=1)
            local9 = F.avg_pool2d(F.pad(image,(4,4,4,4),mode='reflect'),9,stride=1)
            x = torch.cat([image,4*(image-local3),4*(image-local9)],dim=1)
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x,2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x,skip], dim=1))
        return x

    def forward(self, image):
        return self.head(self._features(image))

    def forward_state(self, image):
        features = self._features(image)
        logits = self.head(features)
        severity = self.severity_head(features).reshape(features.shape[0],3,5,*features.shape[-2:]) if self.has_severity else None
        return logits, severity


def segmentation_loss(logits, labels, positive_weight=None):
    probabilities = logits.sigmoid()
    dims = (0,2,3)
    dice = (2*(probabilities*labels).sum(dims)+1)/(probabilities.sum(dims)+labels.sum(dims)+1)
    weight = None
    if positive_weight is not None:
        weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
        if weight.shape != (3,) or not torch.isfinite(weight).all() or not (weight > 0).all():
            raise ValueError('Presence weights must contain three finite positive values')
        weight = weight.reshape(1, 3, 1, 1)
    return F.binary_cross_entropy_with_logits(logits,labels,pos_weight=weight) + 1-dice.mean()


def severity_loss(logits, labels):
    """Cross-entropy on labeled interior only; transitions/normal have target -1."""
    if logits is None: raise ValueError('Severity head is required')
    b, types, levels, h, w = logits.shape
    target = labels.reshape(b*types,h,w)
    prediction = logits.reshape(b*types,levels,h,w)
    if not (target>=0).any():
        return prediction.sum()*0
    return F.cross_entropy(prediction,target,ignore_index=-1)


def initialize_from_checkpoint(model, state):
    """Zero-expansion preserves the RGB function when enabling residual inputs."""
    source = state.get('input_features','rgb')
    weights = state['model']
    conversion = None
    if source != model.input_features:
        if source != 'rgb' or model.input_features != 'rgb_hp':
            raise ValueError('Unsupported initial input-feature conversion')
        weights = dict(weights)
        key = 'encoders.0.0.weight'
        expanded = torch.zeros_like(model.state_dict()[key], device='cpu')
        expanded[:, :3] = weights[key].cpu()
        weights[key] = expanded
        conversion = 'rgb_to_rgb_hp_zero_expansion'
    model.load_state_dict(weights,strict=True)
    return conversion


@torch.inference_mode()
def tiled_probabilities(model, image, device, tile=256, overlap=32, tile_batch_size=4):
    """Shared deployment/validation tiling; batching does not change GN statistics."""
    if tile < 8 or not 0 <= overlap < tile or tile_batch_size < 1:
        raise ValueError('Invalid tile/overlap/batch size')
    image = image_array(image)
    h, w = image.shape[:2]
    padded = np.pad(image, ((0,max(0,tile-h)),(0,max(0,tile-w)),(0,0)), mode='reflect')
    ph, pw = padded.shape[:2]
    total = np.zeros_like(padded, dtype=np.float32)
    levels = np.zeros((ph,pw,3,5), dtype=np.float32)
    weights = np.zeros((ph,pw,1), dtype=np.float32)
    stride = tile-overlap
    ys = sorted(set(list(range(0,ph-tile+1,stride))+[ph-tile]))
    xs = sorted(set(list(range(0,pw-tile+1,stride))+[pw-tile]))
    coordinates = [(y,x) for y in ys for x in xs]
    for start in range(0,len(coordinates),tile_batch_size):
        selected = coordinates[start:start+tile_batch_size]
        batch = np.stack([padded[y:y+tile,x:x+tile] for y,x in selected])
        tensor = torch.from_numpy(batch).permute(0,3,1,2).to(device)
        logits, severity_logits = model.forward_state(tensor)
        predictions = logits.sigmoid().permute(0,2,3,1).cpu().numpy()
        distributions = (severity_logits.softmax(dim=2).permute(0,3,4,1,2).cpu().numpy()
                         if severity_logits is not None else None)
        for index,(y,x) in enumerate(selected):
            total[y:y+tile,x:x+tile] += predictions[index]
            if distributions is not None:
                levels[y:y+tile,x:x+tile] += distributions[index]
            weights[y:y+tile,x:x+tile] += 1
    return (total/weights)[:h,:w], (levels/weights[...,None])[:h,:w]


@torch.inference_mode()
def deployment_logits(model, image, device, tile=256, overlap=32, tile_batch_size=4):
    """Probability-equivalent logits for validation losses, after deployment tiling."""
    probabilities, distributions = tiled_probabilities(model,image,device,tile,overlap,tile_batch_size)
    p = torch.from_numpy(probabilities).permute(2,0,1)[None].to(device)
    logits = torch.logit(p.clamp(1e-6,1-1e-6))
    levels = (torch.from_numpy(distributions).permute(2,3,0,1)[None].to(device).clamp_min(1e-8).log()
              if model.has_severity else None)
    return logits, levels


class Predictor:
    def __init__(self, checkpoint, device='cuda', tile=256, overlap=32, calibration=None, tile_batch_size=4):
        if tile < 8 or not 0 <= overlap < tile or tile_batch_size < 1:
            raise ValueError('Invalid tile/overlap')
        state = torch.load(checkpoint, map_location='cpu')
        has_severity = 'severity_head.weight' in state['model']
        self.model = Localizer(state['width'],severity=has_severity,normalization=state.get('normalization','none'),input_features=state.get('input_features','rgb')).to(device)
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.device, self.tile, self.overlap = device, tile, overlap
        self.tile_batch_size = tile_batch_size
        self.provenance = predictor_provenance(checkpoint,calibration,tile,overlap)
        self.calls = 0
        self.calibration = dict(temperatures=[1.,1.,1.], severity_temperatures=[1.,1.,1.], thresholds=[.5,.5,.5],
                                confidence_threshold=.2, normal_threshold=.3)
        if calibration is not None:
            values = json.loads(Path(calibration).read_text()) if isinstance(calibration,(str,Path)) else calibration
            self.calibration.update(values)
            expected = values.get('checkpoint_sha256')
            if expected and expected != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                raise ValueError('Calibration belongs to another localization checkpoint')
        temperatures = np.asarray(self.calibration['temperatures'],dtype=np.float32)
        if temperatures.shape!=(3,) or not np.isfinite(temperatures).all() or (temperatures<=0).any():
            raise ValueError('Expected three positive calibration temperatures')
        for key in ['severity_temperatures', 'thresholds']:
            values = np.asarray(self.calibration[key])
            if values.shape!=(3,) or not np.isfinite(values).all() or (values<=0).any():
                raise ValueError(f'Invalid calibration {key}')

    def __call__(self, image):
        return self.state(image)['probabilities']

    @torch.inference_mode()
    def state(self, image):
        probabilities, distributions = tiled_probabilities(
            self.model,image,self.device,self.tile,self.overlap,self.tile_batch_size)
        self.calls += 1
        probabilities = temperature_probabilities(probabilities, self.calibration['temperatures']).astype(np.float32)
        if self.model.has_severity:
            distributions = temperature_severity(distributions,self.calibration['severity_temperatures']).astype(np.float32)
        # Conditional severity is independent of presence probability; ignore it for absent/uncertain classes.
        severity = (distributions*np.linspace(0,1,5,dtype=np.float32)).sum(axis=3)
        state = dict(probabilities=probabilities,severity=severity,
                     severity_distribution=distributions,severity_available=self.model.has_severity,
                     thresholds=np.asarray(self.calibration['thresholds'],dtype=np.float32))
        state.update(confidence_state(probabilities,self.calibration['confidence_threshold'],self.calibration['normal_threshold']))
        return state
