"""Identity of perception artifacts used to generate policy supervision."""
import hashlib
import json
from pathlib import Path
import math

def file_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def canonical_json(value):
    def encode(item):
        if hasattr(item,'tolist'): return item.tolist()
        raise TypeError('Unsupported provenance value: ' + type(item).__name__)
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=encode)

def predictor_provenance(checkpoint,calibration=None,tile=256,overlap=32):
    if calibration is None:
        calibration_hash=None
    elif isinstance(calibration,(str,Path)):
        calibration_hash=file_sha256(calibration)
    else:
        calibration_hash=hashlib.sha256(canonical_json(calibration).encode()).hexdigest()
    return dict(schema=1,localizer_sha256=file_sha256(checkpoint),calibration_sha256=calibration_hash,
                tile=int(tile),overlap=int(overlap))

def perception_payload(provenance,threshold):
    if not math.isfinite(threshold) or not 0<threshold<=1:raise ValueError('Invalid policy threshold')
    return dict(predictor=provenance,threshold=float(threshold))

def payload_sha256(payload):
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
