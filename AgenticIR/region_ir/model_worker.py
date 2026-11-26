"""Direct degraded-RGB inference in the corresponding isolated model environment."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['swinir15', 'swinir50', 'restormer', 'fbcnn'], required=True)
    parser.add_argument('--source', required=True, help='Checked-out model repository')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--checkpoint-format', choices=['state_dict', 'torchscript'], default='state_dict')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--metadata')
    args = parser.parse_args()
    started = perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    sys.path.insert(0, str(Path(args.source).resolve()))
    if args.model.startswith('swinir'):
        from models.network_swinir import SwinIR
        model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8, img_range=1.,
                       depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
                       upsampler='', resi_connection='1conv')
        key, window = 'params', 8
    elif args.model == 'restormer':
        # Load the official inference architecture without importing BasicSR's
        # training registry and its obsolete torchvision dependencies.
        architecture = Path(args.source).resolve() / 'basicsr/models/archs/restormer_arch.py'
        spec = importlib.util.spec_from_file_location('official_restormer_arch', architecture)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Restormer = module.Restormer
        model = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8],
                          num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
                          bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
        key, window = 'params', 8
    else:
        from models.network_fbcnn import FBCNN
        model = FBCNN(in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4,
                      act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
        key, window = None, 8
    if args.checkpoint_format == 'torchscript':
        checkpoint = torch.jit.load(args.weights, map_location='cpu').state_dict()
    else:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    state = checkpoint[key] if key and key in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    with Image.open(args.input) as source:
        array = np.asarray(source.convert('RGB'), dtype=np.float32)/255
    tensor = torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).to(args.device)
    h, w = tensor.shape[-2:]
    # Outer regional executor normally pads; replicate supports tiny direct inputs.
    tensor = F.pad(tensor, (0, -w % window, 0, -h % window), mode='replicate')
    with torch.inference_mode():
        result = model(tensor)
        if args.model == 'fbcnn':
            result = result[0]  # Blind quality-factor estimation; no test q supplied.
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite model output before clipping')
    result = result[0, :, :h, :w].clamp(0,1).permute(1,2,0).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Non-finite model output')
    Image.fromarray(np.rint(result*255).astype('uint8')).save(args.output)
    if args.metadata:
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        metadata = dict(model=args.model, checkpoint_format=args.checkpoint_format, device=args.device, torch_version=str(torch.__version__),
                        cuda_version=torch.version.cuda, seconds=perf_counter()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
                        weights_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                        input_shape=list(array.shape), cold_start=True)
        Path(args.metadata).write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
