import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .core import CommandTool, build_regions, load_image, run
from .arrays import load_array
from .policy import Planner, Ranker, ResidualFeedback
from .feedback import Musiq, JointFeedback, load_feedback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input'); p.add_argument('output')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--localizer',help='Trained localization checkpoint')
    source.add_argument('--probabilities',help='Diagnostic HxWx3 mask array; requires --diagnostic')
    p.add_argument('--diagnostic',action='store_true'); p.add_argument('--ranker')
    p.add_argument('--tools',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--budget',type=int,default=12); p.add_argument('--max-steps',type=int,choices=[1,2,3],default=2)
    p.add_argument('--min-confidence',type=float,default=.5,help='Policy confidence gate; choose on development validation, not test images')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--min-gain',type=float,default=.005); p.add_argument('--max-color-shift',type=float,default=.1)
    p.add_argument('--calibration')
    p.add_argument('--musiq-weights'); p.add_argument('--feedback-config')
    p.add_argument('--quality-device', default='cpu')
    p.add_argument('--context',type=int,choices=[16,32,64],default=32)
    p.add_argument('--hard-fusion',action='store_true')
    p.add_argument('--no-history',action='store_true'); p.add_argument('--no-cost',action='store_true')
    p.add_argument('--force-initial-tool',action='store_true')
    p.add_argument('--no-feedback',action='store_true',help='Explicit feedback ablation')
    args=p.parse_args()
    if args.probabilities and not args.diagnostic:
        p.error('External masks require --diagnostic; automatic inference uses --localizer')
    if args.probabilities and args.ranker:
        p.error('Learned policy requires current-image localization; use --localizer')
    start=perf_counter()
    config=json.loads(Path(args.tools).read_text(encoding='utf-8'))
    tools={name:CommandTool(**options) for name,options in config.items()}
    image=load_image(args.input)
    if args.localizer:
        from .localizer import Predictor
        predictor=Predictor(args.localizer,args.device,calibration=args.calibration)
    else:
        probabilities=load_array(args.probabilities)
        if probabilities.shape!=image.shape: raise ValueError('Probability/image size mismatch')
        predictor=lambda current: probabilities
    probabilities=predictor(image)
    regions=build_regions(probabilities,threshold=args.threshold)
    ranker=Ranker(args.ranker) if args.ranker else None
    if ranker: ranker.require_coverage()
    planner=Planner(predictor,tools,ranker,
                    no_history=args.no_history,no_cost=args.no_cost,force_initial_tool=args.force_initial_tool,
                    max_steps=args.max_steps,threshold=args.threshold,min_confidence=args.min_confidence)
    feedback=ResidualFeedback(predictor,args.min_gain,args.max_color_shift)
    scorer=None
    if args.musiq_weights:
        scorer=Musiq(args.musiq_weights,args.quality_device)
        feedback=JointFeedback(predictor,scorer,load_feedback(args.feedback_config))
    if args.no_feedback or args.probabilities:
        feedback=lambda before,after,region:(True,'integrity-only diagnostic/ablation')
    run(image,regions,tools,planner,feedback,args.output,args.budget,args.max_steps,context=args.context,soft=not args.hard_fusion)
    np.save(Path(args.output)/'probabilities.npy',probabilities)
    checkpoints={name:hashlib.sha256(Path(value).read_bytes()).hexdigest()
                 for name,value in [('localizer',args.localizer),('ranker',args.ranker)] if value}
    mode=dict(diagnostic=args.diagnostic,policy='learned-action-utility' if args.ranker else 'rule',
              feedback='integrity-only' if args.no_feedback or args.probabilities else ('joint-musiq-residual-color' if scorer else 'predicted-residual-and-color'),
              predictor_calls=getattr(predictor,'calls',0),seconds=perf_counter()-start,
              checkpoints=checkpoints,arguments=vars(args),tools=config,
              supported_actions=list(ranker.supported_actions) if ranker else None,
              quality_calls=scorer.calls if scorer else 0,quality_metadata=scorer.metadata if scorer else None)
    Path(args.output,'mode.json').write_text(json.dumps(mode,indent=2),encoding='utf-8')


if __name__=='__main__': main()
