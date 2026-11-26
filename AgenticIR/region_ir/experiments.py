"""One shared runner for the proposal's eight groups and explicit ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback
import numpy as np
from .arrays import load_sample_array, sample_array_path
from .core import CommandTool,Region,build_regions,load_image,run,save_image
from .feedback import Musiq,JointFeedback,load_feedback
from .policy import Planner,Ranker
from .provenance import predictor_provenance
from .evaluate import metrics

METHODS=('input','whole_fixed','agentic_local','grid_rule','region_rule','whole_learned','region_learned','oracle_learned')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def file_dependencies(config):
    files=[config['tools']]
    for key in ['musiq_weights','feedback_config','data_manifest']:
        if config.get(key): files.append(config[key])
    for seed in config['seeds']:
        files.extend([seed['localizer'],seed['ranker']])
        if seed.get('calibration'): files.append(seed['calibration'])
    for method in config.get('methods',[]):
        if isinstance(method,dict): files.extend(method.get('rankers',{}).values())
    for tool in json.loads(Path(config['tools']).read_text()).values():
        cmd=tool['command']
        for flag in ['--weights']:
            if flag in cmd:files.append(cmd[cmd.index(flag)+1])
    return sorted(set(files))


def freeze(config):
    if config.get("split","test")!="test":raise ValueError("Formal freeze requires test split")
    if len(config['seeds'])<3 or len({s['seed'] for s in config['seeds']})!=len(config['seeds']):
        raise ValueError('Formal protocol requires at least three distinct training seeds')
    if not config.get('feedback_config') or not config.get('musiq_weights'):
        raise ValueError('Formal protocol requires calibrated feedback and fixed MUSIQ weights')
    if config.get('no_feedback'): raise ValueError('Use method-level ablation for no-feedback')
    required_tools=set(json.loads(Path(config['tools']).read_text()))
    for entry in config['seeds']:
        bound=Ranker(entry['ranker']); bound.require_coverage(required_tools)
        bound.require_perception(predictor_provenance(entry['localizer'],entry.get('calibration')),config.get('options',{}).get('threshold',.5))
    for method in config.get('methods', []):
        if isinstance(method, dict) and method.get('rankers'):
            for entry in config['seeds']:
                bound=Ranker(method['rankers'][str(entry['seed'])]); bound.require_coverage(required_tools)
                threshold=method.get('options',{}).get('threshold',config.get('options',{}).get('threshold',.5))
                bound.require_perception(predictor_provenance(entry['localizer'],entry.get('calibration')),threshold)
    for kind in ['localizer','ranker']:
        if len({sha256(s[kind]) for s in config['seeds']})!=len(config['seeds']):
            raise ValueError('Each training seed needs its own trained checkpoint')
    rows=[json.loads(line) for line in Path(config['data_manifest']).read_text().splitlines()]
    tests=[r for r in rows if r['split']=='test']
    groups={}
    for row in tests:groups.setdefault(row['source_id'],[]).append(row['scenario'])
    if len(groups)!=100 or len(tests)!=600 or any(len(set(v))!=6 for v in groups.values()):
        raise ValueError('Formal core protocol requires 100 test sources x six distinct scenes')
    if {str(k).zfill(4) for k in groups}!={str(i).zfill(4) for i in range(801,901)}:
        raise ValueError('Formal source ids must be 0801..0900')
    paths=file_dependencies(config)
    root=Path(config['data_manifest']).parent
    for row in tests:
        folder=root/row['folder']
        paths.extend(str(folder/name) for name in ['input.png','clean.png'])
        paths.append(str(sample_array_path(folder, 'masks')))
    code=sorted(Path(__file__).parent.glob('*.py'))
    return dict(config=config,artifacts={str(Path(f).resolve()):sha256(f) for f in paths},
                code={p.name:sha256(p) for p in code},frozen=True)


def verify_protocol(protocol):
    for path,digest in protocol['artifacts'].items():
        if sha256(path)!=digest:raise ValueError(f'Frozen artifact changed: {path}')
    for filename,digest in protocol['code'].items():
        if sha256(Path(__file__).parent/filename)!=digest:raise ValueError(f'Frozen code changed: {filename}')


def regions_for(method,image,probabilities,masks=None,threshold=.5):
    h,w=image.shape[:2]
    if method in ('whole_fixed','whole_learned','agentic_local'):
        return [Region(0,np.ones((h,w),np.float32),tuple(probabilities.mean((0,1))))]
    if method=='input': return []
    if method=='grid_rule':
        regions=[]
        for y0,y1 in [(0,h//2),(h//2,h)]:
            for x0,x1 in [(0,w//2),(w//2,w)]:
                mask=np.zeros((h,w),np.float32);mask[y0:y1,x0:x1]=1
                if mask.any():regions.append(Region(len(regions),mask,tuple(probabilities[mask>0].mean(0))))
        return regions
    if method=='oracle_learned':
        if masks is None:raise ValueError('Oracle diagnostic needs masks')
        return build_regions(masks,threshold=threshold)
    if method in ('region_rule','region_learned'):return build_regions(probabilities,threshold=threshold)
    raise ValueError(f'Unknown method: {method}')


class FixedPlanner:
    def __init__(self,order,tools):
        if len(set(order))!=len(order) or any(t not in tools for t in order):raise ValueError('Invalid fixed order')
        self.order=order
    def __call__(self,image,region,history,remaining,failed):
        used={r['tool_id'] for r in history if r['region_id']==region.id}
        return next((t for t in self.order if t not in used and t not in failed),'stop')


def run_case(image,method,output,predictor,tools,feedback,ranker=None,masks=None,options=None):
    opts=options or {};base=method
    if base in ("whole_learned","region_learned","oracle_learned") and ranker is None:
        raise ValueError("Learned groups require a trained ranker")
    probs=np.zeros_like(image) if method=="input" else predictor(image)
    regions=regions_for(base,image,probs,masks,opts.get('threshold',.5))
    planner=Planner(predictor,tools,ranker if base in ('whole_learned','region_learned','oracle_learned') else None,
                    no_history=opts.get('no_history',False),no_cost=opts.get('no_cost',False),
                    force_initial_tool=opts.get('force_initial_tool',False),max_steps=opts.get('max_steps',2),
                    min_confidence=opts.get('min_confidence',.5),threshold=opts.get('threshold',.5))
    if base=='whole_fixed' or opts.get('fixed_order'):
        planner=FixedPlanner(opts.get('fixed_order',['restormer','swinir50','fbcnn']),tools)
    if opts.get('no_feedback'): feedback=lambda before,after,region:(True,'feedback ablation')
    result,history=run(image,regions,tools,planner,feedback,output,budget=opts.get('budget',12),
                       max_steps=opts.get('max_steps',2),context=opts.get('context',32),soft=not opts.get('hard_fusion',False),
                       time_budget=opts.get('time_budget'))
    np.save(Path(output)/'probabilities.npy',probs)
    return result,history,None if method=="input" else probs


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output')
    p.add_argument('--freeze',action='store_true');p.add_argument('--development',action='store_true')
    args=p.parse_args();document=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.freeze:
        with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(freeze(document),stream,indent=2)
        return
    if document.get('frozen'):
        verify_protocol(document);config=document['config']
    elif args.development:config=document
    else:raise ValueError('Freeze the protocol first, or explicitly use --development')
    from .localizer import Predictor
    from .evaluate import LPIPSScorer
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(document,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(platform=platform.platform(),python=platform.python_version())))
    root=Path(config['data_manifest']).parent
    records=[json.loads(s) for s in Path(config['data_manifest']).read_text().splitlines()]
    selected=[r for r in records if r['split']==config.get('split','test')]
    if config.get('limit') is not None:
        if not args.development:raise ValueError('Formal runs cannot truncate samples')
        selected=selected[:config['limit']]
    tool_config=json.loads(Path(config['tools']).read_text())
    for entry in config['seeds']:
        predictor=Predictor(entry['localizer'],config.get('device','cuda'),calibration=entry.get('calibration'))
        ranker=Ranker(entry['ranker']); ranker.require_coverage()
        scorer=Musiq(config['musiq_weights'],config.get('quality_device','cpu'))
        feedback=JointFeedback(predictor,scorer,load_feedback(config.get('feedback_config')))
        lpips=LPIPSScorer(device=config.get('quality_device','cpu')) if config.get('lpips',True) else None
        (out/f'quality_{entry["seed"]}.json').write_text(json.dumps(scorer.metadata,indent=2))
        for record in selected:
            folder=root/record['folder'];image=load_image(folder/'input.png')
            # Ground truth loaded here for diagnostics/evaluation, never passed to automatic planner.
            clean=load_image(folder/'clean.png');masks=load_sample_array(folder, 'masks')
            for spec in config.get('methods',list(METHODS)):
                spec={'name':spec,'base':spec} if isinstance(spec,str) else spec
                method=spec['base'];name=spec['name']
                if method not in METHODS or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):raise ValueError('Invalid method name')
                options={**config.get('options',{}),**spec.get('options',{})}
                case=out/str(entry['seed'])/name/f'{record["source_id"]}_{record["scenario"]}'
                row=dict(source_id=record['source_id'],scenario=record['scenario'],method=name,seed=entry['seed'],
                         dataset=record.get('dataset','core'),status='ok',diagnostic=method=='oracle_learned',run_dir=str(case))
                case_ranker=Ranker(spec['rankers'][str(entry['seed'])]) if spec.get('rankers') else ranker
                case_ranker.require_coverage()
                tools={name:CommandTool(**kwargs) for name,kwargs in tool_config.items()}
                start=perf_counter();before_calls=scorer.calls;prediction_calls=predictor.calls
                try:
                    if method=='agentic_local':
                        from .agentic_adapter import run_adapted
                        result,history=run_adapted(folder/'input.png',case,tools,predictor,feedback,
                                                  options.get('budget',12),options.get('time_budget'))
                        probabilities=predictor(image)
                    else:
                        result,history,probabilities=run_case(image,method,case,predictor,tools,feedback,case_ranker,
                                                            masks if method=='oracle_learned' else None,options)
                    summary=json.loads((case/'summary.json').read_text())
                    row['calls']=summary['calls'];row['termination']=summary['termination']
                    if summary.get('failed_calls',0):row['status']='failed';row['error']='one or more tool/evaluator failures; fallback retained'
                except Exception as error:
                    row.update(status='failed',error=f'{type(error).__name__}: {error}')
                    case.mkdir(parents=True,exist_ok=True)
                    (case/'failure.txt').write_text(traceback.format_exc())
                    result=image;probabilities=None
                    # A pre-existing output is the last completed state, otherwise use input.
                    if (case/'output.png').exists():result=load_image(case/'output.png')
                    else:save_image(case/'output.png',result)
                    traces=case/'trace.jsonl'
                    history=[json.loads(s) for s in traces.read_text().splitlines()] if traces.exists() else []
                    row['calls']=max([r.get('calls',0) for r in history]+[0])
                row['seconds']=perf_counter()-start
                row['quality_calls']=scorer.calls-before_calls;row['predictor_calls']=predictor.calls-prediction_calls
                row['peak_allocated_bytes']=max([((r.get('tool_metadata') or {}).get('peak_allocated_bytes') or 0) for r in history]+[0]) or None
                try:row['metrics']=metrics(clean,image,result,masks,predictions=probabilities,lpips=lpips)
                except Exception as error:row.update(status='failed',metric_error=str(error),metrics={})
                with (out/'results.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(entry['seed'],name,record['source_id'],record['scenario'],row['status'],flush=True)


if __name__=='__main__':main()
