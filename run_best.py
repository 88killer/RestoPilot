from pathlib import Path
from hashlib import sha256
from dataclasses import replace
from zipfile import ZipFile
import json,sys,subprocess,shutil
import numpy as np
BUNDLE=Path(__file__).resolve().parent

def dump(path, value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load(path):
 return json.loads(Path(path).read_text(encoding='utf-8'))

def resolve(value, tools):
 if isinstance(value,dict): return {k:resolve(v,tools) for k,v in value.items()}
 if isinstance(value,list): return [resolve(v,tools) for v in value]
 if value=='PYTHON_EXECUTABLE': return sys.executable
 if isinstance(value,str) and value.startswith('BUNDLE:'):
  p=(BUNDLE/value[7:]).resolve()
  if not p.is_relative_to(BUNDLE) or not p.exists(): raise FileNotFoundError(p)
  return str(p)
 return value

ARCHIVE='https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
VARIANTS={'residual':(True,False)}
CASES=None

def main(inputs):
 import torch
 assert torch.cuda.is_available() and torch.__version__.startswith('2.7.1')
 out=BUNDLE/'输出结果';out.mkdir(parents=True,exist_ok=True)
 if (out/'ablation_cases.jsonl').exists(): raise FileExistsError('输出结果已存在，请先移走或重命名输出结果目录')
 sys.path[:0]=[str(BUNDLE/'AgenticIR'),str(BUNDLE)]
 from region_ir.core import CommandTool,build_regions,load_image,run
 from region_ir.arrays import load_sample_array
 from region_ir.data import write_source
 from region_ir.evaluate import LPIPSScorer,metrics
 from region_ir.feedback import JointFeedback,Musiq,load_feedback,decide
 from region_ir.localizer import Predictor
 from region_ir import policy
 from region_ir.policy import Ranker,Planner
 tools_config=resolve(load(BUNDLE/'cloud_tools.json'),None); dump(out/'tools.json',tools_config)
 cfg=resolve(load(BUNDLE/'config.json'),None);opt=cfg['options'];seed=cfg['seed']
 ck=inputs/'calibrated_ranker.npz';train=load(inputs/'calibrated_training.json');assert sha256(ck.read_bytes()).hexdigest()==train['checkpoint_sha256']
 class NetRanker(Ranker):
  clamp=False
  def score(self,state,action,no_cost=False,*,inference_cost_weight=None):
   assert not no_cost and inference_cost_weight==.025
   z=super().predict(state,action)['gain'];score=float(np.sign(z)*z*z)
   return min(score,0.) if self.clamp and action=='noop' else score
 ranker=NetRanker(ck);predictor=Predictor(seed['localizer'],cfg['device'],calibration=seed['calibration'])
 tools={k:CommandTool(**v) for k,v in load(out/'tools.json').items()};ranker.require_coverage(tools);ranker.require_perception(predictor.provenance,opt['threshold'])
 base=load_feedback(cfg['feedback_config']);loose=replace(base,max_residual_increase=.006)
 feedback=JointFeedback(predictor,Musiq(cfg['musiq_weights'],cfg['quality_device']),base);scorer=LPIPSScorer(cfg['quality_device'])
 data=BUNDLE/'生成数据'
 records=list(map(json.loads,(data.parent/'实例清单.jsonl').read_text().splitlines())); assert len(records)>0
 assert len({(r['source_id'],r['scenario']) for r in records})==len(records)
 class Guard:
  def __init__(self,variant,flags):self.events=[];self.variant=variant;self.flags=flags
  @property
  def last(self):return feedback.last
  def __call__(self,before,after,region):
   _,reason=feedback(before,after,region);v=dict(feedback.last);old=bool(decide(v,base));relaxed=bool(decide(v,loose));new=old or (relaxed and v['normal_change']<=.001)
   extra=False
   if self.flags[region.id]:
    safe=v['normal_change']<=.001 and v['color_shift']<=base.max_color_shift
    extra=safe and v['quality_gain']>=1.0 and v['residual_gain']>=-.04
   self.events.append(dict(region_id=region.id,old=old,relaxed=relaxed,protected=new,extra=extra,feedback=v));return bool(new or extra),reason
 temp=out/'过程记录';temp.mkdir(exist_ok=True)
 with (out/'ablation_cases.jsonl').open('w') as stream:
  for ix,rec in enumerate(records,1):
   sid=rec['source_id'];scenario=rec['scenario'];image=load_image(data/rec['input_path']);clean=load_image(data/rec['clean_path']);masks=load_sample_array((data/rec['masks_path']).parent,'masks')
   initial=predictor.state(image);regions=build_regions(initial['probabilities'],threshold=opt['threshold']); flags={r.id:int(np.sum(initial['probabilities'][r.mask>0].mean(axis=0)>=.35))>=2 for r in regions};results={};names=list(VARIANTS) if ix%2 else list(reversed(VARIANTS))
   for name in names:
    clamp,relax=VARIANTS[name]; min_confidence=.45;ranker.clamp=clamp;guard=Guard(name,flags);events=[]
    planner=Planner(predictor,tools,ranker,max_steps=opt['max_steps'],min_confidence=min_confidence,threshold=opt['threshold'],inference_cost_weight=.025)
    def plan(current,region,history,remaining,failed):
     local=[r for r in history if r['region_id']==region.id];after=any(r.get('accepted') and r.get('tool_id') in tools for r in local)
     orig=policy.candidate_tools
     original_score=ranker.score
     def score_override(state,action,no_cost=False,*,inference_cost_weight=None):
      value=original_score(state,action,no_cost,inference_cost_weight=inference_cost_weight)
      if flags[region.id] and action in tools and value>=-.05: return value+.050000000001
      return value
     ranker.score=score_override
     def select(image,region,state,tools,threshold=.5):
      before=orig(image,region,state,tools,threshold);after_tools=orig(image,region,state,tools,.25) if relax and after else before
      events.append(dict(region_id=region.id,step=len(local),after_accepted=after,threshold=.25 if relax and after else threshold,standard=before,selected=after_tools));return after_tools
     policy.candidate_tools=select
     try:return planner(current,region,history,remaining,failed)
     finally:policy.candidate_tools=orig;ranker.score=original_score
    directory=temp/'runs'/sid/scenario/name;result,traces=run(image,regions,tools,plan,guard,directory,budget=opt['budget'],max_steps=opt['max_steps'],context=opt['context'],time_budget=300)
    summary=load(directory/'summary.json');q=metrics(clean,image,result,masks,lpips=scorer);accepted={}; step_metrics=[]
    for t in traces:
     if t.get('tool_id') in tools:
      stem=f"r{t['region_id']}_s{t['step']}"; before=directory/f'{stem}_before.png'; candidate=directory/f'{stem}_candidate.png'
      if not before.exists() or not candidate.exists(): raise FileNotFoundError(stem)
      q0=metrics(clean,image,load_image(before),masks,lpips=scorer); q1=metrics(clean,image,load_image(candidate),masks,lpips=scorer)
      step_metrics.append(dict(region_id=t['region_id'],step=t['step'],tool=t['tool_id'],accepted=t['accepted'],before={k:q0[k] for k in ('psnr','ssim','lpips')},after={k:q1[k] for k in ('psnr','ssim','lpips')}))
    for t in traces:
     if t.get('accepted') and t.get('tool_id') in tools:accepted[t['region_id']]=accepted.get(t['region_id'],0)+1
    results[name]=dict(step_metrics=step_metrics,quality={k:q[k] for k in ('psnr','ssim','lpips')},calls=summary['calls'],seconds=summary['seconds'],failures=summary['failed_calls'],termination=summary['termination'],two_step_retained_regions=sum(n>=2 for n in accepted.values()),accepted_tool_steps=sum(accepted.values()),feedback_events=guard.events,candidate_events=events,traces=traces)
   previous=None; match_previous=None
   stream.write(json.dumps(dict(source_id=sid,scenario=rec['scenario'],manifest=rec,previous=previous,match_previous=match_previous,composite_flags=flags,variants=results),allow_nan=False)+'\n');stream.flush()
   print('CLIC70',ix,'/',len(records),sid,scenario,{k:(v['calls'],round(v['quality']['psnr'],3) if isinstance(v['quality']['psnr'],(float,int)) else v['quality']['psnr'],v['two_step_retained_regions']) for k,v in results.items()},flush=True)
 dump(out/'ablation_config.json',dict(complete=True,cases=len(records),source_ids=[r['source_id'] for r in records],composite_probability_threshold=.35,utility_threshold=-.05,variants=VARIANTS,seed=43,ranker_sha256=train['checkpoint_sha256'],first_candidate_threshold=opt['threshold'],followup_candidate_threshold=opt['threshold'],confidence_thresholds=[.45],cases_selected='local_images',protection='old OR (residual tolerance .006 AND normal change <= .001)',version='residual_best'))
if __name__=='__main__':main(Path(sys.argv[1]))
