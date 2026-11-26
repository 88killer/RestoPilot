"""Run the actual upstream IRAgent tree/rollback with explicit local substitutions.

Replaces GPT scheduling with deterministic severity ordering and DepictQA with the
trained five-level predictor + shared joint feedback. This is NOT original AgenticIR.
"""
import json
import logging
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from .core import Region,load_image,save_image,execute_region


class BudgetExhausted(RuntimeError): pass


def run_adapted(input_path,output_dir,tools,predictor,feedback,budget=12,time_budget=None):
    # Lazy import keeps upstream optional libraries out of the regional environment.
    from pipeline.iragent import IRAgent
    started=perf_counter(); events=[]
    mapping={'noise':'denoising','blur':'motion deblurring','jpeg':'jpeg compression artifact removal'}
    tool_groups={'denoising':['swinir15','swinir50'],'motion deblurring':['restormer'],
                 'jpeg compression artifact removal':['fbcnn']}
    class WrappedTool:
        def __init__(self,name): self.tool_name=name
        def __call__(self,input_dir,output_dir,silent=True):
            if len(events)>=budget or (time_budget is not None and perf_counter()-started>=time_budget):
                raise BudgetExhausted('Restoration budget exhausted')
            current=load_image(next(Path(input_dir).glob('*.png')))
            event=dict(tool_id=self.tool_name,accepted=False,error=None);events.append(event)
            tick=perf_counter()
            try:
                region=Region(0,np.ones(current.shape[:2],np.float32),(1,1,1))
                candidate=execute_region(current,region,tools[self.tool_name])
                accepted,reason=feedback(current,candidate,region)
                event.update(accepted=bool(accepted),reason=reason,feedback=getattr(feedback,'last',None))
                save_image(Path(output_dir)/'candidate.png',candidate)
                # Upstream expects only output.png in this directory.
                (Path(output_dir)/'candidate.png').replace(Path(output_dir).parent/'candidate.png')
                save_image(Path(output_dir)/'output.png',candidate if accepted else current)
            except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as error:
                event['error']=str(error);save_image(Path(output_dir)/'output.png',current)
            finally:
                event['seconds']=perf_counter()-tick
                event['calls']=len(events)
                event['tool_metadata']=getattr(tools[self.tool_name],'last_metadata',None)
                with (out/'trace.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(event)+'\n')
    class Adapted(IRAgent):
        def _create_components(self,*args):
            self.workflow_logger=logging.getLogger('region_ir.agentic_adapter')
            self.qa_logger=self.workflow_logger
            self.executor=SimpleNamespace(toolbox_router={task:[WrappedTool(t) for t in names if t in tools]
                                                         for task,names in tool_groups.items()})
            if any(not v for v in self.executor.toolbox_router.values()):
                raise ValueError('AgenticIR adapter requires noise, blur and JPEG tool groups')
        def _set_constants(self):
            self.degra_subtask_dict=mapping
            self.subtask_degra_dict={v:k for k,v in mapping.items()}
            self.degradations=set(mapping);self.subtasks=set(mapping.values())
            self.levels=['very low','low','medium','high','very high']
        def evaluate_degradation(self):
            state=predictor.state(load_image(self.cur_node['img_path']))
            if not state.get('severity_available',False):
                raise ValueError('AgenticIR local adapter requires trained five-level severity head')
            p=state['probabilities'];severity=state['severity']
            self.local_severity={}
            for i,kind in enumerate(mapping):
                support=p[:,:,i]>=.5
                level=int(np.clip(np.rint(severity[:,:,i][support].mean()*4),0,4)) if support.any() else 0
                self.local_severity[kind]=level
            return [(kind,self.levels[level]) for kind,level in self.local_severity.items()]
        def extract_agenda(self,evaluation):
            return [mapping[kind] for kind,level in evaluation if self.levels.index(level)>=2]
        def schedule(self,agenda,ps=''):
            return sorted(agenda,key=lambda task:(-self.local_severity[self.subtask_degra_dict[task]],task))
        def evaluate_tool_result(self,img_path,degradation):
            state=predictor.state(load_image(img_path));index=list(mapping).index(degradation)
            mask=state['probabilities'][:,:,index]>=.5
            level=int(np.clip(np.rint(state['severity'][:,:,index][mask].mean()*4),0,4)) if mask.any() else 0
            return self.levels[level]
        def compare_quality(self,img1,img2):
            before,after=load_image(img1),load_image(img2)
            accepted,_=feedback(before,after,Region(0,np.ones(before.shape[:2],np.float32),(1,1,1)))
            return 'latter' if accepted else 'former'
        def _prepare_for_subtask(self,subtask):
            folder=Path(self.cur_node['img_path']).parents[1]/f'subtask-{subtask}'
            folder.mkdir()
            return folder,self.subtask_degra_dict[subtask],self.executor.toolbox_router[subtask]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    agent=Adapted(Path(input_path),out/'upstream',with_retrieval=False,silent=True)
    termination='completed'
    try: agent.run()
    except BudgetExhausted:
        termination='budget';agent._record_res()
    final=load_image(agent.res_path)
    save_image(out/'output.png',final)
    summary=dict(calls=len(events),history=events,termination=termination,seconds=perf_counter()-started,
                 failed_calls=sum(e['error'] is not None for e in events),
                 adaptation='upstream IRAgent traversal/rollback; local severity scheduling and shared feedback')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return final,events
