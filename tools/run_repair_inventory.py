"""Run candidate patches sequentially; failed jobs are explicit ledger entries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('--inventory',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--replay-tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--source-root',type=Path,required=True);p.add_argument('--rows',default='')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
selected={int(x) for x in a.rows.split(',') if x} if a.rows else None
items=json.loads(a.inventory.read_text());env=os.environ.copy();env['PYTHONPATH']=str(a.source_root);env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';ledger=[]
for item in items:
 row=int(item['row'])
 if selected is not None and row not in selected:continue
 out=a.output/f'row{row}';started=time.time()
 if (out/'result.json').exists():
  result=json.loads((out/'result.json').read_text());ledger.append({'row':row,'status':'existing_complete' if result['complete'] else 'existing_incomplete','output':str(out),'result':result});continue
 if out.exists():
  ledger.append({'row':row,'status':'blocked_partial_output','output':str(out)});continue
 with (a.output/f'row{row}.log').open('w') as log:
  command=[sys.executable,str(a.replay_tool),'--dataset',str(a.dataset),'--patch',item['patch'],'--post-padding',str(item.get('post_padding',500)),'--output',str(out)]
  print('START',row,flush=True);proc=subprocess.run(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
 result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
 entry={'row':row,'status':'completed_run' if proc.returncode==0 and result and result['complete'] else 'failed_run','exit_code':proc.returncode,'elapsed_seconds':time.time()-started,'output':str(out),'result':result};ledger.append(entry)
 (a.output/'run_ledger.json').write_text(json.dumps(ledger,indent=2));print('END',row,entry['status'],result,flush=True)
(a.output/'run_ledger.json').write_text(json.dumps(ledger,indent=2))
