"""Combine accepted actual-motion candidates with unchanged first-batch results."""
import argparse,json
from pathlib import Path
import lance
from evaluate_full_repairs import evaluate

p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--first-bundle',type=Path,required=True);p.add_argument('--work-output',type=Path,required=True);p.add_argument('--second-pass-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ds=lance.dataset(a.dataset,version=5);metadata=ds.to_table(columns=['index','trajectory_metadata']).to_pylist();catalog=json.load(open(a.first_bundle/'catalog.json'));registry={i:{'row':i,'uuid':r['index']['uuid'],'gesture':r['index']['gesture'],'status':'unprocessed','attempts':[]} for i,r in enumerate(metadata)}
for c in catalog['samples']:
 i=c['source_row'];assert registry[i]['uuid']==c['source_uuid'];registry[i].update(status='accepted',run=str(a.first_bundle/c['recording']),patch=str(a.first_bundle/c['patch']),validation={'status':'accepted','prior_verified':True,'metrics':c['metrics']})
roots=[a.second_pass_root,*[a.work_output/name for name in ['bowls_pass1','containers_pass1','bowls_pass2','containers_pass2','bowls_pass3','containers_pass3','containers_pass4','containers_pass5','containers_pass6']]]
paths=[]
for root in roots:
 if root.exists():paths.extend(sorted(p.parent for p in root.glob('row*/result.json')))
paths.extend(a.work_output/f'level_row{i}' for i in [51,55] if (a.work_output/f'level_row{i}/result.json').exists())
for path in paths:
 manifest=json.load(open(path/'manifest.json'));i=int(manifest['source']['row']);assert registry[i]['uuid']==manifest['source']['uuid']
 v=json.load(open(path/'validation.json')) if (path/'validation.json').exists() else evaluate(path)
 registry[i]['attempts'].append({'run':str(path),'status':v['status'],'failed_gates':[k for k,value in v['gates'].items() if not value]})
 if v['status']=='accepted':registry[i].update(status='accepted',run=str(path),patch=None,validation={k:x for k,x in v.items() if k!='contact_samples'})
 elif registry[i]['status']!='accepted':registry[i]['status']='needs_fix'
result={'source_dataset':str(a.dataset),'source_version':5,'total_rows':len(registry),'accepted_count':sum(v['status']=='accepted' for v in registry.values()),'rows':list(registry.values())}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print('ACCEPTED',result['accepted_count'],'/',len(registry));print('REMAINING',[(v['row'],v['status'],v['attempts'][-1]['failed_gates'] if v['attempts'] else []) for v in registry.values() if v['status']!='accepted'])
