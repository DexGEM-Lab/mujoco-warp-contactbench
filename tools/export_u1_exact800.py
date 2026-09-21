"""Explicit immutable hidden Lance export and source-bound full readback validation."""
from pathlib import Path
import argparse
import json
import shutil
from sim.manorl.u1_campaign import file_sha, write_json
from sim.manorl.u1_export import collect, make_row, schema, require, CONTRACT, SETTING


def validate(output, registry, plan, attempts):
    import lance
    import pyarrow as pa
    r,p,records,audit=collect(registry,plan,attempts)
    manifest=json.loads((output/'manifest.json').read_text())
    require(manifest['registry_digest']==r['digest'] and manifest['plan_digest']==p['digest'] and manifest['setting_sha256']==SETTING, 'manifest identities')
    hashes=json.loads((output/'sha256.json').read_text())
    require(set(hashes)=={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file() and f.name not in ('sha256.json','sha256.json.sha256')}, 'sidecar coverage')
    require((output/'sha256.json.sha256').read_text().strip()==file_sha(output/'sha256.json'),'hash sidecar')
    for name,h in hashes.items(): require(file_sha(output/name)==h,'output hash '+name)
    require(json.loads((output/'rejections.json').read_text())==audit,'rejection audit')
    require(json.loads((output/'registry.json').read_text())==r and json.loads((output/'plan.json').read_text())==p,'input snapshots')
    ds=lance.dataset(str(output/'compact.lance'))
    require(ds.count_rows()==800 and ds.schema.equals(schema(),check_metadata=True), 'Lance count/schema')
    # Rebuild each row from immutable second-pass evidence. Arrow normalization
    # compares float32 compact values and float64 full state without tolerances.
    for i,record in enumerate(records):
        expected=pa.Table.from_pylist([make_row(registry,record,p)],schema=schema()).to_pylist()[0]
        actual=ds.take([i]).to_pylist()[0]
        require(actual==expected, 'readback content/lineage/contact mismatch at row '+str(i))
    return dict(rows=800,counts={a:160 for a in r['parents']},contract=CONTRACT)


def export(output, registry, plan, attempts):
    import lance
    import pyarrow as pa
    require(output.name.startswith('.'), 'new output directory must be hidden')
    require(not output.exists(), 'existing output is immutable')
    r,p,records,audit=collect(registry,plan,attempts)
    output.mkdir(parents=True,exist_ok=False)
    for src,name in [(registry,'registry.json'),(plan,'plan.json')]: shutil.copyfile(src,output/name)
    write_json(output/'rejections.json',audit)
    write_json(output/'setting.json',r['parents']['003']['setting'])
    write_json(output/'manifest.json',dict(contract=CONTRACT,registry_digest=r['digest'],plan_digest=p['digest'],setting_sha256=SETTING,registry_sha256=file_sha(registry),plan_sha256=file_sha(plan),rows=800,counts={a:160 for a in r['parents']},ordered_uuids=[x['uuid'] for x in records]))
    s=schema()
    batches=(pa.RecordBatch.from_pylist([make_row(registry,x,p)],schema=s) for x in records)
    lance.write_dataset(pa.RecordBatchReader.from_batches(s,batches),str(output/'compact.lance'),mode='create')
    write_json(output/'sha256.json',{str(f.relative_to(output)):file_sha(f) for f in sorted(output.rglob('*')) if f.is_file()})
    (output/'sha256.json.sha256').write_text(file_sha(output/'sha256.json')+'\n')
    return validate(output,registry,plan,attempts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['export','validate'])
    for key in ('registry','plan','attempts','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--execute-export',action='store_true')
    a=parser.parse_args()
    if a.mode=='export': require(a.execute_export,'export requires --execute-export')
    print(json.dumps((export if a.mode=='export' else validate)(a.output,a.registry,a.plan,a.attempts),indent=2))

if __name__=='__main__': main()
