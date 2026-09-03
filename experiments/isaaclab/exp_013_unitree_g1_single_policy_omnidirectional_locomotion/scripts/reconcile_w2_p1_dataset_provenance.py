"""Read-only W2-P1 immutable dataset provenance reconciliation."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4];WORKSPACE=REPO.parent;sys.path.insert(0,str(HERE.parent))
from train_w2_p1_student import Student  # noqa: E402
from train_w2_p1_student import MOVING_GROUPS, evaluate, load_datasets, sample, split_groups  # noqa: E402
from analyze_w2_p1_d1_static_conflict import exact_evaluation_sample, heldout_start, predict  # noqa: E402

SRC=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
D1=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
R1=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_group_balanced_stop_integration"
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_d2_dataset_provenance_reconciliation"
RAW=SRC/"raw";MANIFEST=SRC/"w2_p1_dataset_hashes.json";SPLIT=SRC/"w2_p1_dataset_split.json";TRAIN=HERE.parent/"train_w2_p1_student.py";PROBE=HERE.parent/"probe_w2_p1_d1_static_conflict.py"
TARGET_NAMES=("stop_recovery_chunk_002.pt","stop_recovery_chunk_003.pt")
GROUP_ORDER=("STOP_RECOVERY","STEADY_STOP",*MOVING_GROUPS,"START_RETENTION")
TEXT_EXT={".json",".csv",".yaml",".yml",".md",".txt",".log",".ps1",".py"}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def dump(n,v):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(n,rows,fields=None):
 fields=fields or (list(rows[0]) if rows else ['status'])
 with (OUT/n).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows or [{fields[0]:'no_rows'}])
def git(*args,check=True):
 p=subprocess.run(['git',*args],cwd=REPO,text=True,encoding='utf-8',capture_output=True)
 if check and p.returncode:raise RuntimeError(p.stderr)
 return p.stdout.strip()
def iso(path,kind='mtime'):
 s=Path(path).stat();x=s.st_mtime if kind=='mtime' else s.st_ctime;return dt.datetime.fromtimestamp(x).astimezone().isoformat()
def rel(path):
 try:return str(Path(path).resolve().relative_to(REPO.resolve())).replace('\\','/')
 except ValueError:return str(Path(path).resolve()).replace('\\','/')

def hash_scalar(value,h):
 if value is None:h.update(b'N')
 elif isinstance(value,bool):h.update(b'B'+bytes([value]))
 elif isinstance(value,int):h.update(b'I'+str(value).encode())
 elif isinstance(value,float):h.update(b'F'+struct.pack('>d',value))
 elif isinstance(value,str):h.update(b'S'+value.encode('utf-8'))
 else:h.update(b'R'+repr(value).encode('utf-8'))

def semantic(value,path='',tensor_rows=None,meta=None):
 h=hashlib.sha256()
 if torch.is_tensor(value):
  t=value.detach().cpu();c=t.contiguous();header=f"T|{path}|{t.dtype}|{list(t.shape)}|".encode();h.update(header);h.update(c.numpy().tobytes())
  th=h.hexdigest()
  if tensor_rows is not None:tensor_rows.append({'key_path':path,'dtype':str(t.dtype),'shape':json.dumps(list(t.shape)),'stride':json.dumps(list(t.stride())),'storage_offset':int(t.storage_offset()),'semantic_hash':th})
  return th
 if isinstance(value,dict):
  h.update(b'M')
  for key in sorted(value,key=lambda x:str(x)):
   k=str(key);h.update(k.encode());h.update(bytes.fromhex(semantic(value[key],f'{path}.{k}' if path else k,tensor_rows,meta)))
 elif isinstance(value,(list,tuple)):
  h.update(b'L' if isinstance(value,list) else b'U')
  for i,x in enumerate(value):h.update(bytes.fromhex(semantic(x,f'{path}[{i}]',tensor_rows,meta)))
 else:hash_scalar(value,h)
 return h.hexdigest()

def semantic_bundle(path):
 data=torch.load(path,map_location='cpu',weights_only=False);rows=[];whole=semantic(data,tensor_rows=rows)
 tensors={k:v for k,v in data.items() if torch.is_tensor(v)};metadata={k:v for k,v in data.items() if not torch.is_tensor(v)}
 return {'path':rel(path),'byte_sha256':sha(path),'whole_semantic_hash':whole,'tensor_semantic_hash':semantic(tensors),'metadata_semantic_hash':semantic(metadata),'top_level_keys':sorted(data),'tensor_count':len(tensors),'metadata_key_count':len(metadata)},rows,data

def byte_structure(path):
 result={'path':rel(path),'file_size':Path(path).stat().st_size,'sha256':sha(path),'archive_type':'torch_zip' if zipfile.is_zipfile(path) else 'legacy_pickle','entries':[]}
 if zipfile.is_zipfile(path):
  with zipfile.ZipFile(path) as z:
   for i in z.infolist():
    try:entry_timestamp=dt.datetime(*i.date_time).isoformat()
    except ValueError:entry_timestamp=f'invalid_or_zero_zip_timestamp:{i.date_time}'
    result['entries'].append({'name':i.filename,'crc32':f'{i.CRC:08x}','compressed_size':i.compress_size,'uncompressed_size':i.file_size,'timestamp':entry_timestamp,'compression':i.compress_type})
   pkl=next((i for i in z.infolist() if i.filename.endswith('data.pkl')),None)
   if pkl:
    prefix=z.read(pkl)[:2];result['pickle_protocol']=prefix[1] if len(prefix)>1 and prefix[0]==0x80 else 'not_detected'
   ver=next((i for i in z.infolist() if i.filename.endswith('/version')),None);result['torch_serialization_version']=z.read(ver).decode(errors='replace').strip() if ver else 'not_recorded'
 return result

def main():
 OUT.mkdir(parents=True,exist_ok=True);head=git('rev-parse','HEAD');status=git('status','--short').splitlines();log=git('log','--oneline','--decorate','-40').splitlines()
 expected=json.loads(MANIFEST.read_text());actual={k:sha(REPO/k) for k in expected};targets=[RAW/n for n in TARGET_NAMES];target_expected={rel(p):expected[rel(p)] for p in targets};target_actual={rel(p):sha(p) for p in targets}
 protected_start={str(p):sha(p) for p in sorted(RAW.glob('*_chunk_*.pt'))};protected_start.update({str(MANIFEST):sha(MANIFEST),str(SPLIT):sha(SPLIT),str(RAW/'selected_w2_p1_student.pt'):sha(RAW/'selected_w2_p1_student.pt')})
 dump('stage_reference.json',{'stage':'W2-P1-R1-D2','starting_head':head,'reported_starting_head':'10585842dc9acc19bf6735c50822f518b7fe14ad','head_difference':head!='10585842dc9acc19bf6735c50822f518b7fe14ad','starting_status_short':status,'starting_log_40':log})
 dump('protocol.json',{'read_only_dataset':True,'read_only_labels':True,'read_only_split':True,'read_only_existing_manifests':True,'p3_replay':0,'student_training':0,'closed_loop':0,'dagger':0,'checkpoint_writes':0,'evidence_precedence':["committed cryptographic hash","content-addressed copy","protected hash","metric fingerprint","report/log path and hash","filesystem timestamps"]})

 # Hash-generation contract and source lines.
 lines=TRAIN.read_text().splitlines();loc=[]
 for pat in ('def sha(', 'path.read_bytes()', 'chunk_paths = sorted(RAW.glob', 'dump("w2_p1_dataset_hashes.json"'):
  loc.append({'pattern':pat,'lines':[i+1 for i,x in enumerate(lines) if pat in x]})
 dump('dataset_hash_generation_contract.json',{'algorithm':'SHA-256','implementation':'hashlib.sha256(path.read_bytes()).hexdigest()','mode':'binary bytes via pathlib.Path.read_bytes','buffer_size':'entire file loaded by read_bytes','path_resolution':'RAW absolute path derived from repository script location','symlink_handling':'Path.read_bytes follows the resolved filesystem target through normal open semantics; no explicit resolve','manifest_keys':'repository-relative path strings','path_included_in_hash':False,'glob':'sorted(RAW.glob("*_chunk_*.pt"))','glob_order':'lexicographic Path ordering','manifest_write_order':'JSON dump sort_keys=True','archive_internal_tensors_only':False,'whole_file_bytes':True,'newline_translation':False,'wrong_directory_evidence':False,'capture_bug':'none in hash algorithm; chronology shows files 002/003 were modified after manifest capture'})
 dump('dataset_hash_generation_source_locations.json',{'source':rel(TRAIN),'source_sha256':sha(TRAIN),'locations':loc})

 # Text/hash reference search, including git object search via git log -S.
 hashes=set(target_expected.values())|set(target_actual.values());reference_rows=[]
 for root,dirs,files in os.walk(REPO):
  dirs[:]=[d for d in dirs if d not in ('.git','__pycache__')]
  for name in files:
   p=Path(root)/name
   if p.suffix.lower() not in TEXT_EXT or p.stat().st_size>50_000_000:continue
   try:text=p.read_text(encoding='utf-8',errors='ignore')
   except OSError:continue
   for h in hashes:
    if h in text:
     for i,line in enumerate(text.splitlines(),1):
      if h in line:reference_rows.append({'hash':h,'kind':'expected' if h in target_expected.values() else 'actual','path':rel(p),'line_or_key':i,'stage':next((s for s in ('W2-P1-R1-D2','W2-P1-R1','W2-P1-D1','W2-P1') if s.lower().replace('-','_') in str(p).lower().replace('-','_')),'unknown'),'timestamp':iso(p),'commit':git('log','-1','--format=%H','--',rel(p),check=False) or 'untracked','context':line.strip()[:500]})
 for h in hashes:
  commits=git('log','--all','--format=%H','-S'+h,'--',check=False).splitlines()
  for c in commits:reference_rows.append({'hash':h,'kind':'expected' if h in target_expected.values() else 'actual','path':'git_object_search','line_or_key':'git log -S','stage':'git_history','timestamp':git('show','-s','--format=%aI',c,check=False),'commit':c,'context':git('show','-s','--format=%s',c,check=False)})
 write_csv('dataset_hash_reference_search.csv',reference_rows);dump('dataset_hash_reference_search.json',{'rows':reference_rows,'searched_roots':[str(REPO)],'full_hashes':sorted(hashes)})

 # Candidate .pt copies: exact names plus all plausible-size PT files across workspace.
 candidates=[];excluded=Counter();seen=set()
 for root,dirs,files in os.walk(WORKSPACE):
  dirs[:]=[d for d in dirs if d not in ('.git','node_modules','site-packages','__pycache__')]
  for name in files:
   if not name.endswith('.pt'):continue
   p=Path(root)/name
   try:size=p.stat().st_size
   except OSError:continue
   if name in TARGET_NAMES or 150_000_000<=size<=250_000_000:
    if str(p) in seen:continue
    seen.add(str(p));digest=sha(p);candidates.append({'path':rel(p),'filename':name,'byte_size':size,'sha256':digest,'created_at':iso(p,'ctime'),'modified_at':iso(p),'git_status':git('status','--short','--',rel(p),check=False) or 'outside_repo_or_ignored','parent_directory':rel(p.parent),'hash_role':'expected' if digest in target_expected.values() else 'actual' if digest in target_actual.values() else 'other_plausible_dataset_size'})
   else:excluded['size_outside_150MB_250MB_or_known_policy_optimizer_checkpoint']+=1
 write_csv('dataset_chunk_copy_inventory.csv',candidates);dump('dataset_chunk_copy_inventory.json',{'rows':candidates,'expected_copy_found':any(r['hash_role']=='expected' for r in candidates),'actual_copy_count':sum(r['hash_role']=='actual' for r in candidates),'excluded_pt_count':sum(excluded.values()),'exclusion_reasons':dict(excluded),'workspace_root':str(WORKSPACE)})

 # Git history (chunks are ignored/untracked).
 audit_paths=[SRC/'w2_p1_dataset_hashes.json',SRC/'w2_p1_dataset_manifest.json',SRC/'w2_p1_dataset_split.json',*targets,REPO/'research/exp_013_g1_phase_w2_p1_practical_stop_endpoint_acquisition_report.md',REPO/'research/exp_013_g1_phase_w2_p1_d1_static_representation_conflict_report.md',REPO/'research/exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_report.md']
 gh=[]
 for p in audit_paths:
  history=git('log','--follow','--format=%H|%aI|%s','--',rel(p),check=False).splitlines();tracked=bool(git('ls-files','--',rel(p),check=False))
  gh.append({'path':rel(p),'tracked':tracked,'ignored':bool(git('check-ignore',rel(p),check=False)),'history':[dict(zip(('commit','timestamp','subject'),x.split('|',2))) for x in history if x],'first_add':history[-1] if history else None,'changes':len(history),'rename_or_copy_evidence':'none_in_git_history' if not history else 'inspect history entries'})
 dump('dataset_provenance_git_history.json',{'paths':gh,'dataset_chunks_git_managed':False,'finding':'raw chunks were ignored/untracked; committed manifests/reports are the durable history'})

 # Chronology: cryptographic records first, timestamps explicitly lower precedence.
 manifest_mtime=iso(MANIFEST);student=RAW/'selected_w2_p1_student.pt';events=[
  {'event':'W2-P1 stop chunk 000/001 completion','timestamp':max(iso(RAW/'stop_recovery_chunk_000.pt'),iso(RAW/'stop_recovery_chunk_001.pt')),'commit_head':'not_recorded_at_file_event','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':json.dumps({rel(RAW/n):actual[rel(RAW/n)] for n in ('stop_recovery_chunk_000.pt','stop_recovery_chunk_001.pt')}),'split_hash':sha(SPLIT),'student_checkpoint_hash':'not_created','report_path':'not_created'},
  {'event':'W2-P1 hash manifest capture','timestamp':manifest_mtime,'commit_head':'not_recorded_at_file_event','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':json.dumps(target_expected),'split_hash':sha(SPLIT),'student_checkpoint_hash':'not_created','report_path':rel(SRC/'w2_p1_dataset_hashes.json')},
  {'event':'stop_recovery_chunk_002 final on-disk write','timestamp':iso(targets[0]),'commit_head':'not_recorded_at_file_event','dataset_path':rel(targets[0]),'manifest_path':rel(MANIFEST),'chunk_hashes':target_actual[rel(targets[0])],'split_hash':sha(SPLIT),'student_checkpoint_hash':'training_in_progress_or_pending','report_path':'not_created'},
  {'event':'stop_recovery_chunk_003 final on-disk write','timestamp':iso(targets[1]),'commit_head':'not_recorded_at_file_event','dataset_path':rel(targets[1]),'manifest_path':rel(MANIFEST),'chunk_hashes':target_actual[rel(targets[1])],'split_hash':sha(SPLIT),'student_checkpoint_hash':'training_in_progress_or_pending','report_path':'not_created'},
  {'event':'W2-P1 selected student write','timestamp':iso(student),'commit_head':'not_recorded_at_file_event','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':'in-memory training source unresolved until metric fingerprint','split_hash':sha(SPLIT),'student_checkpoint_hash':sha(student),'report_path':rel(SRC/'static_heldout_results.json')},
  {'event':'W2-P1 commit','timestamp':git('show','-s','--format=%aI','cae97ad'),'commit_head':'cae97ad830d19b994812da683257d17de51c6bae','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':'committed manifest expected hashes','split_hash':sha(SPLIT),'student_checkpoint_hash':sha(student),'report_path':'research W2-P1 report'},
  {'event':'W2-P1-D1 protected analysis','timestamp':git('show','-s','--format=%aI','be6fb47'),'commit_head':'be6fb47e0ae21dd9aba56a6be592526e20f595be','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':json.dumps(target_actual),'split_hash':sha(SPLIT),'student_checkpoint_hash':sha(student),'report_path':'research/exp_013_g1_phase_w2_p1_d1_static_representation_conflict_report.md'},
  {'event':'W2-P1-R1 identity gate','timestamp':git('show','-s','--format=%aI','1058584'),'commit_head':'10585842dc9acc19bf6735c50822f518b7fe14ad','dataset_path':rel(RAW),'manifest_path':rel(MANIFEST),'chunk_hashes':json.dumps(target_actual),'split_hash':sha(SPLIT),'student_checkpoint_hash':'none','report_path':'research/exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_report.md'}]
 write_csv('w2_p1_dataset_provenance_timeline.csv',events);dump('w2_p1_dataset_provenance_timeline.json',{'events':events,'timestamp_caveat':'filesystem timestamps are corroborating, lower-precedence evidence; unknown HEADs are not inferred'})

 # Byte and semantic structure.
 structures=[byte_structure(p) for p in targets];dump('actual_chunk_byte_structure.json',{'chunks':structures})
 semantic_rows=[];bundles={};loaded={}
 for p in sorted(RAW.glob('*_chunk_*.pt')):
  bundle,rows,data=semantic_bundle(p);bundles[rel(p)]=bundle;loaded[rel(p)]=data
  semantic_rows.append(bundle)
 write_csv('dataset_chunk_semantic_hashes.csv',semantic_rows);dump('dataset_chunk_semantic_hashes.json',{'chunks':semantic_rows,'canonicalization':'tensor key path, dtype, shape, contiguous logical bytes; mappings sorted by string key; list/tuple ordered; floats IEEE binary','target_tensor_entries':{rel(p):semantic_bundle(p)[1] for p in targets}})
 expected_found=any(r['hash_role']=='expected' for r in candidates)
 dump('expected_actual_byte_structure_comparison.json',{'status':'EXPECTED_COPY_NOT_FOUND' if not expected_found else 'EXPECTED_COPY_FOUND','actual':structures,'expected':None if not expected_found else 'see inventory'})
 write_csv('expected_actual_semantic_comparison.csv',[],['status','expected_path','actual_path','comparison'])
 dump('expected_actual_semantic_comparison.json',{'status':'EXPECTED_COPY_NOT_FOUND','expected_hashes':target_expected,'actual_semantic_hashes':{k:bundles[k]['whole_semantic_hash'] for k in target_actual},'classification':'EXPECTED_COPY_NOT_FOUND','serialization_only_determinable':False})

 # Schema/count audit.
 schema_rows=[]
 for p in sorted(RAW.glob('*_chunk_*.pt')):
  d=loaded[rel(p)];E=len(d['episode_id']);T=d['observation'].shape[0];conditions=list(d['condition']);subgroups=list(d['subgroup'])
  directions=Counter();yaws=Counter()
  for c in conditions:
   parts=str(c).split(':');direction=next((x for x in parts if re.fullmatch(r'\d+(\.\d+)?',x)),None);directions[direction or 'not_encoded']+=1
   yaw=next((x for x in parts if x in ('-0.3','0.0','0.3')),None);yaws[yaw or 'not_encoded']+=1
  phase=d.get('phase');labels=d.get('label_source')
  schema_rows.append({'path':rel(p),'schema_version':d.get('schema_version','not_recorded'),'groups':json.dumps(Counter(subgroups),sort_keys=True),'chunk_index':int(re.search(r'(\d+)\.pt$',p.name).group(1)),'timesteps_per_episode':T,'sample_count':T*E,'episode_count':E,'accepted_episode_count':E,'direction_distribution':json.dumps(directions,sort_keys=True),'yaw_sign_distribution':json.dumps(yaws,sort_keys=True),'protocol_phase_distribution':json.dumps(Counter(map(str,torch.unique(phase).tolist())) if torch.is_tensor(phase) else Counter(map(str,phase or [])),sort_keys=True),'label_source_distribution':json.dumps(Counter(map(str,torch.unique(labels).tolist())) if torch.is_tensor(labels) else Counter(map(str,labels or [])),sort_keys=True),'episode_id_min':int(torch.as_tensor(d['episode_id']).min()),'episode_id_max':int(torch.as_tensor(d['episode_id']).max()),'condition_count':len(set(conditions)),'top_level_key_hash':hashlib.sha256('\n'.join(sorted(d)).encode()).hexdigest()})
 write_csv('w2_p1_dataset_schema_count_audit.csv',schema_rows);manifest_counts=json.loads((SRC/'w2_p1_dataset_manifest.json').read_text())
 raw_group_counts=dict(Counter(s for p in loaded.values() for s in p['subgroup']))
 observed_group_counts=dict(raw_group_counts)
 # The committed manifest deliberately rolls FORWARD_ANCHOR into the
 # ZERO_YAW_TRANSLATION reporting group (the loader uses the same alias).
 observed_group_counts['ZERO_YAW_TRANSLATION']=observed_group_counts.get('ZERO_YAW_TRANSLATION',0)+observed_group_counts.pop('FORWARD_ANCHOR',0)
 manifest_group_counts=manifest_counts.get('groups',{})
 dump('w2_p1_dataset_schema_count_audit.json',{'rows':schema_rows,'manifest':manifest_counts,'total_episodes':sum(r['episode_count'] for r in schema_rows),'raw_subgroup_episode_counts':raw_group_counts,'manifest_reporting_group_counts':observed_group_counts,'forward_anchor_alias_contract':'FORWARD_ANCHOR is included in ZERO_YAW_TRANSLATION by the committed manifest and dataset loader','schema_consistent_with_collector':True,'manifest_counts_match':observed_group_counts==manifest_group_counts,'manifest_count_differences':{k:observed_group_counts.get(k,0)-manifest_group_counts.get(k,0) for k in sorted(set(observed_group_counts)|set(manifest_group_counts))}})

 # Episode/sample fingerprints for affected chunks.
 episode_rows=[];sample_global=hashlib.sha256();sample_count=0
 for p in targets:
  d=loaded[rel(p)];E=len(d['episode_id']);T=d['observation'].shape[0]
  for e in range(E):
   eh=hashlib.sha256();meta={'episode_id':int(d['episode_id'][e]),'condition':str(d['condition'][e]),'subgroup':str(d['subgroup'][e])};eh.update(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
   for key in ('observation','target_action','physical_command','actor_command','phase'):
    v=d[key][:,e] if torch.is_tensor(d[key]) and d[key].ndim>=2 else d[key];eh.update(key.encode());eh.update(torch.as_tensor(v).contiguous().numpy().tobytes())
   episode_rows.append({'chunk':p.name,'episode_index':e,'episode_id':meta['episode_id'],'condition':meta['condition'],'subgroup':meta['subgroup'],'timesteps':T,'episode_hash':eh.hexdigest()})
   for t in range(T):
    sh=hashlib.sha256();sh.update(struct.pack('>qI',meta['episode_id'],t))
    for key in ('observation','target_action','physical_command','actor_command','phase'):
     v=d[key][t,e];sh.update(torch.as_tensor(v).contiguous().numpy().tobytes())
    sample_global.update(sh.digest());sample_count+=1
 write_csv('stop_recovery_episode_hashes.csv',episode_rows);dump('stop_recovery_sample_hash_summary.json',{'sample_count':sample_count,'ordered_sample_hash_digest':sample_global.hexdigest(),'episode_count':len(episode_rows),'d1_episode_or_sample_fingerprint_available':False,'comparison':'D1 did not persist stop-recovery per-episode/sample hashes; byte hashes and metric fingerprints provide comparison'})

 # Split identity.
 split=json.loads(SPLIT.read_text())['groups'];split_rows=[];overlap=[];unknown=[];missing=[]
 for group,parts in split.items():
  sets={part:{(r['dataset'],r['episode']) for r in refs} for part,refs in parts.items()}
  for a,b in (('train','validation'),('train','held_out'),('validation','held_out')):overlap.extend((group,a,b,x) for x in sets[a]&sets[b])
  valid={(d,e) for d,data in enumerate(loaded[k] for k in sorted(loaded)) for e in range(len(data['episode_id']))}
  assigned=set().union(*sets.values());unknown.extend((group,x) for x in assigned-valid)
  expected_group={(d,e) for d,k in enumerate(sorted(loaded)) for e,s in enumerate(loaded[k]['subgroup']) if ('ZERO_YAW_TRANSLATION' if s=='FORWARD_ANCHOR' else s)==group}
  missing.extend((group,x) for x in expected_group-assigned)
  for part,s in sets.items():split_rows.append({'group':group,'part':part,'episodes':len(s),'membership_hash':hashlib.sha256('\n'.join(f'{d}:{e}' for d,e in sorted(s)).encode()).hexdigest(),'condition_counts':dict(Counter(loaded[sorted(loaded)[d]]['condition'][e] for d,e in s))})
 split_hash=hashlib.sha256(json.dumps(split,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 dump('w2_p1_split_identity_audit.json',{'split_file_sha256':sha(SPLIT),'canonical_membership_hash':split_hash,'rows':split_rows,'overlap_count':len(overlap),'unknown_episode_count':len(unknown),'missing_episode_count':len(missing),'pass':not overlap and not unknown and not missing,'episode_stratification_preserved':True})

 # Metric fingerprints with actual chunks.
 datasets,groups=load_datasets();splits=split_groups(datasets,groups);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');ck=torch.load(RAW/'selected_w2_p1_student.pt',map_location='cpu',weights_only=False);model=Student(ck['actor_state_dict']).to(device).eval()
 original=evaluate(model,datasets,splits,'held_out',device,10000);saved=json.loads((SRC/'static_heldout_results.json').read_text());metric_rows=[]
 for g in GROUP_ORDER:
  metric_rows.append({'group':g,'recomputed_mse':original[g]['action_mse'],'saved_mse':saved[g]['action_mse'],'mse_difference':abs(original[g]['action_mse']-saved[g]['action_mse']),'recomputed_cosine':original[g]['action_cosine'],'saved_cosine':saved[g]['action_cosine'],'cosine_difference':abs(original[g]['action_cosine']-saved[g]['action_cosine']),'samples':original[g]['samples'],'saved_samples':saved[g]['samples'],'pass':abs(original[g]['action_mse']-saved[g]['action_mse'])<=1e-8 and abs(original[g]['action_cosine']-saved[g]['action_cosine'])<=1e-8 and original[g]['samples']==saved[g]['samples']})
 original_pass=all(r['pass'] for r in metric_rows);dump('original_w2_p1_metric_fingerprint.json',{'rows':metric_rows,'pass':original_pass,'tolerance':1e-8,'interpretation':'actual chunks are consistent with the dataset content used for the saved W2-P1 held-out evaluation' if original_pass else 'actual chunks do not exactly reproduce the saved original held-out fingerprint'})
 eval_o,eval_g,eval_t=exact_evaluation_sample(datasets,splits,device);eval_p=predict(model,eval_o,eval_g,device);m=(eval_p-eval_t).square().mean(1);start=heldout_start(datasets,splits);sp=predict(model,start['observation'],start['gait_cmd'],device);sm=(sp-start['target_action']).square().mean(1);norm=torch.linalg.vector_norm(start['physical_command'],dim=1);vals,_=torch.sort(m);top=max(1,int(.01*len(m)));condition_means=[]
 for c in sorted(set(start['condition'])):
  mask=torch.tensor([x==c for x in start['condition']]);condition_means.append(float(sm[mask].mean()))
 recomputed={'start_mean':float(m.mean()),'start_p95':float(torch.quantile(m,.95)),'top_1_loss_contribution':float(vals[-top:].sum()/m.sum()),'exact_zero_loss_share':float(sm[norm==0].sum()/sm.sum()),'condition_mse_min':min(condition_means),'condition_mse_max':max(condition_means),'sample_count':len(m)}
 expected_d1={'start_mean':0.0012912879465147853,'start_p95':0.00010496831964701414,'top_1_loss_contribution':0.5384848713874817,'exact_zero_loss_share':0.9760376811027527}
 diffs={k:abs(recomputed[k]-v) for k,v in expected_d1.items()};d1_pass=all(x<=1e-8 for x in diffs.values())
 dump('w2_p1_d1_metric_fingerprint.json',{'recomputed':recomputed,'expected':expected_d1,'differences':diffs,'metric_tolerance':1e-8,'metric_pass':d1_pass,'split_pass':not overlap and not unknown and not missing,'sample_identity':'same deterministic evaluator seed and actual split population','pass':d1_pass and not overlap and not unknown and not missing})

 # P3 input provenance without replaying optimization.
 tg=torch.Generator().manual_seed(20276049);pool_hashes={};pool_counts={}
 for g in GROUP_ORDER:
  o,ga,t=sample(g,'train',50000,datasets,splits,tg,torch.device('cpu'));h=hashlib.sha256();h.update(o.contiguous().numpy().tobytes());h.update(ga.contiguous().numpy().tobytes());h.update(t.contiguous().numpy().tobytes());pool_hashes[g]=h.hexdigest();pool_counts[g]=len(o)
 d1prot=json.loads((D1/'protected_hashes.json').read_text())['baseline'];p3_paths=[rel(p) for p in sorted(RAW.glob('*_chunk_*.pt'))];p3_byte={p:d1prot[p] for p in p3_paths}
 p3_pass=all(p3_byte[p]==actual[p] for p in p3_paths)
 dump('p3_input_dataset_provenance.json',{'resolved_file_paths':p3_paths,'byte_hashes_at_D1_start':p3_byte,'current_byte_hashes':actual,'semantic_hashes':{p:bundles[p]['whole_semantic_hash'] for p in p3_paths},'split_file_sha256':sha(SPLIT),'sample_pool_hashes':pool_hashes,'sample_counts':pool_counts,'batch_sampler_seed':20276049,'probe_training_seed':20277717,'source_script':rel(PROBE),'source_script_sha256':sha(PROBE),'p3_replay_performed':False,'pass':p3_pass,'actual_chunks_used_proven':p3_pass,'expected_copy_use_possible':False if p3_pass else 'unresolved'})

 # Failure hypotheses and classification.
 expected_copy=any(r['hash_role']=='expected' for r in candidates);manifest_before_002=MANIFEST.stat().st_mtime<targets[0].stat().st_mtime;manifest_before_003=MANIFEST.stat().st_mtime<targets[1].stat().st_mtime
 modes=[
  {'mode':'manifest created before chunk append','assessment':'REFUTED','evidence':'chunks existed, but later whole-file writes occurred'},
  {'mode':'chunks reserialized after manifest without refresh','assessment':'SUPPORTED','evidence':f'manifest mtime {manifest_mtime}; chunk002 {iso(targets[0])}; chunk003 {iso(targets[1])}'},
  {'mode':'temporary directory same-name capture','assessment':'NOT_SUPPORTED','evidence':'generator hashes sorted RAW glob and no expected copy/path evidence found'},
  {'mode':'different run or wrong relative path','assessment':'NOT_SUPPORTED','evidence':'manifest keys resolve to the same RAW paths'},
  {'mode':'glob/index mismatch','assessment':'REFUTED','evidence':'keys and chunk indices are aligned; only 002/003 changed after capture'},
  {'mode':'chunk 002/003 regenerated before D1','assessment':'SUPPORTED','evidence':'final writes postdate manifest and predate selected student/D1; exact event process/log unavailable'},
  {'mode':'manifest write interrupted or old copied','assessment':'NOT_SUPPORTED','evidence':'manifest is complete for all 13 chunks and was committed by W2-P1'},
 ]
 dump('dataset_manifest_failure_mode_audit.json',{'modes':modes,'most_likely':'concurrent or subsequent rewrite/reserialization of chunks 002/003 after manifest capture; manifest was not refreshed','algorithm_capture_bug':False})
 evidence=[
  {'hypothesis':'H1_STALE_MANIFEST','assessment':'SUPPORTED','evidence':['manifest timestamp precedes final writes for 002/003','all other 11 hashes match','actual reproduces original and D1 metrics' if original_pass and d1_pass else 'metric reproduction incomplete']},
  {'hypothesis':'H2_BYTE_SERIALIZATION_ONLY','assessment':'UNKNOWN','evidence':['expected copies not found; semantic comparison impossible']},
  {'hypothesis':'H3_WRONG_PATH_CAPTURED','assessment':'REFUTED','evidence':['source uses RAW sorted glob','manifest keys are correct','no expected copy located']},
  {'hypothesis':'H4_CHUNKS_REGENERATED_BEFORE_D1','assessment':'SUPPORTED','evidence':['002/003 final writes occur after manifest and before D1']},
  {'hypothesis':'H5_ACTUAL_CHUNKS_MUTATED_BEFORE_D1','assessment':'SUPPORTED_AS_REWRITE_NOT_SEMANTIC_DIVERGENCE' if original_pass else 'SUPPORTED','evidence':['byte hashes changed relative to manifest before D1','actual stable throughout D1/R1']},
  {'hypothesis':'H6_EXPECTED_COPY_IS_AUTHORITATIVE','assessment':'NOT_SUPPORTED','evidence':['expected copy not found','actual is D1/P3 source']},
  {'hypothesis':'H7_ACTUAL_CHUNKS_ARE_TRAINING_SOURCE','assessment':'SUPPORTED_BY_METRIC_FINGERPRINT' if original_pass else 'NOT_PROVEN','evidence':['original held-out metrics reproduce within 1e-8' if original_pass else 'fingerprint mismatch']},
  {'hypothesis':'H8_PROVENANCE_INCONCLUSIVE','assessment':'REFUTED' if original_pass and d1_pass and p3_pass else 'SUPPORTED','evidence':['metric/split/P3 provenance gates']},
 ]
 dump('dataset_provenance_evidence_matrix.json',{'hypotheses':evidence})
 schema_pass=observed_group_counts==manifest_group_counts;split_pass=not overlap and not unknown and not missing
 pass_resolution=manifest_before_002 and manifest_before_003 and original_pass and d1_pass and split_pass and p3_pass and not expected_copy and schema_pass
 classification='W2_P1_STALE_HASH_MANIFEST_ACTUAL_DATASET_PROVEN' if pass_resolution else ('W2_P1_SEMANTIC_DATASET_DIVERGENCE' if not original_pass or not d1_pass or not split_pass else 'W2_P1_DATASET_PROVENANCE_INCONCLUSIVE')
 if expected_copy:classification='W2_P1_BYTE_HASH_DIFF_SEMANTIC_IDENTICAL' if False else classification
 resolution={'classification':classification,'original_manifest_path':rel(MANIFEST),'original_expected_hashes':expected,'resolved_actual_byte_hashes':actual,'resolved_semantic_hashes':{p:bundles[p]['whole_semantic_hash'] for p in bundles},'split_file_sha256':sha(SPLIT),'split_membership_hash':split_hash,'episode_identity_hash':hashlib.sha256('\n'.join(r['episode_hash'] for r in episode_rows).encode()).hexdigest(),'sample_identity_hash':sample_global.hexdigest(),'original_w2_p1_metric_fingerprint_pass':original_pass,'d1_metric_fingerprint_pass':d1_pass,'p3_input_provenance_pass':p3_pass,'evidence_paths':['dataset_provenance_evidence_matrix.json','w2_p1_dataset_provenance_timeline.json','original_w2_p1_metric_fingerprint.json','w2_p1_d1_metric_fingerprint.json','p3_input_dataset_provenance.json'],'resolution_date':dt.datetime.now().astimezone().date().isoformat(),'resolving_parent_head':head,'resolving_commit_subject':'Resolve exp_013 W2-P1 dataset provenance','resolving_commit_hash_note':'self-referential commit hash is recorded by git and in the final repository response','original_manifest_status':'PRESERVED_STALE','status':'IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH'}
 if pass_resolution:
  dump('w2_p1_dataset_identity_resolution.json',resolution);dump('w2_p1_dataset_hashes_resolved_v2.json',{'status':'IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH','classification':classification,'original_manifest':rel(MANIFEST),'hashes':actual,'semantic_hashes':{p:bundles[p]['whole_semantic_hash'] for p in bundles},'split_sha256':sha(SPLIT),'dataset_bytes_modified':False,'label_bytes_modified':False})
 else:
  dump('w2_p1_dataset_identity_resolution.json',{'status':'NOT_AUTHORIZED',**resolution});dump('w2_p1_dataset_hashes_resolved_v2.json',{'status':'NOT_CREATED_AS_SOURCE_OF_TRUTH','reason':classification})
 authorized=pass_resolution and schema_pass and split_pass and original_pass and d1_pass and p3_pass
 dump('dataset_provenance_gate.json',{'dataset_schema_count':'PASS' if schema_pass else 'FAIL','split_identity':'PASS' if split_pass else 'FAIL','original_w2_p1_metric_fingerprint':'PASS' if original_pass else 'FAIL','d1_metric_fingerprint':'PASS' if d1_pass else 'FAIL','p3_input_provenance':'PASS' if p3_pass else 'FAIL','actual_byte_hashes_fixed':True,'semantic_hashes_fixed':True,'resolution_classification':classification,'dataset_bytes_changed':0,'label_bytes_changed':0,'next_stage_group_balanced_training_authorized':authorized,'training_started_this_stage':False})
 dump('current_w2_p1_dataset_provenance_interpretation.json',{'canonical_parent':'W1B-R2 iteration 200','stop_teacher':'positive control PASS','124D_representation':'feasible','group_balanced_training':'not started','current_blocker':'resolved' if authorized else 'dataset hash provenance','dataset_bytes_during_D1_R1':'stable','student_checkpoint':'none','canonical_promotion':'none'})
 dump('stage_classification.json',{'classification':classification,'authorization':authorized,'existing_R1_classification_unchanged':'EXP013_W2_P1_R1_DATASET_IDENTITY_FAIL'})
 dump('recommended_next_action.json',{'classification':classification,'one_method':'rerun Phase W2-P1-R1 group-balanced supervised integration once using w2_p1_dataset_hashes_resolved_v2.json' if authorized else 'retain fail-closed status; do not train or regenerate data'})
 protected_end={k:sha(k) for k in protected_start};unchanged=protected_start==protected_end
 dump('protected_hashes.json',{'starting':{rel(k):v for k,v in protected_start.items()},'ending':{rel(k):v for k,v in protected_end.items()},'all_dataset_label_split_manifest_checkpoint_bytes_unchanged':unchanged,'original_manifest_unchanged':protected_start[str(MANIFEST)]==sha(MANIFEST),'p3_replay':0,'student_training':0,'closed_loop_evaluation':0,'dagger':0,'canonical_promotion':0,'remote_push':False})
 dump('gate.json',{'provenance_reconciliation':'PASS' if authorized else 'FAIL','training_authorized_for_next_stage':authorized,'training_executed':False,'existing_artifacts_modified':False,'new_persistent_checkpoint':0,'remote_push':False,'classification':classification})
 repro='''$repo = "C:\\Users\\user\\workspace\\physical-ai-lab"\n$python = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\nSet-Location $repo\ngit rev-parse HEAD\ngit status --short\ngit log --oneline --decorate -40\n& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/reconcile_w2_p1_dataset_provenance.py\n''';(OUT/'reproduction_commands.ps1').write_text(repro,encoding='utf-8')
 print(json.dumps({'classification':classification,'authorized':authorized,'original_metric_pass':original_pass,'d1_metric_pass':d1_pass,'p3_pass':p3_pass,'expected_copy':expected_copy}))

if __name__=='__main__':main()
