from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .io import ContractError, read_json, confined, sha256

ROLES={'train','inner_validation','validation','test','deployment'}
POOLS={'state','dynamics','thermocline','winter','boundary_response','spatial'}
TIME_KINDS={'instant','interval_mean','interval_total','unknown_instant_within_interval'}
REQUIRED_ARRAYS={'node_features','node_mass_kg','node_xyz_m','edge_index',
 'tokens','token_start_s','token_end_s','token_available_s','token_kind',
 'block_knots_s','physics_times_s'}

def require(condition: bool, text: str):
    if not condition: raise ContractError(text)

def finite(name, x):
    require(bool(np.isfinite(x).all()), f'{name}: nonfinite values')

@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    physical_lake_id: str
    isolation_group_id: str
    role: str
    external_path: Path
    targets_path: Path | None
    metadata_path: Path
    pools: tuple[str,...]

@dataclass
class ExternalEpisode:
    ref: EpisodeRef
    arrays: dict[str,np.ndarray]
    meta: dict[str,Any]

class Dataset:
    """Manifest-first, role-separated reader. Construction NEVER reads label files."""
    def __init__(self, manifest: Path):
        self.path=Path(manifest).resolve(); self.root=self.path.parent
        self.document=read_json(self.path)
        require(self.document.get('schema_version')=='lake-stinverse/1', 'Unsupported schema')
        require(self.document.get('data_kind') in {'real','synthetic_fixture'}, 'Explicit data_kind required')
        self.kind=self.document['data_kind']
        self.refs=[]; seen=set(); lake_roles={}; group_roles={}
        for r in self.document['episodes']:
            eid=r['episode_id']; role=r['role']; lake=r['physical_lake_id']; group=r['isolation_group_id']
            require(eid not in seen, f'Duplicate episode {eid}'); seen.add(eid)
            require(role in ROLES, f'Invalid role {role}')
            for key,mapping,label in [(lake,lake_roles,'physical lake'),(group,group_roles,'isolation group')]:
                require(key not in mapping or mapping[key]==role, f'Cross-role {label}: {key}')
                mapping[key]=role
            pools=tuple(r['pools']); require(set(pools)<=POOLS,'Unknown capability pool')
            self.refs.append(EpisodeRef(eid,lake,group,role,
                confined(self.root,r['external']), (confined(self.root,r['targets']) if role!='deployment' else None),
                confined(self.root,r['metadata']),pools))
        require(bool(self.refs),'Empty dataset')
        feat=self.document.get('feature_contract',{})
        require(bool(feat.get('node_names')) and bool(feat.get('token_names')),'Named feature contract required')
        require(feat.get('internal_temperature_features') is False,'No internal profile features allowed')
        self.feature_contract=feat
        if self.kind=='real':
            lineage=self.document.get('lineage',{})
            for k in ['identity_review','source_versions','split_protocol','time_policy','qa_policy']:
                require(bool(lineage.get(k)),f'Real data requires provenance: {k}')
            require(self.document.get('uses_interpolated_profiles_as_truth') is False,
                    'Interpolated profiles cannot be new truth')

    def _verify_file(self, path: Path) -> None:
        relative=str(path.relative_to(self.root))
        expected=self.document.get('file_hashes',{}).get(relative)
        require(expected is not None, f'Missing file SHA: {relative}')
        require(sha256(path)==expected, f'File changed: {relative}')

    def select(self, roles: set[str]) -> list[EpisodeRef]:
        require(roles<=ROLES,'Invalid role request')
        return [r for r in self.refs if r.role in roles]

    def external(self, ref: EpisodeRef) -> ExternalEpisode:
        """No target access, including in normalization and capability inventory."""
        self._verify_file(ref.metadata_path)
        self._verify_file(ref.external_path)
        meta=read_json(ref.metadata_path)
        require(meta['episode_id']==ref.episode_id,'Metadata ID mismatch')
        require(meta['data_kind']==self.kind,'Mixed synthetic and real datasets forbidden')
        if self.kind=='real':
            require(meta.get('time_origin_utc') is not None,'Absolute UTC origin required')
            require(meta.get('depth_reference') is not None,'Depth reference required')
            require(meta.get('raw_input_provenance') is not None,'Input lineage required')
            require(meta.get('external_QA_certified') is True,'External native QA not certified')
        with np.load(ref.external_path,allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        require(REQUIRED_ARRAYS<=a.keys(),f'Missing arrays: {REQUIRED_ARRAYS-a.keys()}')
        require(all(k in REQUIRED_ARRAYS or k.startswith('phys_') for k in a),
                'Unexpected external key; do not export internal labels or pseudo-initial profiles')
        for k,v in a.items():
            require(v.dtype.kind not in {'O','U','S'},f'Array {k} must be numerical')
            finite(k,v)
        n,g=a['node_features'].shape; s,f=a['tokens'].shape
        require(n>0 and s>0,'Need active nodes and external tokens')
        require(g==len(self.feature_contract['node_names']),'Node feature schema mismatch')
        require(f==len(self.feature_contract['token_names']),'Token feature schema mismatch')
        require(a['node_mass_kg'].shape==(n,) and bool((a['node_mass_kg']>0).all()),'Positive active masses required')
        require(a['node_xyz_m'].shape==(n,3),'Node coordinates must be [N,3]')
        e=a['edge_index']; require(e.ndim==2 and e.shape[0]==2,'edge_index must be [2,E]')
        require(e.dtype.kind in {'i','u'},'Edges must be integer indices')
        require(e.size==0 or (e.min()>=0 and e.max()<n),'Edge out of bounds')
        require(not bool((e[0]==e[1]).any()),'Self edges not allowed in energy graph')
        pairs=[tuple(sorted(x)) for x in e.T.tolist()]
        require(len(pairs)==len(set(pairs)),'Duplicate internal edge')
        for k in ['token_start_s','token_end_s','token_available_s','token_kind']:
            require(a[k].shape==(s,), f'{k} shape mismatch')
        require(bool((a['token_end_s']>=a['token_start_s']).all()),'Negative support duration')
        kinds=self.feature_contract.get('token_kinds',{'0':'forcing','1':'surface observation'})
        require(set(a['token_kind'].tolist())<={int(k) for k in kinds},'Undeclared external token kind')
        require(bool((a['token_available_s']>=a['token_end_s']).all()),
                'Retrospective observed forcing/event cannot be available before support ends')
        knots=a['block_knots_s']; pt=a['physics_times_s']
        require(knots.ndim==1 and len(knots)>=2 and knots[0]==0 and np.all(np.diff(knots)>0),
                'Blocks must start at zero and increase')
        require(pt.ndim==1 and len(pt)>=2 and pt[0]==0 and pt[-1]==knots[-1] and np.all(np.diff(pt)>0),
                'Physical quadrature must cover full interval')
        require(set(knots.tolist())<=set(pt.tolist()),'Physical quadrature must include block boundaries')
        k=int(meta['columns']); nz=int(meta['nodes_per_column'])
        require(k*nz==n,'Column node ordering mismatch')
        for c in range(k):
            require(np.all(np.diff(a['node_xyz_m'][c*nz:(c+1)*nz,2])>0),'Depths ordered within columns')
        require(meta.get('spatial_basis') in {'piecewise_constant_cell_means','piecewise_linear_mass_lumped'},
                'Use one declared basis for both observation and energy')
        if self.kind=='real' and meta['spatial_basis']=='piecewise_linear_mass_lumped':
            require(meta.get('basis_mass_weights_verified') is True,'P1 mass quadrature/basis consistency unverified')
        require(meta.get('input_semantics_certified') is True,'Input time/units/missingness not certified')
        require(set(meta.get('required_physics',[])),'Declare required physics even for data-only model')
        if 'spatial' in ref.pools:
            require(bool(meta.get('spatial_evidence')),'Spatial pool requires actual evidence')
        return ExternalEpisode(ref,a,meta)

    def targets(self, ref: EpisodeRef, allowed_roles: set[str]) -> list[dict]:
        require(ref.role!='deployment', 'Deployment inputs have no target reader')
        require(ref.role in allowed_roles, f'Unauthorized target read: {ref.role}/{ref.episode_id}')
        self._verify_file(ref.targets_path)
        obj=read_json(ref.targets_path)
        require(obj['episode_id']==ref.episode_id,'Target ID mismatch')
        ext_meta=read_json(ref.metadata_path)
        n=ext_meta['columns']*ext_meta['nodes_per_column']; stop=ext_meta['duration_s']
        events=obj['events']; ids=set()
        for ev in events:
            validate_event(ev,n,stop)
            require(ev['event_id'] not in ids,'Duplicate event within episode');ids.add(ev['event_id'])
        return events

    def audit(self, verify_file_hashes: bool=True) -> dict:
        counts={r:0 for r in ROLES}; pools={p:0 for p in POOLS}; failures=[]
        for r in self.refs:
            counts[r.role]+=1
            for p in r.pools:pools[p]+=1
            try:
                self.external(r)
                # Labels stay unopened. Hashing is off for protected labels unless explicitly enabled.
                if verify_file_hashes:
                    expected=self.document.get('file_hashes',{}).get(str(r.external_path.relative_to(self.root)))
                    require(expected is not None, 'Missing external file SHA')
                    require(expected==sha256(r.external_path),'External file changed')
            except Exception as exc: failures.append({'episode_id':r.episode_id,'error':str(exc)})
        return {'data_kind':self.kind,'episodes':len(self.refs),'roles':counts,'capability_counts':pools,
                'failures':failures,'passed':not failures,
                'note':'Metadata/external audit only; not a proof of physical identity or target QC.'}

def validate_event(ev: dict,n: int,stop: float):
    required={'event_id','raw_record_ids','lineage_id','variable','unit','time_kind','times_s',
              'weights','values','covariance','basis','depths_m','support_certified'}
    require(required<=ev.keys(),f'Missing observation contract fields: {required-ev.keys()}')
    require(ev['support_certified'] is True,'Uncertified observation support')
    require(ev['time_kind'] in TIME_KINDS,'Invalid time kind')
    ts=np.asarray(ev['times_s'],float); w=np.asarray(ev['weights'],float)
    y=np.asarray(ev['values'],float); R=np.asarray(ev['covariance'],float); basis=np.asarray(ev['basis'],float)
    for k,x in [('times',ts),('weights',w),('values',y),('R',R),('basis',basis)]:finite(k,x)
    d=len(y); require(d>0 and ts.ndim==1 and len(ts)>0,'Empty event')
    require(np.all(ts>=0) and np.all(ts<=stop),'Query outside certified segment')
    require(len(set(ts.tolist()))==len(ts),'Repeated time quadrature/candidate')
    require(w.shape==ts.shape and np.all(w>=0),'Bad weights')
    if ev['time_kind']=='interval_total':
        require(ev.get('integral_weight_unit')=='s' and w.sum()>0,'Totals use duration weights in seconds')
    else: require(np.isclose(w.sum(),1.0),'Mean/clock weights must sum to one')
    if ev['time_kind']=='instant':require(len(ts)==1,'Known instant uses one time')
    require(basis.shape==(d,n) and np.all(basis>=0) and np.allclose(basis.sum(1),1),
            'Observation basis must be positive rows summing to one; no extrapolation')
    require(R.shape==(d,d) and np.allclose(R,R.T),'Covariance must be [D,D] symmetric')
    try: np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc: raise ContractError('Covariance not positive definite') from exc
    require(len(ev['depths_m'])==d and len(ev['raw_record_ids'])==d,'Depth/record shape mismatch')
    require(len(set(ev['raw_record_ids']))==d,'Duplicated raw record in cast')
    if ev['variable']=='water_temperature_C':
        require(ev['unit']=='degC','Water temperature unit must be degC')
        require(ev['time_kind']!='interval_total','A temperature integral needs a distinct observable/unit; not degC')
    if ev['time_kind']=='unknown_instant_within_interval':
        require(ev.get('clock_scope')=='shared_cast','One unknown time shared by complete cast')
        require(bool(ev.get('clock_prior_source')),'Frozen sampling-protocol prior required')
    for pair in ev.get('contrast_pairs',[]):
        require(len(pair)==3 and 0<=pair[0]<d and 0<=pair[1]<d and pair[2]>0,'Invalid contrast pair')
