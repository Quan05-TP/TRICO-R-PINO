from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from .contracts import ExternalEpisode, require

@dataclass
class Batch:
    node_features: Tensor
    mass: Tensor
    xyz: Tensor
    adjacency: Tensor
    edges: Tensor
    tokens: Tensor
    token_start: Tensor
    token_end: Tensor
    token_available: Tensor
    token_kind: Tensor
    token_valid: Tensor
    columns: int
    levels: int
    external: list[ExternalEpisode]

    @property
    def size(self): return self.mass.shape[0]
    @property
    def nodes(self): return self.mass.shape[1]
    @property
    def device(self): return self.mass.device
    def physics_array(self,name: str,dtype=None) -> Tensor:
        vals=[e.arrays[name] for e in self.external]
        require(len({v.shape for v in vals})==1,f'Physics array shape mismatch: {name}; use tighter bucket')
        return torch.as_tensor(np.stack(vals),device=self.device,dtype=dtype or self.mass.dtype)

def bucket_key(e: ExternalEpisode) -> tuple:
    a=e.arrays; m=e.meta
    return (m['columns'],m['nodes_per_column'],a['node_features'].shape[1],a['tokens'].shape[1],
            tuple(a['edge_index'].ravel().tolist()),tuple(a['block_knots_s'].tolist()),
            tuple(a['physics_times_s'].tolist()),tuple(sorted(m['required_physics'])),m.get('state_layout_id',''))

def collate(episodes: list[ExternalEpisode],device='cpu') -> Batch:
    require(bool(episodes),'Empty batch')
    require(len({bucket_key(e) for e in episodes})==1,'Batch must have same topology/block/physics layout')
    b=len(episodes); n=episodes[0].arrays['node_mass_kg'].size
    f=episodes[0].arrays['tokens'].shape[1]; s=max(len(e.arrays['tokens']) for e in episodes)
    tok=np.zeros((b,s,f),np.float32); start=np.zeros((b,s)); end=start.copy(); av=start.copy()
    kind=np.zeros((b,s),np.int64); valid=np.zeros((b,s),bool)
    for i,e in enumerate(episodes):
        k=len(e.arrays['tokens']);tok[i,:k]=e.arrays['tokens']
        start[i,:k]=e.arrays['token_start_s'];end[i,:k]=e.arrays['token_end_s']
        av[i,:k]=e.arrays['token_available_s'];kind[i,:k]=e.arrays['token_kind'];valid[i,:k]=True
    edge=episodes[0].arrays['edge_index'].astype(np.int64)
    # Geometry-aware dimensionless feature averaging. This is NOT a heat flux matrix.
    adj=np.zeros((b,n,n),np.float32)
    for i,e in enumerate(episodes):
        m=e.arrays['node_mass_kg']
        adj[i,np.arange(n),np.arange(n)]=1
        for u,v in edge.T:
            adj[i,u,v]=float(m[v]/(m[u]+m[v])); adj[i,v,u]=float(m[u]/(m[u]+m[v]))
        adj[i]/=adj[i].sum(-1,keepdims=True)
    def t(x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=device)
    return Batch(t(np.stack([e.arrays['node_features'] for e in episodes])),
                 t(np.stack([e.arrays['node_mass_kg'] for e in episodes]),torch.float64),
                 t(np.stack([e.arrays['node_xyz_m'] for e in episodes]),torch.float64),
                 t(adj),t(edge,torch.long),t(tok),t(start,torch.float64),t(end,torch.float64),
                 t(av,torch.float64),t(kind,torch.long),t(valid,torch.bool),
                 episodes[0].meta['columns'],episodes[0].meta['nodes_per_column'],episodes)
