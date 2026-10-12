"""Direct state-conditioned time-block inverse operator, not a time-step lake solver.

Water h and raw auxiliary channels are decoded by a qualified differentiable physics
adapter. Query and budget use the same cell-mean field. No target tensors enter here.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn, Tensor
from .batch import Batch

@dataclass
class ModelConfig:
    node_features: int
    token_features: int
    token_kinds: int=2
    width: int=64
    blocks: int=4
    heads: int=4
    spectral_modes: int=6
    aux_channels: int=0
    enthalpy_scale_J_kg: float=41800.0
    time_scale_s: float=86400.0
    process_control_amplitude: float=1.0
    # Optional initialization recipe, frozen from TRAIN external forcing only.
    # These set random output-head amplitudes once; the forward has no clamp.
    output_head_init_scale: float | None=None
    initial_temperature_reference_C: float | None=None
    initial_aux_reference: tuple[float,...] | None=None

@dataclass
class State:
    water_h: Tensor             # [B,N], J/kg
    aux_raw: Tensor             # [B,N,A], adapter-defined, no hidden temperature input
    memory: Tensor              # [B,N,W]

@dataclass
class RawField:
    water_h: Tensor             # [B,Q,N]
    aux_raw: Tensor             # [B,Q,N,A]
    memory: Tensor              # [B,Q,N,W]
    process_control: Tensor     # [B,Q,N], bounded correction, NOT a free heat source

class ReflectedSpectral(nn.Module):
    """Depth-only feature kernel on a reflected extension; no periodic water boundary.

    Actual boundary/physics remains the adapter's responsibility. Resolution
    convergence must be measured; reflected feature FFT is not an EOS/PDE solver.
    """
    def __init__(self,width:int,modes:int):
        super().__init__();self.modes=modes
        scale=1/(width**0.5)
        self.real=nn.Parameter(torch.randn(modes,width,width)*scale)
        self.imag=nn.Parameter(torch.randn(modes,width,width)*scale)
    def forward(self,x:Tensor,columns:int,levels:int)->Tensor:
        b,q,n,w=x.shape
        v=x.reshape(b*q*columns,levels,w).transpose(1,2)
        ext=torch.cat([v,torch.flip(v,[-1])],dim=-1)
        freq=torch.fft.rfft(ext,dim=-1,norm='ortho')
        m=min(self.modes,freq.shape[-1])
        out=torch.zeros_like(freq)
        weights=torch.complex(self.real[:m],self.imag[:m])
        out[:,:,:m]=torch.einsum('bim,mio->bom',freq[:,:,:m],weights)
        y=torch.fft.irfft(out,n=2*levels,dim=-1,norm='ortho')[...,:levels]
        return y.transpose(1,2).reshape(b,q,n,w)

class GeometryBlock(nn.Module):
    def __init__(self,width:int,modes:int):
        super().__init__()
        self.norm=nn.LayerNorm(width);self.local=nn.Linear(width,width)
        self.graph=nn.Linear(width,width,bias=False)
        self.spectral=ReflectedSpectral(width,modes) if modes else None
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
    def forward(self,x:Tensor,batch:Batch)->Tensor:
        y=self.norm(x)
        message=torch.einsum('bij,bqjw->bqiw',batch.adjacency,y)
        d=self.local(y)+self.graph(message)
        if self.spectral is not None:d=d+self.spectral(y,batch.columns,batch.levels)
        x=x+0.2*torch.nn.functional.gelu(d)
        return x+0.2*self.ff(self.norm(x))

class DirectInverseOperator(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__();self.cfg=cfg;w=cfg.width
        if w%cfg.heads:raise ValueError('width must be divisible by heads')
        self.register_buffer('node_mean',torch.zeros(cfg.node_features))
        self.register_buffer('node_std',torch.ones(cfg.node_features))
        self.register_buffer('token_mean',torch.zeros(cfg.token_features))
        self.register_buffer('token_std',torch.ones(cfg.token_features))
        self.node_encoder=nn.Sequential(nn.Linear(cfg.node_features,w),nn.GELU(),nn.Linear(w,w))
        self.token_encoder=nn.Sequential(nn.Linear(cfg.token_features+4,w),nn.GELU(),nn.Linear(w,w))
        self.kind_embedding=nn.Embedding(cfg.token_kinds,w)
        self.null_token=nn.Parameter(torch.zeros(1,1,w))
        self.time_encoder=nn.Sequential(nn.Linear(5,w),nn.GELU(),nn.Linear(w,w))
        self.attention=nn.MultiheadAttention(w,cfg.heads,batch_first=True,dropout=0.0)
        self.blocks=nn.ModuleList([GeometryBlock(w,cfg.spectral_modes) for _ in range(cfg.blocks)])
        self.state_encoder=nn.Linear(1+cfg.aux_channels+w,w)
        self.initial_shape=nn.Linear(w,1);self.initial_mean=nn.Linear(w,1)
        self.initial_aux=nn.Linear(w,cfg.aux_channels) if cfg.aux_channels else None
        self.delta_shape=nn.Linear(w,1);self.delta_mean=nn.Linear(w,1)
        self.delta_aux=nn.Linear(w,cfg.aux_channels) if cfg.aux_channels else None
        self.delta_memory=nn.Linear(w,w);self.process=nn.Linear(w,1)
        if cfg.output_head_init_scale is not None:
            if not 0<cfg.output_head_init_scale<=1:
                raise ValueError('Output-head initialization scale must be in (0,1]')
            with torch.no_grad():
                for head in (self.initial_shape,self.initial_mean,self.delta_shape,self.delta_mean,
                             self.initial_aux,self.delta_aux,self.delta_memory):
                    if head is not None:
                        head.weight.mul_(cfg.output_head_init_scale)
                        head.bias.mul_(cfg.output_head_init_scale)
        if cfg.initial_temperature_reference_C is not None:
            if not math.isfinite(cfg.initial_temperature_reference_C):
                raise ValueError('External initial reference must be finite')
            with torch.no_grad():
                self.initial_mean.bias.fill_(4180.*cfg.initial_temperature_reference_C/cfg.enthalpy_scale_J_kg)
        if cfg.initial_aux_reference is not None:
            if len(cfg.initial_aux_reference)!=cfg.aux_channels:
                raise ValueError('Initial auxiliary reference differs from actual layout')
            if self.initial_aux is not None:
                with torch.no_grad():
                    self.initial_aux.bias.copy_(torch.as_tensor(cfg.initial_aux_reference))

    def set_normalization(self,node_mean,node_std,token_mean,token_std):
        for name,val in [('node_mean',node_mean),('node_std',node_std),('token_mean',token_mean),('token_std',token_std)]:
            target=getattr(self,name);v=torch.as_tensor(val,dtype=target.dtype,device=target.device)
            if target.shape!=v.shape:raise ValueError('Normalization shape mismatch')
            target.copy_(v)

    def _tokens(self,b:Batch)->Tensor:
        scale=self.cfg.time_scale_s
        extra=torch.stack([b.token_start/scale,b.token_end/scale,
                           (b.token_end-b.token_start)/scale,b.token_available/scale],-1).float()
        z=torch.cat([(b.tokens-self.token_mean)/self.token_std,extra],-1)
        y=self.token_encoder(z)+self.kind_embedding(b.token_kind)
        return torch.cat([self.null_token.expand(b.size,-1,-1),y],1)

    def _context(self,b:Batch,times:Tensor,cutoff_s:float,mode:str)->Tensor:
        if mode not in {'historical','causal','forcing_only'}:raise ValueError('Unknown information mode')
        if mode!='historical' and any(e.meta.get('public_support',{}).get('historical_chemistry_priors') for e in b.external):
            raise ValueError('Undated published chemistry priors are retrospective historical inputs only; causal and forcing-only availability is uncertified')
        node=self.node_encoder((b.node_features-self.node_mean)/self.node_std)
        q=len(times);ts=times/self.cfg.time_scale_s
        time_f=torch.stack([ts,torch.sin(ts/365.25*2*math.pi),torch.cos(ts/365.25*2*math.pi),
                            torch.zeros_like(ts),torch.ones_like(ts)],-1).float()
        query=node[:,None]+self.time_encoder(time_f)[None,:,None,:]
        # Keys are independently encoded: no future-contaminating bidirectional token self-attention.
        keys=self._tokens(b)
        cap=times if mode=='causal' else torch.full_like(times,cutoff_s)
        cap=torch.minimum(cap,torch.full_like(cap,cutoff_s))
        valid=(b.token_valid[:,None,:] & (b.token_available[:,None,:]<=cap[None,:,None])
               & (b.token_end[:,None,:]<=cap[None,:,None]))
        if mode=='forcing_only':valid=valid & (b.token_kind[:,None,:]==0)
        valid=torch.cat([torch.ones(b.size,q,1,dtype=torch.bool,device=b.device),valid],-1)
        mask=(~valid)[:,:,None,:].expand(-1,-1,b.nodes,-1).reshape(b.size,q*b.nodes,-1)
        mask=mask.repeat_interleave(self.cfg.heads,dim=0)
        y,_=self.attention(query.reshape(b.size,q*b.nodes,-1),keys,keys,
                           attn_mask=mask,need_weights=False)
        return query+y.reshape(b.size,q,b.nodes,-1)

    def _mix(self,x:Tensor,b:Batch)->Tensor:
        for block in self.blocks:x=block(x,b)
        return x

    def _energy_shape(self,x:Tensor,b:Batch,mean_head,shape_head)->Tensor:
        mass=b.mass.to(x.dtype);weight=mass/mass.sum(-1,keepdim=True)
        pooled=torch.einsum('bn,bqnw->bqw',weight,x)
        mean=mean_head(pooled).squeeze(-1)
        shape=shape_head(x).squeeze(-1)
        shape=shape-(shape*weight[:,None,:]).sum(-1,keepdim=True)
        return (mean[:,:,None]+shape)*self.cfg.enthalpy_scale_J_kg

    def initialize(self,b:Batch,cutoff_s:float,mode:str)->State:
        t=torch.zeros(1,dtype=torch.float64,device=b.device)
        x=self._mix(self._context(b,t,cutoff_s,mode),b)
        h=self._energy_shape(x,b,self.initial_mean,self.initial_shape)[:,0]
        aux=self.initial_aux(x)[:,0] if self.initial_aux else x.new_empty(b.size,b.nodes,0)
        return State(h,aux,x[:,0])

    def decode(self,b:Batch,state:State,times:Tensor,start_s:float,end_s:float,
               cutoff_s:float,mode:str)->RawField:
        if end_s<=start_s:raise ValueError('Nonpositive block length')
        if bool(((times<start_s)|(times>end_s)).any()):raise ValueError('Query outside block')
        frac=((times-start_s)/(end_s-start_s)).float()
        dt=(times-start_s)/self.cfg.time_scale_s
        time_f=torch.stack([dt,torch.sin(dt),torch.cos(dt),
                            torch.full_like(dt,math.log1p((end_s-start_s)/self.cfg.time_scale_s)),
                            frac.double()],-1).float()
        state_in=torch.cat([state.water_h[:,:,None]/self.cfg.enthalpy_scale_J_kg,
                            state.aux_raw,state.memory],-1)
        x=self._context(b,times,cutoff_s,mode)
        x=x+self.state_encoder(state_in)[:,None]+self.time_encoder(time_f)[None,:,None,:]
        x=self._mix(x,b)
        dh=self._energy_shape(x,b,self.delta_mean,self.delta_shape)
        h=state.water_h[:,None]+frac[None,:,None]*dh
        if self.delta_aux:
            aux=state.aux_raw[:,None]+frac[None,:,None,None]*self.delta_aux(x)
        else:aux=x.new_empty(b.size,len(times),b.nodes,0)
        mem=state.memory[:,None]+frac[None,:,None,None]*self.delta_memory(x)
        control=self.cfg.process_control_amplitude*torch.tanh(self.process(x).squeeze(-1))
        return RawField(h,aux,mem,control)

    def config_dict(self):return asdict(self.cfg)
