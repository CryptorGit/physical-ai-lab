"""In-place physical-env cohort activation without state transfer."""
from __future__ import annotations
import torch

class InPlaceEnvIdCohort:
    def __init__(self,num_envs:int,cohort_size:int,seed:int,device="cpu"):
        if cohort_size>num_envs: raise ValueError("cohort exceeds physical env count")
        self.num_envs=num_envs;self.cohort_size=cohort_size;self.seed=seed;self.generation=0
        self.device=torch.device(device)
        self.source_ready=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.source_ready_since=torch.full((num_envs,),-1,dtype=torch.long,device=self.device)
        self.selected=torch.zeros(num_envs,dtype=torch.bool,device=self.device);self.selected_env_ids=None
        self.segment_id=torch.full((num_envs,),-1,dtype=torch.long,device=self.device)
    def update_ready(self,contract_valid:torch.Tensor,step:int):
        lost=self.source_ready&~contract_valid;self.source_ready[~contract_valid]=False;self.source_ready_since[lost]=-1
        new=contract_valid&~self.source_ready;self.source_ready[new]=True;self.source_ready_since[new]=step
    def activate(self,contract_valid:torch.Tensor,global_previous_action:torch.Tensor):
        eligible=torch.nonzero(self.source_ready&contract_valid).flatten()
        if len(eligible)<self.cohort_size: raise RuntimeError("ready cohort incomplete")
        # CPU generator is intentionally used so the seeded selection is identical
        # across CUDA versions; only the small permutation index crosses devices.
        g=torch.Generator(device="cpu").manual_seed(self.seed+104729*self.generation)
        order=torch.randperm(len(eligible),generator=g)[:self.cohort_size].to(eligible.device)
        ids=eligible[order]
        self.selected.zero_();self.selected[ids]=True;self.selected_env_ids=ids.clone()
        self.segment_id[ids]=self.generation;self.generation+=1
        return {"physical_env_ids":ids,"cohort_previous_action":global_previous_action[ids].clone(),
                "state_copy":False,"setter_calls":0,"teleport_calls":0}
    def gather(self,full_tensor): return full_tensor[self.selected_env_ids]
