"""Auditable cohort storage for transition-only PPO segments."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class SegmentStep:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    log_prob: torch.Tensor

class TransitionOnlyStorage:
    """Contains no source-preparation prefix by construction."""
    def __init__(self, cohort_size:int, gamma=.99, lam=.95):
        self.cohort_size=cohort_size;self.gamma=gamma;self.lam=lam;self.steps=[];self.segment_started=False
    def begin(self, source_ready:torch.Tensor):
        if source_ready.shape != (self.cohort_size,) or not bool(source_ready.all()):
            raise ValueError("every cohort member must satisfy the source contract")
        self.steps.clear();self.segment_started=True
    def add(self, step:SegmentStep):
        if not self.segment_started: raise RuntimeError("source prefix cannot enter storage")
        if step.observation.shape[0]!=self.cohort_size: raise ValueError("cohort mismatch")
        self.steps.append(step)
    def finish(self,last_value:torch.Tensor):
        T=len(self.steps)
        adv=torch.zeros(T,self.cohort_size,device=last_value.device,dtype=last_value.dtype)
        gae=torch.zeros(self.cohort_size,device=last_value.device,dtype=last_value.dtype);nextv=last_value
        for t in range(T-1,-1,-1):
            s=self.steps[t]; boundary=s.terminated|s.truncated
            bootstrap=torch.where(boundary,torch.zeros_like(nextv),nextv)
            delta=s.reward+self.gamma*bootstrap-s.value
            gae=delta+self.gamma*self.lam*torch.where(boundary,torch.zeros_like(gae),gae)
            adv[t]=gae;nextv=s.value
        returns=adv+torch.stack([s.value for s in self.steps])
        return returns,adv

class TransitionOnlyOnPolicyRunner:
    """Two-phase runner boundary; preparation calls are never stored."""
    def __init__(self,cohort_size,gamma=.99,lam=.95):
        self.storage=TransitionOnlyStorage(cohort_size,gamma,lam);self.physical_steps=0;self.source_steps=0;self.transition_steps=0
    def preparation_step(self):
        self.physical_steps+=1;self.source_steps+=1
    def start_transition(self,source_ready):
        self.storage.begin(source_ready)
    def transition_step(self,step):
        self.physical_steps+=1;self.transition_steps+=1;self.storage.add(step)
