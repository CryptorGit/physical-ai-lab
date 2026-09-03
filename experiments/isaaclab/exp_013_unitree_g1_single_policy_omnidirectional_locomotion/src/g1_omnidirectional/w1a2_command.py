"""Fixed, boundary-derived W1A2 continuous direction curriculum."""
import json, math
from pathlib import Path
import torch
from g1_omnidirectional.w1a_command import W1AContinuousTranslationCommand

class W1A2SpeedEnvelopeCommand(W1AContinuousTranslationCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        repo=Path(__file__).resolve().parents[5]
        path=repo/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/resolved_w1a2_sector_curriculum.json"
        data=json.loads(path.read_text(encoding="utf-8"))
        keys=list(data["sectors"])
        self.sectors=torch.tensor([float(k) for k in keys],device=self.device)*math.pi/180
        self.e1_max=torch.tensor([data["sectors"][k]["E1_max"] for k in keys],device=self.device)
    @property
    def phase(self):
        return ("E1_BOUNDARY_CONSOLIDATION" if self.training_iteration<=40 else
                "E2_0P5_EXPANSION" if self.training_iteration<=80 else
                "E3_0P6_EXPANSION" if self.training_iteration<=130 else
                "E4_ALL_DIRECTION_CONSOLIDATION")
    def _resample_command(self,env_ids):
        n=len(env_ids)
        if n==0:return
        u=torch.rand(n,device=self.device); theta=torch.empty(n,device=self.device); speed=torch.empty(n,device=self.device)
        a=u<.20; b=(u>=.20)&(u<.70); c=(u>=.70)&(u<.90); d=u>=.90
        theta[a]=torch.rand(int(a.sum()),device=self.device)*2*math.pi-math.pi
        speed[a]=.25+torch.rand(int(a.sum()),device=self.device)*.10
        if b.any():
            idx=torch.randint(len(self.sectors),(int(b.sum()),),device=self.device)
            centers=self.sectors[idx]; theta[b]=centers+(torch.rand(int(b.sum()),device=self.device)-.5)*(math.pi/8)
            if self.training_iteration<=40:
                maxima=self.e1_max[idx]
            elif self.training_iteration<=80: maxima=torch.full((int(b.sum()),),.50,device=self.device)
            else: maxima=torch.full((int(b.sum()),),.60,device=self.device)
            speed[b]=.30+torch.rand(int(b.sum()),device=self.device)*(maxima-.30)
        if c.any():
            m=int(c.sum()); anchor=torch.rand(m,device=self.device)
            diag=anchor>=.5; signs=torch.where(torch.rand(m,device=self.device)<.5,-1.,1.)
            theta[c]=torch.where(diag,signs*(math.pi/4)*(0.75+torch.rand(m,device=self.device)*.25),torch.zeros(m,device=self.device))
            speed[c]=torch.where(diag,.6+torch.rand(m,device=self.device)*.4,.6+torch.rand(m,device=self.device)*.6)
        if d.any():
            ids=torch.where(d)[0]; pairs=(len(ids)//2)*2
            base=torch.rand(pairs//2,device=self.device)*math.pi
            pair_theta=torch.stack((base,-base),dim=1).flatten(); pair_speed=.3+torch.rand(pairs//2,device=self.device)*.3
            theta[ids[:pairs]]=pair_theta; speed[ids[:pairs]]=torch.stack((pair_speed,pair_speed),dim=1).flatten()
            if pairs<len(ids): theta[ids[-1]]=0; speed[ids[-1]]=.6
        if self.training_iteration>130:
            target=torch.rand(n,device=self.device)<.55
            theta[target]=torch.rand(int(target.sum()),device=self.device)*2*math.pi-math.pi
            speed[target]=.55+torch.rand(int(target.sum()),device=self.device)*.05
        self.vel_command_b[env_ids,0]=speed*torch.cos(theta)
        self.vel_command_b[env_ids,1]=speed*torch.sin(theta)
        self.vel_command_b[env_ids,2]=0
