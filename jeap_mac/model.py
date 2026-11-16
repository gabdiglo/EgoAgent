"""Small JEAP adaptation with the official InternLMModel initialized from scratch."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from vendor.configuration_internlm import InternLMConfig
from vendor.modeling_internlm import InternLMModel

@dataclass
#configurazione ridotta del modello:
#Per rendere l’esperimento eseguibile su Mac, la configurazione è stata ridotta a due layer Transformer, 
# quattro teste di attenzione e hidden size 128. 
# L’architettura conserva però gli elementi concettuali della JEAP.
class Config:
    image_size:int=64
    patch_size:int=16
    hidden_size:int=128
    intermediate_size:int=352
    layers:int=2
    heads:int=4
    prototypes:int=256
    bottleneck:int=64
    joints:int=17
    ema:float=.996
    center_momentum:float=.9
    student_temperature:float=.1
    teacher_temperature:float=.04

class DINOHead(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(c.hidden_size,c.hidden_size),nn.GELU(),nn.Linear(c.hidden_size,c.bottleneck))
        self.last=nn.utils.weight_norm(nn.Linear(c.bottleneck,c.prototypes,bias=False))
        self.last.weight_g.data.fill_(1);self.last.weight_g.requires_grad_(False)
    def forward(self,x):return self.last(F.normalize(self.mlp(x),dim=-1))

#proiezione visiva e backbone InternLM
#L’immagine viene suddivisa in patch attraverso una proiezione convoluzionale. 
# I token visivi e la state query vengono elaborati dal backbone InternLM, inizializzato da zero e utilizzato con input embedding continui.
#non vengono usati pesi linguistici pre-addestrati.
class VisualBranch(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.patch=nn.Conv2d(3,c.hidden_size,c.patch_size,stride=c.patch_size)
        self.q_state=nn.Parameter(torch.randn(1,1,c.hidden_size)*.02)
        self.backbone=InternLMModel(InternLMConfig(vocab_size=8,hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,num_hidden_layers=c.layers,num_attention_heads=c.heads,max_position_embeddings=2048,rotary={'base':10000,'type':'origin'},attn_implementation='eager',use_cache=False))
        self.backbone.embed_tokens.requires_grad_(False) # continuous inputs; language table unused
        self.head=DINOHead(c)
    def patches(self,x):
        if x.shape[-2:]!=(self.c.image_size,self.c.image_size):raise ValueError('Unexpected image resolution')
        return self.patch(x).flatten(2).transpose(1,2)
    def encode(self,x):
        tokens=torch.cat([self.patches(x),self.q_state.expand(len(x),-1,-1)],dim=1)
        z=self.backbone(inputs_embeds=tokens,use_cache=False).last_hidden_state[:,-1]
        return self.head(z),z

class JEAP(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.c=c or Config();c=self.c
        self.predictor=VisualBranch(c);self.observer=VisualBranch(c)
        self.observer.load_state_dict(self.predictor.state_dict());self.observer.requires_grad_(False);self.observer.eval()
        self.q_action=nn.Parameter(torch.randn(1,1,c.hidden_size)*.02)
        self.action_conv=nn.Conv1d(c.joints*3,c.hidden_size,1)
        self.action_norm=nn.LayerNorm(c.hidden_size)
        self.action_head=nn.Sequential(nn.Linear(c.hidden_size,c.hidden_size),nn.GELU(),nn.Linear(c.hidden_size,c.joints*3))
        self.register_buffer('center',torch.zeros(1,c.prototypes))
    def train(self,mode=True):
        super().train(mode);self.observer.eval();return self
    def forward(self,frames,actions=None,action_mask=None):
        b,t=frames.shape[:2]
        patches=self.predictor.patches(frames.flatten(0,1)).reshape(b,t,-1,self.c.hidden_size)
        if actions is not None:
            if actions.shape!=(b,t,self.c.joints,3):raise ValueError('Action shape mismatch')
            a=actions if action_mask is None else actions*action_mask[...,None]
            a=self.action_conv(a.reshape(b,t,-1).transpose(1,2)).transpose(1,2)
            a=F.gelu(self.action_norm(a))
        chunks=[];qa=[];qs=[];offset=0
        for k in range(t):

            #Per ogni istante vengono inseriti nell’ordine i token dell’immagine, la action query, il token dell’azione e la state query. 
            # La action query precede l’azione corrente, impedendo che il target sia visibile durante la previsione; 
            # la state query segue invece l’azione, permettendo di condizionare la previsione dello stato su di essa.
            chunks.append(patches[:,k]);offset+=patches.shape[2]
            if actions is not None:
                chunks.append(self.q_action.expand(b,-1,-1));qa.append(offset);offset+=1
                chunks.append(a[:,k:k+1]);offset+=1
            chunks.append(self.predictor.q_state.expand(b,-1,-1));qs.append(offset);offset+=1
        z=self.predictor.backbone(inputs_embeds=torch.cat(chunks,dim=1),use_cache=False).last_hidden_state
        states=z[:,qs];result={'state_logits':self.predictor.head(states),'state_features':states}
        if actions is not None:result['actions']=self.action_head(z[:,qa]).reshape(b,t,self.c.joints,3)
        return result
    def dino(self,student,teacher):
        target=F.softmax((teacher.detach()-self.center)/self.c.teacher_temperature,dim=-1)
        return -(target*F.log_softmax(student/self.c.student_temperature,dim=-1)).sum(-1).mean()
    def losses(self,frames,view1,view2,actions=None,action_mask=None):
        # T+1 frames, T actions. Future frame is only passed to the observer.
        prediction=self(frames[:,:-1],actions,action_mask)
        s1,_=self.predictor.encode(view1.flatten(0,1));s2,_=self.predictor.encode(view2.flatten(0,1))
        with torch.no_grad():
            target,_=self.observer.encode(frames[:,1:].flatten(0,1))
            t1,_=self.observer.encode(view1.flatten(0,1));t2,_=self.observer.encode(view2.flatten(0,1))
        rep=.5*(self.dino(s1,t2)+self.dino(s2,t1))
        pred=self.dino(prediction['state_logits'].flatten(0,1),target)
        act=rep.new_zeros(())
        if actions is not None:
            mask=torch.ones_like(actions[...,0]) if action_mask is None else action_mask
            if mask.sum().item()==0:raise ValueError('No observed joints')
            act=((prediction['actions']-actions).abs()*mask[...,None]).sum()/(3*mask.sum())
        return 2*rep+pred+3*act,{'representation':rep,'prediction':pred,'action':act},torch.cat([t1,t2,target]).detach()
    @torch.no_grad()
    def update_observer(self,teacher_logits):
        source=dict(self.predictor.named_parameters())
        for name,p in self.observer.named_parameters():p.mul_(self.c.ema).add_(source[name],alpha=1-self.c.ema)
        buffers=dict(self.predictor.named_buffers())
        for name,b in self.observer.named_buffers():
            if name in buffers and b.shape==buffers[name].shape:b.copy_(buffers[name])
        self.center.mul_(self.c.center_momentum).add_(teacher_logits.mean(0,keepdim=True),alpha=1-self.c.center_momentum)
    def configuration(self):return asdict(self.c)
