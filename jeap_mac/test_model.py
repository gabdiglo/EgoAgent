import json
from pathlib import Path
import torch
from model import JEAP

def main():
    torch.manual_seed(42);torch.set_num_threads(2)
    m=JEAP();m.eval();x=torch.randn(2,3,3,64,64);a=torch.randn(2,3,17,3);mask=torch.ones(2,3,17)
    with torch.no_grad():
        before=m(x,a,mask);changed=a.clone();changed[:,1]+=2;after=m(x,changed,mask)
        assert torch.allclose(before['actions'][:,1],after['actions'][:,1],atol=1e-6),'Action query leakage'
        assert not torch.allclose(before['state_logits'][:,1],after['state_logits'][:,1]),'State ignores action'
        future=x.clone();future[:,2]+=3;later=m(future,a,mask)
        for key in ['state_logits','actions']:assert torch.allclose(before[key][:,:2],later[key][:,:2],atol=1e-6),'Future image leakage'
    m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    old={k:v.detach().clone() for k,v in m.observer.named_parameters()}
    loss,parts,teacher=m.losses(torch.randn(2,4,3,64,64),x,x+.1,a,mask);loss.backward()
    assert all(p.grad is None for p in m.observer.parameters())
    for params in [m.predictor.backbone.parameters(),m.action_head.parameters(),m.action_conv.parameters()]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    opt.step();m.update_observer(teacher)
    for name,p in m.observer.named_parameters():
        expected=old[name]*m.c.ema+dict(m.predictor.named_parameters())[name].detach()*(1-m.c.ema)
        assert torch.allclose(p,expected,atol=1e-6),name
    assert any(not torch.equal(old[n],p) for n,p in m.observer.named_parameters())
    p=Path(__file__).parent/'test_checkpoint.pt';torch.save(m.state_dict(),p);other=JEAP();other.load_state_dict(torch.load(p,weights_only=True));p.unlink()
    m.eval();other.eval()
    assert torch.allclose(m(x,a,mask)['actions'],other(x,a,mask)['actions'])
    result={'test_type':'synthetic architectural checks; not scientific performance','official_backbone_class':type(m.predictor.backbone).__name__,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'total_parameters_including_observer':sum(p.numel() for p in m.parameters()),'checks':['action-query causality','action-conditioned state','no future-image leakage','observer stop-gradient','backbone/action gradients','exact EMA update','checkpoint round trip'],'synthetic_losses':{k:float(v.detach()) for k,v in parts.items()}}
    (Path(__file__).parent/'test_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
