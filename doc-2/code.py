// Code used for doc-2 from repo Open-SnowflakeCore

import torch as t
import torch.nn as n
import torch.optim as o
import torch.utils.data as d
from torch.nn.utils.rnn import pad_sequence as ps
from datasets import load_dataset as ld
from transformers import BertTokenizer as bt, PretrainedConfig as pc
import math as m, os
import gc
from safetensors.torch import save_model as sm

msl,dm,nh,nl,fd,bs,ne,lr,sd,gas=384,384,6,4,768,4,20,3e-5,"Open-SnowflakeCore",4
ds=ld("FlameF0X/DialogMLM")
tk=bt.from_pretrained("bert-base-uncased")
vs=tk.vocab_size

def tf(ex):
 return tk(ex["text"],truncation=True,padding="max_length",max_length=msl,return_tensors="pt")

tds=ds.map(tf,batched=True,batch_size=24,remove_columns=["text"])
tds.set_format("torch")
tr=tds["train"]
if "validation" in tds:
 vl=tds["validation"]
else:
 sp=tr.train_test_split(test_size=0.1)
 tr,vl=sp["train"],sp["test"]

def cf(b):
 ii=t.stack([i['input_ids'] for i in b])
 am=t.stack([i['attention_mask'] for i in b])
 return {'input_ids':ii,'attention_mask':am}

trl=d.DataLoader(tr,batch_size=bs,shuffle=True,collate_fn=cf,pin_memory=True,num_workers=1)
vll=d.DataLoader(vl,batch_size=bs,collate_fn=cf,pin_memory=True)

class fqa(n.Module):
 def __init__(s,dm,nh):
  super().__init__()
  s.dm,s.nh,s.hd=dm,nh,dm//nh
  s.qkv=n.Linear(dm,3*dm)
  s.wo=n.Linear(dm,dm)
  n.init.xavier_uniform_(s.qkv.weight)
  n.init.xavier_uniform_(s.wo.weight)
  n.init.zeros_(s.qkv.bias)
  n.init.zeros_(s.wo.bias)
 def forward(s,x,am=None):
  b,sl,_=x.shape
  qkv=s.qkv(x).reshape(b,sl,3,s.nh,s.hd)
  qkv=qkv.permute(2,0,3,1,4)
  q,k,v=qkv[0],qkv[1],qkv[2]
  as_=t.matmul(q,k.transpose(-2,-1))/m.sqrt(s.hd)
  if am is not None:
   am=am.unsqueeze(1).unsqueeze(2)
   as_=as_.masked_fill(am==0,float('-inf'))
  aw=t.softmax(as_,dim=-1)
  c=t.matmul(aw,v)
  c=c.transpose(1,2).reshape(b,sl,s.dm)
  return s.wo(c)

class eff(n.Module):
 def __init__(s,dm,fd,dr=0.1):
  super().__init__()
  s.l1,s.d1,s.l2,s.d2,s.a=n.Linear(dm,fd),n.Dropout(dr),n.Linear(fd,dm),n.Dropout(dr),n.GELU()
  n.init.xavier_uniform_(s.l1.weight)
  n.init.xavier_uniform_(s.l2.weight)
  n.init.zeros_(s.l1.bias)
  n.init.zeros_(s.l2.bias)
 def forward(s,x):
  return s.d2(s.l2(s.d1(s.a(s.l1(x)))))

class stb(n.Module):
 def __init__(s,dm,nh,fd,dr=0.1):
  super().__init__()
  s.att,s.n1,s.d1,s.ff,s.n2,s.d2=fqa(dm,nh),n.LayerNorm(dm,eps=1e-6),n.Dropout(dr),eff(dm,fd,dr),n.LayerNorm(dm,eps=1e-6),n.Dropout(dr)
 def forward(s,x,am=None):
  ai=s.n1(x)
  ao=s.att(ai,am)
  x=x+s.d1(ao)
  fi=s.n2(x)
  fo=s.ff(fi)
  x=x+s.d2(fo)
  return x

class sc(n.Module):
 def __init__(s,vs,msl,dm,nh,nl,fd,dr=0.1):
  super().__init__()
  s.emb=n.Embedding(vs,dm)
  s.pe=n.Parameter(t.zeros(1,msl,dm))
  pos=t.arange(msl).unsqueeze(1).float()
  dt=t.exp(t.arange(0,dm,2).float()*(-m.log(10000.0)/dm))
  penc=t.zeros(1,msl,dm)
  penc[0,:,0::2]=t.sin(pos*dt)
  penc[0,:,1::2]=t.cos(pos*dt)
  s.pe.data=penc.data
  s.lys=n.ModuleList([stb(dm,nh,fd,dr) for _ in range(nl)])
  s.fn,s.dr,s.fc=n.LayerNorm(dm,eps=1e-6),n.Dropout(dr),n.Linear(dm,vs)
  s.fc.weight=s.emb.weight
  n.init.normal_(s.emb.weight,mean=0,std=0.02)
  s.config=pc(vocab_size=vs,hidden_size=dm,num_attention_heads=nh,num_hidden_layers=nl,max_position_embeddings=msl,intermediate_size=fd,model_type="snowflakecore",architectures=["SnowflakeCoreCore"])
 def forward(s,ii,am=None):
  sl=ii.size(1)
  x=s.emb(ii)+s.pe[:,:sl,:]
  x=s.dr(x)
  for ly in s.lys:
   x=ly(x,am)
  x=s.fn(x)
  return s.fc(x)
 def get_input_embeddings(s):
  return s.emb
 def set_input_embeddings(s,emb):
  s.emb=emb
  s.fc.weight=s.emb.weight
 def save_pretrained(s,sv):
  os.makedirs(sv,exist_ok=True)
  s.config.save_pretrained(sv)
  mp=os.path.join(sv,"pytorch_model.bin")
  t.save(s.state_dict(),mp)
  sd=s.half().state_dict()
  md={"format":"pt"}
  sp=os.path.join(sv,"model.safetensors")
  sm(s.half(),sp,metadata=md)

dv=t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {dv}")

mdl=sc(vocab_size=vs,msl=msl,dm=dm,nh=nh,nl=nl,fd=fd,dr=0.1).to(dv)
cr=n.CrossEntropyLoss(ignore_index=tk.pad_token_id)
opt=o.AdamW(mdl.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
scl=t.cuda.amp.GradScaler()
ts=len(trl)*ne
sch=o.lr_scheduler.CosineAnnealingLR(opt,T_max=ts)
st,al=0,0

print(f"Starting training Open-SnowflakeCore model with config: D_MODEL={dm}, HEADS={nh}, LAYERS={nl}, FF_DIM={fd}")

for ep in range(ne):
 mdl.train()
 tl=0
 t.cuda.empty_cache()
 gc.collect()
 print(f"Epoch {ep+1}/{ne} started")
 for bi,bt_ in enumerate(trl):
  ii=bt_['input_ids'].to(dv)
  am=bt_['attention_mask'].to(dv)
  lb=ii.clone()
  with t.cuda.amp.autocast():
   op=mdl(ii,am)
   ls=cr(op.view(-1,vs),lb.view(-1))
   ls=ls/gas
  scl.scale(ls).backward()
  al+=ls.item()
  st+=1
  if st%gas==0 or bi==len(trl)-1:
   t.nn.utils.clip_grad_norm_(mdl.parameters(),max_norm=1.0)
   scl.step(opt)
   scl.update()
   sch.step()
   opt.zero_grad(set_to_none=True)
   tl+=al
   al=0
   if st%(gas*10)==0:
    print(f"  Batch {bi}/{len(trl)}, Loss: {ls.item()*gas:.4f}")
   del ii,am,lb,op,ls
   t.cuda.empty_cache()
 avl=tl/(len(trl)//gas+1)
 print(f"Epoch {ep+1}/{ne}, Train Loss: {avl:.4f}")
 mdl.eval()
 vls=0
 with t.no_grad():
  for bt_ in vll:
   ii=bt_['input_ids'].to(dv)
   am=bt_['attention_mask'].to(dv)
   lb=ii.clone()
   with t.cuda.amp.autocast():
    op=mdl(ii,am)
    ls=cr(op.view(-1,vs),lb.view(-1))
    vls+=ls.item()
   del ii,am,lb,op,ls
   t.cuda.empty_cache()
 avv=vls/len(vll)
 print(f"Epoch {ep+1}/{ne}, Val Loss: {avv:.4f}")

os.makedirs(sd,exist_ok=True)
hfc=pc(vocab_size=vs,hidden_size=dm,num_attention_heads=nh,num_hidden_layers=nl,max_position_embeddings=msl,intermediate_size=fd,model_type="gpt2",architectures=["SnowflakeCoreCore"])
hfc.save_pretrained(sd)
t.save(mdl.state_dict(),os.path.join(sd,"pytorch_model.bin"))
sm(mdl.half(),os.path.join(sd,"model.safetensors"),metadata={"format":"pt"})
tk.save_pretrained(sd)

with open(os.path.join(sd,"README.md"),"w") as f:
 f.write(f"""# Open-SnowflakeCore-G0

[Open-SnowflakeCore-G0](https://github.com/FlameF0X/Open-SnowflakeCore-G0) is a open-sourse pre-train version of SnowflakeCore-G0-Release series.

This is the initial release of the Open-SnowflakeCore series language models, trained on the DialogMLM-50K dataset with optimized memory usage.

## Model details
- Architecture: SnowflakeCore
- Hidden size: {dm}
- Number of attention heads: {nh}
- Number of layers: {nl}
- Feed-forward dimension: {fd}
- Maximum sequence length: {msl}
- Vocabulary size: {vs}

## HuggingFace Transformers Compatibility
This model is fully compatible with the HuggingFace Transformers library. You can load it using:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/Open-SnowflakeCore")
config = AutoConfig.from_pretrained("path/to/Open-SnowflakeCore")
model = AutoModel.from_pretrained("path/to/Open-SnowflakeCore")
```

## Memory Optimization Techniques
- Mixed precision training
- Gradient accumulation ({gas} steps)
- Fused QKV projection
- Pre-norm architecture
- Weight tying between embedding and output layers
- Half-precision model storage

The model weights are stored in both PyTorch (.bin) and safetensors format for improved security, loading efficiency, and compatibility.
""")

print(f"Open-SnowflakeCore model saved in {sd}")