# oci-bloom-finetune
Finetune bloom LLM with Oracle Cloud information

# For multi-gpu training
```bash
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 main.py
    CUDA_VISIBLE_DEVICES=0,1 python main.py
```


