# llama-pipeline-parallel

This is a experimental repo to explore how to implement LLaMA with Deepspeed Pipeline Parallelism since the document is incomplete and very few projects are working on this. The repo hopes to provide a minimal prototype and training loop to implemement PP training for LLaMA and keep a note of possible bugs and the corresponding solutions.

We have provided a minimal template to launch hybrid training of PP and DP, and the config can be found in `conf/llama_65b_metir_v1_pv91_v91_v5_0.yaml`.
It should be noted that the template cannot be directly run since this is extracted from another project and some parts are omitted. 
But you can still quickly adapt it to your own usage by removing the relevant parts of dataset and collator initialization.

## Updates

2023/07/02: Successfully enabling hybrid training of LLaMA-65B on two nodes with 16 * 80G A100.

2023/06/25: Repo established. Add some notes first and the code will soon be released when the clear is ready.

## Core Code Snippets

### Model initialization

There are two main approaches to enable model initialization and loading pre-trained weights. One is first initializing the model using the `from_pretrained` function of HuggingFace's `transformers` repo.
In this case, you may refer to `models.llama_ds_mp_wrap.get_model` for details.
The drawback of this method is that it will load the whole model for each worker. This will cause out-of-CPU-memory-usage when the model is large.
Another method is first initializing the sharded models with DeepSpeed's `LayerSpec` class to implement post-initialization after pipeline parallelism partition. Then each rank only need to load the pre-trained weights for each own partition:

```python
model_or_config = transformers.AutoConfig.from_pretrained(cfg.model_name_or_path)
layers = models.llama_ds_mp_wrap.get_layers_from_config(model_or_config)
model_pipe = PipelineModule(layers=layers,
                            num_stages=cfg.num_stages,
                            loss_fn=models.llama_ds_mp_wrap.loss_fn,
                            activation_checkpoint_interval=getattr(cfg, "activation_checkpoint_interval", 0)
                            )
...
model.load_checkpoint(cfg.model_name_or_path, load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)
```

Note that the pre-trained weights should be converted from HF format by using `convert2ckpt.py`.


### Hybrid Training of Pipeline Parallelism (PP) and Distributed Data Parallel (DP)

When `dist.world_size` > `num_stages`, hybrid training is automatically enabled. The number of stages of pipeline parallel (PP) is `num_stages`
while the degree of data-parallel (DP) is `dist.world_size // num_stages`.

### No Weight Typing of Word Embedding

Different from traditional pre-trained language models, LLaMA do not need weight typing. So do not use `TiedLayerSpec` to wrap `embed_tokens` and `lm_head` modules.

### Distributed Sampler Setting

When hybrid training of PP and DP is enabled, `DistributedSampler` should be carefully set for each rank w.r.t. its state (PP stage and DP group).

The core code snippet is as follows:

```python
dp_degree = dist.get_world_size() // cfg.num_stages

if dp_degree > 1:
    dp_id = model.grid.get_data_parallel_id()
    sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
else:
    sub_train_sampler = RandomSampler(sub_train_dataset)
```

### Data Fetch Design of DeepSpeed and CPU Memory Reduction

In DeepSpeed design, among specific PP group, only the first and the last rank, i.e., `stage=0 or stage=num_stages - 1`, 
will fetch minibatch from dataloader, and the other ranks never fetch data.

Based on this, for the ranks where the dataloader will never be used, we can use placeholders to allocate the memory usage. This could be especially useful when training large models.
For example, when training LLaMA-65B with `offload_optimizer=True` and `num_stages=8`, the CPU memory usage is already nearly 800GB,
which will cause CPU memory OOM when you are using large dataset.

The code of dataset placeholder is as follows:

```python
def load_empty_dataset_and_collator(cfg: DictConfig):
    from data.test import TestDataset
    from data.flan import FlanCollatorOverCollator

    dataset = TestDataset(None, None, getattr(cfg, "total_dataset_len", -1))
    collator = FlanCollatorOverCollator(collator=None,
                                        tokenizer=cfg.model_name_or_path,
                                        max_seq_length=128,
                                        decoder_only=True,
                                        return_standard_inputs=True,
                                        )

    # Keep consistent with `load_and_cache_examples`.
    if getattr(cfg, "dist_load_data_barrier", True):
        dist.barrier()

    if dist.is_initialized():
        dist.barrier()

    return dataset, collator


if model.is_first_stage() or model.is_last_stage():
    sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)

    if dp_degree > 1:
        dp_id = model.grid.get_data_parallel_id()
        sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
    else:
        sub_train_sampler = RandomSampler(sub_train_dataset)
    sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None

    sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                      sampler=sub_train_sampler,
                                      batch_size=cfg.train_batch_size,
                                      collate_fn=sub_train_collator,
                                      num_workers=cfg.num_workers,
                                      pin_memory=True,
                                      prefetch_factor=cfg.prefetch_factor,
                                      drop_last=True,
                                      )
else:
    sub_train_dataset, sub_train_collator = load_empty_dataset_and_collator(cfg)
    sub_train_sampler = None

    sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                      batch_size=cfg.train_batch_size,
                                      collate_fn=sub_train_collator,
                                      drop_last=True,
                                      shuffle=False)

```

where `TestDataset` is an empty dataset and the collator is arbitrary one meeting the input format.

## Know Problems and Possible Solutions

### BF16 Support
Bfloat16 can be used by setting the following in deepspeed config:
```
data_types:
  grad_accum_dtype: "fp32"
```
However, bfloat16 cannot be used with optimizer offload. Note that pipeline parallelism is designed not to support optimizer offload (see issue [\#3866](https://github.com/microsoft/DeepSpeed/issues/3866)). Nevertheless, it can still be enabled under fp16 training.

### Flash Attention

I cannot enable flash attention using both the original implementation or `torch.nn.functional.scaled_dot_product_attention` from pytorch 2.0. See issue [here](https://github.com/HuangLK/llama-deepspeed/issues/36) and [here](https://github.com/microsoft/DeepSpeed/issues/3868).

### Torch Compile

Torch compilation is not supported in the template, which perhaps becuase my writing is incorrect.

## Reference & Acknowledgement

1. [llama-deepspeed](https://github.com/HuangLK/llama-deepspeed/tree/main)
2. [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
3. [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)

[//]: # (### Quick Notes)

[//]: # ()
[//]: # (#### Data fetech)

[//]: # ()
[//]: # (1. Currently most implementations uses `shuffle=True` instead of `DistributedSampler` or `RandomSampler` of pytorch in data loader. I find that for `wordld_size=4` scenario, only the first rank and the last one fetech data from data loader. This can be verified by adding print information in `__getitem__` method of specific dataset. However, when really training, I find that only the batch feteched from the first rank will be really send to model. This is consistent with what I thought about pipeline parallelism that only one rank feteches data and the other ranks only take the outputs from the previous rank as iputs.)

[//]: # (2. There is a bug in Deepspeed hybrid engine loading model checkpoint that there mush be optimizer states in the specific dir, check it [here]&#40;https://github.com/HuangLK/llama-deepspeed/issues/28&#41;.)
