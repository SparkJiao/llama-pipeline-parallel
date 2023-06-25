# llama-pipeline-parallel

This is a experimental repo to explore how to implemente LLaMA with Deepspeed Pipeline Parallelism since the document is incomplete and very few projects are working on this. The repo hopes to provide a minimal prototype and training loop to implemement PP training for LLaMA and keep a note of possible bugs and the corresponding solutions.

### Updates

2023/06/25: Repo established. Add some notes first and the code will soon be released when the clear is ready.

### Quick Notes

#### Data fetech

1. Currently most implementations uses `shuffle=True` instead of `DistributedSampler` or `RandomSampler` of pytorch in data loader. I find that for `wordld_size=4` scenario, only the first rank and the last one fetech data from data loader. This can be verified by adding print information in `__getitem__` method of specific dataset. However, when really training, I find that only the batch feteched from the first rank will be really send to model. This is consistent with what I thought about pipeline parallelism that only one rank feteches data and the other ranks only take the outputs from the previous rank as iputs.
2. There is a bug in Deepspeed hybrid engine loading model checkpoint that there mush be optimizer states in the specific dir, check it [here](https://github.com/HuangLK/llama-deepspeed/issues/28).
