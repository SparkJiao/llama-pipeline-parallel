import logging
from typing import Union, Tuple, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.tokenization_utils import expand_special_tokenizer

logger = logging.getLogger(__name__)


def load_flan_data_w_filter(file_path: str):
    logger.info(f"Loading FLAN data from {file_path}...")
    data = torch.load(file_path, map_location="cpu")
    new_data = []
    cnt = 0
    for item in data:
        if item["inputs"].strip() == "":
            continue
        if item["targets"].strip() == "":
            cnt += 1
            continue
        new_data.append(item)
    logger.info(f"Removed {cnt} empty examples.")
    logger.info(f"Loaded {len(new_data)} examples.")
    return new_data


# def load_gpt4all_data():
#     return load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision='v1.2-jazzy')["train"]


class PromptDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, cfg: DictConfig):
        self.data = hydra.utils.instantiate(cfg, file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": {
                "inputs": self.data[idx]["prompt"],
                "targets": self.data[idx]["response"],
            }
        }


class FLANDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        self.data = load_flan_data_w_filter(file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WikiPathDatasetV5WFlan(Dataset):
    def __init__(self, raw_data: Union[Tuple, DictConfig], flan_file: str, file_path: str, tokenizer: PreTrainedTokenizer):
        # print(type(raw_data))
        if isinstance(raw_data, DictConfig):
            raw_data = hydra.utils.instantiate(raw_data, file_path=file_path, tokenizer=tokenizer)

        self.examples = raw_data[0]
        self.flan_data = load_flan_data_w_filter(flan_file)

    def __len__(self):
        return max(len(self.examples), len(self.flan_data))

    def __getitem__(self, index):
        example = self.examples[index % len(self.examples)]
        flan = self.flan_data[index % len(self.flan_data)]
        # example = self.examples[index]
        # if index >= len(self.flan_data):
        # flan = random.choice(self.flan_data)
        # else:
        #     flan = self.flan_data[index]
        return {
            "example": example,
            "flan": flan,
            "index": index,
        }


class WikiPathDatasetV5WithDataset(Dataset):
    def __init__(self, raw_data: Union[Tuple, DictConfig], extra_data: Union[PromptDataset, DictConfig],
                 file_path: str, tokenizer: PreTrainedTokenizer, add_wiki_text: bool = False):
        if isinstance(raw_data, DictConfig):
            raw_data = hydra.utils.instantiate(raw_data, file_path=file_path, tokenizer=tokenizer)

        if isinstance(extra_data, DictConfig):
            extra_data = hydra.utils.instantiate(extra_data, tokenizer=tokenizer)

        self.examples = raw_data[0]
        self.extra_data = extra_data

        self.add_wiki_text = add_wiki_text
        if self.add_wiki_text:
            self.wiki_texts = raw_data[1]

    def __len__(self):
        return max(len(self.examples), len(self.extra_data))

    def __getitem__(self, index):
        example = self.examples[index % len(self.examples)]
        flan = self.extra_data[index % len(self.extra_data)]
        res = {
            "example": example,
            "index": index,
        }
        res.update(flan)
        if self.add_wiki_text:
            res["text"] = self.wiki_texts[index % len(self.wiki_texts)]
        return res


class FlanCollectionGroupDataset(Dataset):
    def __init__(self, file_path: str, tokenizer=None):
        super().__init__()
        logger.info(f"Loading FLAN data from {file_path}...")
        data = torch.load(file_path, map_location="cpu")
        self.data = []
        cnt = 0
        for item in data:
            if item["inputs"].strip() == "":
                continue
            if item["targets"].strip() == "":
                cnt += 1
                continue
            self.data.append(item)
        logger.info(f"Removed {cnt} empty examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


def vanilla_seq2seq_convertor(examples, tokenizer: PreTrainedTokenizer, max_seq_length, decoder_only: bool = False):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            outputs.append(exp["inputs"] + " " + exp["targets"] + tokenizer.eos_token)
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding="longest",
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="longest",
                                 truncation=True, return_tensors="pt")
        new_input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
        input_lens = input_lens.to(torch.long)
        model_inputs["input_lens"] = input_lens

    return model_inputs


def combine_tensor_on_length(a: torch.Tensor, b: torch.Tensor, pad_id: int):
    max_len = max(a.size(1), b.size(1))
    new_tensor = torch.zeros(a.size(0) + b.size(0), max_len, dtype=a.dtype, device=a.device).fill_(pad_id)
    new_tensor[:a.size(0), :a.size(1)] = a
    new_tensor[a.size(0):, :b.size(1)] = b
    return new_tensor


def get_lm_labels(input_lens, input_ids, pad_token_id, ignore_index=-100):
    labels = input_ids.clone()

    label_mask = labels.ne(pad_token_id)
    lens_mask = torch.arange(labels.size(1))[None, :] >= input_lens[:, None]
    label_mask = label_mask & lens_mask

    labels = labels.masked_fill(~label_mask, ignore_index).contiguous()

    return labels


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            torch.float16,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, torch.float16, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def convert_to_standard_inputs(model_inputs: Dict, tokenizer: PreTrainedTokenizer, ignored_index: int = -100):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    # input_lens = getattr(model_inputs, "input_lens", None)
    input_lens = model_inputs["input_lens"]

    labels = get_lm_labels(input_lens, input_ids, tokenizer.pad_token_id, ignored_index)

    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_decoder_attention_mask(attention_mask, input_ids.shape, 0)

    return input_ids, attention_mask, position_ids, labels


class FlanCollatorOverCollator:
    def __init__(self, collator, tokenizer: str, max_seq_length: int, decoder_only: bool = False, return_standard_inputs: bool = False):
        self.collator = collator
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only
        self.convert_to_standard_inputs = return_standard_inputs

    def __call__(self, batch):
        flan_batch = []
        for item in batch:
            flan_batch.append(item.pop("flan"))

        index = torch.tensor([b["index"] for b in batch], dtype=torch.long)

        if self.collator is not None:
            model_inputs = self.collator(batch)
            orig_batch_size = model_inputs["input_ids"].size(0)
            flan_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)
            for k, v in flan_inputs.items():
                if k == "input_lens":
                    if "flan_input_lens" in model_inputs:
                        model_inputs["flan_input_lens"] = torch.cat([model_inputs["flan_input_lens"], v], dim=0)
                    else:
                        empty_input_lens = torch.zeros(orig_batch_size, dtype=torch.long, device=v.device)
                        model_inputs[f"flan_input_lens"] = torch.cat([empty_input_lens, v], dim=0)
                    continue

                if f"flan_{k}" in model_inputs:
                    model_inputs[f"flan_{k}"] = combine_tensor_on_length(model_inputs[f"flan_{k}"], v, self.tokenizer.pad_token_id)
                else:
                    model_inputs[f"flan_{k}"] = v
        else:
            model_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)

        if self.convert_to_standard_inputs:
            input_ids, attention_mask, position_ids, labels = convert_to_standard_inputs(model_inputs, self.tokenizer)

            labels = torch.cat([labels, index.unsqueeze(1)], dim=1)

            return (
                (input_ids, attention_mask, position_ids, index),
                labels,
            )

        return model_inputs
