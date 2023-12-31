import logging

from transformers import PreTrainedTokenizer
import os
from data.data_utils import tokenizer_get_name

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

logger = logging.getLogger(__name__)


def expand_special_tokenizer(tokenizer: PreTrainedTokenizer):
    tokenizer_name = tokenizer_get_name(tokenizer)
    if "llama" in tokenizer_name:
        special_tokens_map = {}
        eos_token = os.environ.get("EOS_TOKEN", None)
        if eos_token or (not tokenizer.eos_token):
            special_tokens_map["eos_token"] = eos_token if eos_token else DEFAULT_EOS_TOKEN

        bos_token = os.environ.get("BOS_TOKEN", None)
        if bos_token or (not tokenizer.bos_token):
            special_tokens_map["bos_token"] = bos_token if bos_token else DEFAULT_BOS_TOKEN

        unk_token = os.environ.get("UNK_TOKEN", None)
        if not tokenizer.unk_token:
            special_tokens_map["unk_token"] = unk_token if unk_token else DEFAULT_UNK_TOKEN

        pad_token = os.environ.get("PAD_TOKEN", None)
        if not tokenizer.pad_token:
            special_tokens_map["pad_token"] = pad_token if pad_token else DEFAULT_PAD_TOKEN

        new_tokens = tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_map
        )
        # new_tokens = tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN))
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # assert new_tokens == 1
    elif "gptneox" in tokenizer_name:
        special_tokens_map = {}
        eos_token = os.environ.get("EOS_TOKEN", None)
        if eos_token:
            special_tokens_map["eos_token"] = eos_token if eos_token else DEFAULT_EOS_TOKEN

        new_tokens = tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_map
        )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(tokenizer)


def is_seq2seq_tokenizer(tokenizer: PreTrainedTokenizer):
    tokenizer_name = tokenizer_get_name(tokenizer)
    return any([x in tokenizer_name for x in ["t5", "bart", "pegasus", "mbart", "marian", "blenderbot"]])
