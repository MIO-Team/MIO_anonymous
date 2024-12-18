"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


from .qformer_causual import BertConfig, BertLMHeadModel

from .utils import download_cached_file, get_rank, get_dist_info, get_world_size, main_process, is_dist_avail_and_initialized, is_url
from .eva_vit import create_eva_vit_g
from .clip_vit import create_clip_vit_L
from transformers import BertTokenizer


# class Blip2Base(BaseModel):
class Blip2Base(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, model_root_dir='models'):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, precision, model_root_dir=model_root_dir)

        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision, model_root_dir=model_root_dir)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        if self.vit_name == "eva_clip_g":
            vit_num_layers = self.visual_encoder.get_num_layer()
            lr_scales = list(lr_scale**(vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

            parameter_group_names = {}
            parameter_group_vars = {}

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias"):
                    group_name = "no_decay"
                    this_weight_decay = 0.
                else:
                    group_name = "decay"
                    this_weight_decay = weight_decay
                if 'visual_encoder' in name:
                    layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.', ''))
                    group_name = "vit_layer_%d_%s" % (layer_id, group_name)
                else:
                    layer_id = None

                if group_name not in parameter_group_names:
                    if layer_id is not None:
                        scale = lr_scales[layer_id]
                    else:
                        scale = 1
                    parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
                    parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
                parameter_group_vars[group_name]["params"].append(param)
                parameter_group_names[group_name]["params"].append(name)
            # import json
            # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
            optim_params = list(parameter_group_vars.values())
            return optim_params
        else:
            return super().get_optimizer_params(weight_decay, lr_scale)

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error("""
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """)
                exit(1)

        return self._lemmatizer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


