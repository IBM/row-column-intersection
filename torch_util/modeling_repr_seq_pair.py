import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch_util.hypers_base import HypersBase
import logging
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,

    BertConfig,
    BertModel,
    BertTokenizer,
)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}


class TransformerReprSequencePairClassification(nn.Module):

    def __init__(self, hypers: HypersBase):
        super().__init__()
        self.num_labels = hypers.num_labels if hasattr(hypers, 'num_labels') else 2
        config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]

        # TODO: smarter about preventing multiple downloads in distributed setting?
        # TODO: but how would we load from a saved model without loading the pretrained first?
        config = config_class.from_pretrained(
            hypers.config_name if hypers.config_name else hypers.model_name_or_path,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None
        )
        self.transformer_a = model_class.from_pretrained(
            hypers.model_name_or_path,
            from_tf=bool(".ckpt" in hypers.model_name_or_path),
            config=config,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        )
        self.transformer_b = model_class.from_pretrained(
            hypers.model_name_or_path,
            from_tf=bool(".ckpt" in hypers.model_name_or_path),
            config=config,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        )

        self.config = config
        self.representation_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        two_layer = hasattr(hypers, 'two_layer_classifier') and hypers.two_layer_classifier
        if two_layer:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, self.num_labels),
            )
        else:
            self.classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        token_type_ids_a=None,
        vectors_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        token_type_ids_b=None,
        vectors_b=None,
        labels=None,
    ):
        if input_ids_b is not None:
            pooled_output_b = self.transformer_b(
                input_ids_b,
                attention_mask=attention_mask_b,
                token_type_ids=token_type_ids_b,
            )[1]
        else:
            pooled_output_b = vectors_b

        if input_ids_a is None and vectors_a is None and input_ids_b is not None:
            return (pooled_output_b,)

        if input_ids_a is not None:
            pooled_output_a = self.transformer_a(
                input_ids_a,
                attention_mask=attention_mask_a,
                token_type_ids=token_type_ids_a,
            )[1]
        else:
            pooled_output_a = vectors_a

        if input_ids_b is None and vectors_b is None and input_ids_a is not None:
            return (pooled_output_a,)

        # we can accept input like  query (batch_size x seq_len) vectors_b (batch_size x num_cols x embed_dim)
        if len(pooled_output_b.shape) > 2:
            pooled_output_a = pooled_output_a.unsqueeze(1).expand(pooled_output_b.shape)
            pooled_output_a = pooled_output_a.reshape(-1, self.representation_dim)
            pooled_output_b = pooled_output_b.reshape(-1, self.representation_dim)

        if pooled_output_a.shape[0] != pooled_output_b.shape[0]:
            pooled_output_a = pooled_output_a.expand(pooled_output_b.shape)

        pooled_output = torch.cat((pooled_output_a, pooled_output_b,
                                   pooled_output_a * pooled_output_b,
                                   (pooled_output_a - pooled_output_b)**2), 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
