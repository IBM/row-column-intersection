import logging
from dataloader.distloader_seq_pair import SeqPairLoader, standard_json_mapper, sentence_to_inputs
from torch_util.hypers_base import HypersBase, fill_hypers_from_args
from torch_util.transformer_optimize import TransformerOptimize, set_seed, LossHistory
from torch_util.transformer_loader import load_pretrained, save_transformer, load_tokenizer, save_extended_model, load_only_extended_model
from torch_util.modeling_repr_seq_pair import TransformerReprSequencePairClassification
import torch
import numpy as np
import time
import os
import ujson as json
from torch_util.distributed import all_gather, reduce
from torch_util.validation import score_metrics, multiclass_score_metrics
import torch.nn.functional as F
import torch.nn
from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


class SeqPairHypers(HypersBase):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.max_seq_length = 128
        self.num_labels = 2
        self.single_sequence = False
        self.additional_special_tokens = ''
        self.is_separate = False
        # for reasonable values see the various params.json under
        #    https://github.com/peterliht/knowledge-distillation-pytorch
        self.kd_alpha = 0.9
        self.kd_temperature = 10.0


class SeqPairArgs(SeqPairHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.dev_dir = ''
        self.train_instances = 0  # we need to know the total number of training instances (should just be total line count)
        self.hyper_tune = 0  # number of trials to search hyperparameters
        self.prune_after = 5
        self.save_per_epoch = False
        self.teacher_labels = ''  # the labels from the teacher for the train_dir dataset


def load(hypers: SeqPairHypers):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]

    if hypers.is_separate:
        tokenizer = load_tokenizer(hypers, tokenizer_class,
                                   additional_special_tokens=hypers.additional_special_tokens.split(','))
        if hypers.resume_from:
            model, _ = load_only_extended_model(hypers, TransformerReprSequencePairClassification, hypers.resume_from)
        else:
            model = TransformerReprSequencePairClassification(hypers).to(hypers.device)
    else:
        if not hypers.resume_from:
            model, tokenizer = load_pretrained(hypers, config_class, model_class, tokenizer_class,
                                               additional_special_tokens=hypers.additional_special_tokens.split(','),
                                               num_labels=hypers.num_labels)
        else:
            tokenizer = load_tokenizer(hypers, tokenizer_class,
                                       additional_special_tokens=hypers.additional_special_tokens.split(','))
            model = model_class.from_pretrained(hypers.resume_from)
            model.to(hypers.device)

    return model, tokenizer


def save(hypers: SeqPairHypers, model, *, save_dir=None, tokenizer=None):
    if hypers.is_separate:
        save_extended_model(hypers, model, tokenizer=tokenizer, save_dir=save_dir)
    else:
        save_transformer(hypers, model, tokenizer, save_dir=save_dir)


def add_kd_loss(hypers: SeqPairHypers, logits, teacher_labels, hard_loss):
    T = hypers.kd_temperature
    kd_loss = torch.nn.KLDivLoss()(F.log_softmax(logits/T, dim=1), F.softmax(teacher_labels/T, dim=1)) * (T * T)
    return hypers.kd_alpha * kd_loss + (1.0 - hypers.kd_alpha) * hard_loss, hard_loss, kd_loss


def train(args: SeqPairArgs, train_dataset: SeqPairLoader, model):
    # transformer_optimize
    instances_to_train_over = args.train_instances * args.num_train_epochs
    toptimizer = TransformerOptimize(args, instances_to_train_over, model)

    # Train!
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    toptimizer.model.zero_grad()
    set_seed(args)
    loss_history = LossHistory(args.train_instances //
                               (args.full_train_batch_size // args.gradient_accumulation_steps))
    prev_epoch = 1
    while True:
        start_time = time.time()
        loader = train_dataset.get_dataloader()
        if train_dataset.on_epoch > prev_epoch:
            logger.info(f'On epoch {train_dataset.on_epoch}')
            if args.save_per_epoch:
                save(args, model, save_dir=os.path.join(args.output_dir, f'epoch_{prev_epoch}'))
            prev_epoch = train_dataset.on_epoch
        if loader is None:
            break
        for batch in loader:
            toptimizer.model.train()
            inputs = train_dataset.batch_dict(batch)
            if 'teacher_labels' in inputs:
                teacher_labels = inputs['teacher_labels']
                del inputs['teacher_labels']
            else:
                teacher_labels = None
            outputs = toptimizer.model(**inputs)
            loss, logits = outputs[0:2]  # model outputs are always tuple in transformers (see doc)
            if teacher_labels is not None:
                loss, hard_loss, kd_loss = add_kd_loss(args, logits, teacher_labels, loss)
                loss_val = toptimizer.step_loss(loss, hard_loss=hard_loss, kd_loss=kd_loss)
            else:
                loss_val = toptimizer.step_loss(loss)
            if loss_val is None:
                return loss_history.loss_history
            loss_history.note_loss(loss_val, hypers=args)
        logger.info(f'one group of train files took {(time.time()-start_time)/60} minutes')

    return loss_history.loss_history


def evaluate(args: SeqPairHypers, eval_dataset: SeqPairLoader, model):
    start_time = time.time()
    eval_dataset.reset(files_per_dataloader=-1)
    loader = eval_dataset.get_dataloader()
    model.eval()
    eval_loss = 0.0
    nb_eval_count = 0
    preds = None
    labels = None
    with torch.no_grad():
        for batch in loader:
            inputs = eval_dataset.batch_dict(batch)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_count += 1
            logits = logits.detach().cpu().numpy()
            input_labels = inputs["labels"].detach().cpu().numpy()
            if preds is None:
                preds = logits
                labels = input_labels
            else:
                preds = np.append(preds, logits, axis=0)
                labels = np.append(labels, input_labels, axis=0)
    eval_loss = eval_loss / nb_eval_count
    logger.info(f'went through {nb_eval_count} batches of evaluate')
    # gather results for distributed setting
    if args.local_rank != -1:
        preds = all_gather(torch.tensor(preds, dtype=torch.float32).to(args.device)).cpu().numpy()
        labels = all_gather(torch.tensor(labels, dtype=torch.int32).to(args.device)).cpu().numpy()
        eval_loss = reduce(args, eval_loss).item()/args.world_size
    # score
    if args.num_labels == 2:
        results = score_metrics(preds[:, 1], np.argmax(preds, axis=1), labels)
    else:
        results = multiclass_score_metrics(preds, np.argmax(preds, axis=1), labels)
    results['eval_loss'] = eval_loss

    logger.info("***** Eval results *****")
    logger.info(f'took {(time.time()-start_time)/60} mins')
    return results


def main():
    args = SeqPairArgs()
    fill_hypers_from_args(args)
    # Set seed
    set_seed(args)

    # load model and tokenizer
    model, tokenizer = load(args)
    logger.info(f'{sentence_to_inputs("This is a simple sentence.", tokenizer=tokenizer, max_seq_length=128)}')

    # Training
    loss_history = None
    if args.train_dir:
        assert args.train_instances > 0
        train_dataset = SeqPairLoader(args, args.per_gpu_train_batch_size, tokenizer, args.train_dir,
                                      is_separate=args.is_separate, is_single=args.single_sequence,
                                      teacher_labels=args.teacher_labels,
                                      json_mapper=standard_json_mapper)
        loss_history = train(args, train_dataset, model)
        # save model
        save(args, model, tokenizer=tokenizer)
        if args.world_size > 1:
            torch.distributed.barrier()

    # Evaluation
    if args.dev_dir:
        args.resume_from = args.output_dir
        model, tokenizer = load(args)

        eval_dataset = SeqPairLoader(args, args.per_gpu_eval_batch_size, tokenizer, args.dev_dir,
                                     is_separate=args.is_separate, is_single=args.single_sequence,
                                     json_mapper=standard_json_mapper)
        results = evaluate(args, eval_dataset, model)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if args.global_rank == 0:
            output_eval_file = os.path.join(args.output_dir, f"eval_results.json")
            with open(output_eval_file, "w") as writer:
                results['hypers'] = args.to_dict()
                writer.write(json.dumps(results, indent=2) + '\n')

    logger.info(f'loss_history = {loss_history}')


"""

"""
if __name__ == "__main__":
    main()
