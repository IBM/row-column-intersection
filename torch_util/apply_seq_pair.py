import logging
from dataloader.distloader_seq_pair import SeqPairLoader, standard_json_mapper
from torch_util.classify_seq_pair import SeqPairHypers, load
import torch
import os
import ujson as json
from util.line_corpus import write_open

logger = logging.getLogger(__name__)


class SeqPairArgs(SeqPairHypers):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.input_dir = ''


def evaluate(args: SeqPairHypers, eval_dataset: SeqPairLoader, model):
    eval_dataset.reset(uneven_batches=True, files_per_dataloader=-1)
    loader = eval_dataset.get_dataloader()
    model.eval()
    with torch.no_grad(), write_open(os.path.join(args.output_dir, f'results{args.global_rank}.jsonl.gz')) as f:
        for batch in loader:
            ids = batch[0]
            inputs = eval_dataset.batch_dict(batch)
            if 'labels' in inputs:
                del inputs['labels']
            logits = model(**inputs)[0].detach().cpu().numpy()
            for id, pred in zip(ids, logits):
                assert type(id) == str
                pred = [float(p) for p in pred]
                assert len(pred) == args.num_labels
                assert all([type(p) == float for p in pred])
                f.write(json.dumps({'id': id, 'predictions': pred})+'\n')


def main():
    args = SeqPairArgs().fill_from_args()
    args.set_seed()

    # load model and tokenizer
    model, tokenizer = load(args)

    eval_dataset = SeqPairLoader(args, args.per_gpu_eval_batch_size, tokenizer, args.input_dir,
                                 is_separate=args.is_separate, is_single=args.single_sequence,
                                 json_mapper=standard_json_mapper)
    evaluate(args, eval_dataset, model)


"""

"""
if __name__ == "__main__":
    main()
