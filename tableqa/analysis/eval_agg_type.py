from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines
import ujson as json
import numpy as np
from collections import defaultdict


class Options:
    def __init__(self):
        self.input = ''
        self.gt = ''
        self.softmax = False


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def main():
    opts = Options()
    fill_from_args(opts)

    id2gt = dict()
    for line in jsonl_lines(opts.gt):
        jobj = json.loads(line)
        qid = jobj['id']
        id2gt[qid] = jobj['agg_index']

    sums = defaultdict(float)
    counts = defaultdict(float)
    for line in jsonl_lines(opts.input):
        jobj = json.loads(line)
        qid = jobj['id']
        gt = id2gt[qid]
        preds = np.array(jobj['predictions'], dtype=np.float32)
        correct = 1 if np.argmax(preds) == gt else 0
        counts[f'accuracy_{gt}'] += 1
        sums[f'accuracy_{gt}'] += correct
        counts[f'accuracy'] += 1
        sums[f'accuracy'] += correct
    metric_names = list(sums.keys())
    metric_names.sort()
    for n in metric_names:
        print(f'{n} = {sums[n]/counts[n]} over {counts[n]}')


"""
python eval_agg_type.py \
--input /users/mrglass/TableQA/wikisqla/seqpair/apply/dev/agg_alb \
--gt /users/mrglass/TableQA/wikisqla/dev_std.jsonl
"""
if __name__ == "__main__":
    main()
