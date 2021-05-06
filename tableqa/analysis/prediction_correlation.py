import ujson as json
from util.line_corpus import jsonl_lines
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--predictions_a", default=None, type=str, required=True)
parser.add_argument("--predictions_b", default=None, type=str, required=True)
args = parser.parse_args()


def qid2predictions(pred_file):
    qid2preds = dict()
    for line in jsonl_lines(pred_file):
        jobj = json.loads(line)
        preds = []
        for p in jobj['predictions']:
            preds.append(f'{p[0]}-{p[1]}')
        qid2preds[jobj['id']] = preds
    return qid2preds


qid2preds_a = qid2predictions(args.predictions_a)
qid2preds_b = qid2predictions(args.predictions_b)

assert len(qid2preds_a) == len(qid2preds_b)
sum_intersection_1 = 0
sum_intersection_5 = 0
for qid in qid2preds_a.keys():
    preds_a = qid2preds_a[qid]
    preds_b = qid2preds_b[qid]
    sum_intersection_1 += 1 if preds_a[0] == preds_b[0] else 0
    sum_intersection_5 += len(set(preds_a[:5]).intersection(set(preds_b[:5]))) / 5.0

print(f'{sum_intersection_1/len(qid2preds_a)} match at 1')
print(f'{sum_intersection_5/len(qid2preds_a)} match at 5')
