import ujson as json
from util.line_corpus import read_open, write_open
import argparse
import re
import logging
from tabulate import tabulate

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--pred_a", default=None, type=str, required=True)
parser.add_argument("--pred_b", default=None, type=str, required=True)
parser.add_argument("--out", default=None, type=str, required=True)
parser.add_argument("--gt", default=None, type=str, required=True)
parser.add_argument("--answer_in_header_a", action="store_true")
parser.add_argument("--answer_in_header_b", action="store_true")

args = parser.parse_args()


def _evaluate_pred_list(pred_list, jobj, *, answer_in_header):
    answers = jobj['answers']
    rows = jobj['rows']
    if answer_in_header:
        rows = [jobj['header']] + rows
    target_column = None
    if 'target_column' in jobj:
        target_column = jobj['target_column']

    target_rows = set()
    target_cols = set()
    for ri, row in enumerate(rows):
        for ci, cv in enumerate(row):
            if cv in answers:
                target_rows.add(ri)
                target_cols.add(ci)
    if target_column is not None:
        target_cols = [target_column]
    col_rank = None
    row_rank = None
    cell_rank = None
    for ndx, pred in enumerate(pred_list):
        if cell_rank is None:
            cell_value = rows[pred[0]][pred[1]]
            if cell_value in answers:
                cell_rank = ndx + 1
                if col_rank is None:
                    col_rank = ndx + 1
        if col_rank is None:
            if pred[1] in target_cols:
                col_rank = ndx+1
        if row_rank is None:
            if pred[0] in target_rows:
                row_rank = ndx+1
    return cell_rank is not None and cell_rank == 1, rows[pred_list[0][0]][pred_list[0][1]]


def get_pred_map(file):
    qid2pred = dict()
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            jobj = json.loads(line)
            qid = jobj['id']
            preds = jobj['cell_predictions']
            qid2pred[qid] = preds
    return qid2pred


pred_a = get_pred_map(args.pred_a)
pred_b = get_pred_map(args.pred_b)

whitespace = re.compile(r"\s+")

with read_open(args.gt) as gt_f, write_open(args.out) as out_f:
    for line in gt_f:
        jobj = json.loads(line)
        qid = str(jobj['id'])
        correct_a, ans_a = _evaluate_pred_list(pred_a[qid], jobj, answer_in_header=args.answer_in_header_a)
        correct_b, ans_b = _evaluate_pred_list(pred_b[qid], jobj, answer_in_header=args.answer_in_header_b)
        if correct_a and not correct_b:
            out_f.write('\n\n'+'='*80+'\n')
            out_f.write(tabulate(jobj['rows'], headers=jobj['header'], showindex=True)+'\n')
            out_f.write(jobj['question']+'\n')
            out_f.write(f"{whitespace.sub(' ', ans_a)} ({pred_a[qid][:10]})\n")
            out_f.write(f"{whitespace.sub(' ', ans_b)} ({pred_b[qid][:10]})\n")
