import ujson as json
import numpy as np
import argparse
import logging
from util.line_corpus import read_lines

logger = logging.getLogger(__name__)


def _evaluate_pred_list(pred_list, rows, target_column, answers, cutoffs, correctness):
    target_rows = set()
    target_cols = set()
    for ri, row in enumerate(rows):
        for ci, cv in enumerate(row):
            if cv in answers:
                target_rows.add(ri)
                target_cols.add(ci)
    if target_column is not None:
        target_cols = [target_column]
    assert len(correctness.shape) == 2
    assert correctness.shape[0] == len(cutoffs) + 1
    assert correctness.shape[1] == 3
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

    assert cell_rank is None or col_rank <= cell_rank
    assert cell_rank is None or row_rank <= cell_rank
    correctness[0, 0] += 1.0 / cell_rank if cell_rank is not None else 0.0
    correctness[0, 1] += 1.0 / col_rank if col_rank is not None else 0.0
    correctness[0, 2] += 1.0 / row_rank if row_rank is not None else 0.0
    for ci, cutoff in enumerate(cutoffs):
        if cell_rank is not None and cell_rank <= cutoff:
            correctness[ci + 1, 0] += 1
        if col_rank is not None and col_rank <= cutoff:
            correctness[ci + 1, 1] += 1
        if row_rank is not None and row_rank <= cutoff:
            correctness[ci + 1, 2] += 1


# data file = os.path.join(data_dir, "dev_std.jsonl")
def evaluate_pred_map(gt_file, qid2pred, cutoffs, *,
                      answer_in_header=False, ignore_target_column=False):
    # correct, correct_column, correct_row
    # first correctness row is MRR, then Hit@cutoffs[i]
    correctness = np.zeros((1+len(cutoffs), 3), dtype=np.float32)
    total = 0
    answerable = 0
    missing_predictions = 0
    multi_answer_count = 0
    for line in read_lines(gt_file):
        jobj = json.loads(line)
        qid = str(jobj['id'])
        if 'agg_index' in jobj and jobj['agg_index'] != 0:
            continue
        if not qid in qid2pred:
            logger.warning(f'No predictions for {qid}')
            missing_predictions += 1
            continue
        answers = jobj['answers']
        if len(set(answers)) > 1:
            multi_answer_count += 1
        rows = jobj['rows']
        if answer_in_header:
            rows = [jobj['header']] + rows
        pred = qid2pred[qid]
        if type(pred) != list:
            pred = [pred]
        target_column = None
        if 'target_column' in jobj and not ignore_target_column:
            target_column = jobj['target_column']
        _evaluate_pred_list(pred, rows, target_column, answers, cutoffs, correctness)
        def is_answerable(rows, answers):
            for row in rows:
                for cv in row:
                    if cv in answers:
                        return True
            return False

        if is_answerable(rows, answers):
            answerable += 1

        total += 1

    correctness /= total

    print(f'Answerable {answerable/total} over {total}')
    print(f'MRR cell = {correctness[0, 0]}, column = {correctness[0, 1]}, row = {correctness[0, 2]}')
    for ci, cutoff in enumerate(cutoffs):
        print(f'Hit@{cutoff} cell = {correctness[ci+1, 0]}, '
              f'column = {correctness[ci+1, 1]}, row = {correctness[ci+1, 2]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()
    qid2pred = dict()
    for line in read_lines(args.preds):
        jobj = json.loads(line)
        qid = jobj['id']
        preds = jobj['cell_predictions']
        qid2pred[qid] = preds
    cutoffs = (1, 2, 3, 4, 5)
    evaluate_pred_map(args.gt, qid2pred, cutoffs)
