from util.args_help import fill_from_args
from util.line_corpus import write_open, jsonl_lines
import ujson as json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score


class Options:
    def __init__(self):
        self.col = ''
        self.row = ''
        self.output = ''
        self.gt = ''
        self.cell_prediction_output = ''
        self.softmax = False


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def gather_predictions(input_file, *, softmax=False):
    predictions = defaultdict(list)
    for line in jsonl_lines(input_file):
        jobj = json.loads(line)
        if softmax:
            pred = log_softmax(np.array(jobj['predictions'], dtype=np.float32))[1]
        else:
            pred = jobj['predictions'][1]
        qid, ndx_str = jobj['id'].split(':')
        predictions[qid].append((int(ndx_str), pred))
    return predictions


def to_ndarray(preds):
    num = max(preds, key=lambda x: x[0])[0] + 1
    assert len(preds) == num
    arr = np.zeros(num, dtype=np.float32)
    for ndx, p in preds:
        assert arr[ndx] == 0
        arr[ndx] = p
    return arr


def to_cell_predictions(cell_preds, *, top_k=20):
    cell_predictions = []
    for ri in range(cell_preds.shape[0]):
        for ci in range(cell_preds.shape[1]):
            cell_predictions.append((ri, ci, float(cell_preds[ri, ci])))
    cell_predictions.sort(key=lambda x: x[2], reverse=True)
    return cell_predictions[:top_k]


def main():
    opts = Options()
    fill_from_args(opts)

    if opts.gt:
        id2gt = dict()
        lookup_subset = set()
        for line in jsonl_lines(opts.gt):
            jobj = json.loads(line)
            qid = jobj['id']
            tbl = jobj['rows']
            correct_cells = np.zeros((len(tbl), len(tbl[0])), dtype=np.bool)
            target_rows = jobj['target_rows'] if 'target_rows' in jobj else [jobj['target_row']]
            target_cols = jobj['target_columns'] if 'target_columns' in jobj else [jobj['target_column']]
            # TODO: also support getting correct cells from answers list
            for r in target_rows:
                for c in target_cols:
                    correct_cells[r, c] = True
            #if correct_cells.sum() == 0:
            #    print(f'No answer! {target_rows}, {target_cols}, {jobj["agg_index"]}')
            id2gt[qid] = correct_cells
            if 'agg_index' not in jobj or jobj['agg_index'] == 0:
                lookup_subset.add(qid)
    else:
        id2gt = None
        lookup_subset = None

    sums = defaultdict(float)
    counts = defaultdict(float)
    table_count = 0
    no_answer_count = 0
    col_predictions = gather_predictions(opts.col, softmax=opts.softmax)
    row_predictions = gather_predictions(opts.row, softmax=False)
    if opts.cell_prediction_output:
        cell_prediction_output = write_open(opts.cell_prediction_output)
    else:
        cell_prediction_output = None
    with write_open(opts.output) as out:
        for qid, col_preds in col_predictions.items():
            col_preds = to_ndarray(col_preds)
            row_preds = to_ndarray(row_predictions[qid])
            cell_preds = row_preds.reshape((-1, 1)) + col_preds.reshape((1, -1))
            if id2gt is not None:
                correct_cells = id2gt[qid]
                if correct_cells.sum() > 0:
                    avg_p = average_precision_score(y_true=correct_cells.reshape(-1), y_score=cell_preds.reshape(-1))
                    sums['auc'] += avg_p
                    counts['auc'] += 1
                    if qid in lookup_subset:
                        sums['auc (lookup)'] += avg_p
                        counts['auc (lookup)'] += 1
                    else:
                        sums['auc (aggregation)'] += avg_p
                        counts['auc (aggregation)'] += 1
                else:
                    no_answer_count += 1
            table_count += 1
            out.write(json.dumps({'qid': qid, 'cells': cell_preds.tolist(),
                                  'rows': row_preds.tolist(), 'cols': col_preds.tolist()})+'\n')
            if cell_prediction_output is not None:
                cell_prediction_output.write(json.dumps({'id': qid,
                                                         'cell_predictions': to_cell_predictions(cell_preds, top_k=20)})+'\n')
    if cell_prediction_output is not None:
        cell_prediction_output.close()
    for n, v in sums.items():
        print(f'{n} = {v/counts[n]}')
    print(f'Over {table_count} tables')
    if id2gt is not None and no_answer_count > 0:
        print(f'{no_answer_count} tables with no correct answer')


if __name__ == "__main__":
    main()
