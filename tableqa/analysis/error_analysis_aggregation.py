from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open, read_lines
import ujson as json
import numpy as np
from collections import defaultdict
from tableqa.tapas_text_utils import normalize_answers
from tabulate import tabulate


class Options:
    def __init__(self):
        self.agg_preds = ''
        self.error_qids = ''  # file with qids to use for error analysis, one per line
        self.cell_preds = ''
        self.lookup_preds = ''  # if provided, use these predictions for lookup questions (we should get cell predictions for the lookup model too)
        self.gt = ''
        self.output = ''
        self.threshold_per_agg = False
        self.use_threshold = -1000.0  # if > -1000, will use this threshold rather than compute best from gt


class QInfo:
    def __init__(self):
        self.qid = None
        self.question = None
        self.gt_agg_index = None
        self.header = None
        self.rows = None
        self.cell_gt = None
        self.col_gt = None
        self.row_gt = None
        self.col_vals = None
        self.answers_gt = None
        self.cell_confs = None
        self.agg_pred = None
        self.agg_confs = None
        self.threshold_range = None
        self.agg_answers = None

    def to_string(self, threshold):
        """
        question, table, ground truth answer, ground truth aggregation
        aggregation prediction
        cell confidences
        answer, threshold exists
        :return:
        """
        tbl = []
        for rndx, row in enumerate(self.rows):
            r = []
            for cndx, cell in enumerate(row):
                r.append(f'{">>>" if self.cell_gt[rndx, cndx] else ""}{self.cell_confs[rndx, cndx]:.2f}: {cell}')
            tbl.append(r)
        agg_ops = ['LOOKUP', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        qstr = '\n\n' + '=' * 80 + f'\n{self.qid}\n' + \
              tabulate(tbl, headers=self.header, showindex=True) + '\n' + self.question + '\n' + \
              f'agg gold = {agg_ops[self.gt_agg_index]} vs {agg_ops[self.agg_pred]} ({list(zip(agg_ops, self.agg_confs))})'
        pstr = f'answer = {self.answers_gt} vs predicted = {self.answer_at_threshold(threshold)}\nthreshold range = {self.threshold_range}, threshold = {threshold:.2f}'
        return qstr + '\n' + pstr

    def fill_from_gt(self, jobj, *, blind_gt=False):
        self.qid = jobj['id']
        self.header = jobj['header']
        self.rows = jobj['rows']
        self.question = jobj['question']
        self.set_correct_cells(jobj)
        self.col_vals = get_numeric_col_vals(self.rows)
        if blind_gt:
            self.answers_gt = ['X']
            self.gt_agg_index = -1
        else:
            self.answers_gt = jobj['answers']
            self.gt_agg_index = jobj['agg_index']
            if self.gt_agg_index != 0 and self.gt_agg_index != 3:
                for tc in self.col_gt:
                    if self.col_vals[tc] is None:
                        print(f'{self.col_gt} but non-numeric!\n{self.rows}\n{jobj["question"]}\naggregation = {self.gt_agg_index}')
                        raise Exception

    def set_correct_cells(self, jobj):
        tbl = jobj['rows']
        self.cell_gt = np.zeros((len(tbl), len(tbl[0])), dtype=np.bool)
        self.row_gt = jobj['target_rows'] if 'target_rows' in jobj else [jobj['target_row']]
        self.col_gt = jobj['target_columns'] if 'target_columns' in jobj else [jobj['target_column']]
        for r in self.row_gt:
            for c in self.col_gt:
                self.cell_gt[r, c] = True

    def compute_agg_pred(self):
        col_ndx = np.argmax(self.cell_confs[0])
        if self.col_vals[col_ndx] is None and self.agg_pred != 0 and self.agg_pred != 3:
            if self.agg_confs[0] > self.agg_confs[3]:
                self.agg_pred = 0
            else:
                self.agg_pred = 3
        return self.agg_pred

    def answer_at_threshold(self, threshold):
        self.compute_agg_pred()
        col_ndx = np.argmax(self.cell_confs[0])
        #if self.agg_pred == 0:
        #    # FIXME: actually should return a list sometimes
        #    row_ndx = np.argmax(self.cell_confs[:, col_ndx])
        #    return [self.rows[row_ndx][col_ndx]]
        sorted_ndxs = np.argsort(self.cell_confs[:, col_ndx])[::-1]
        values = []
        for k, ndx in enumerate(sorted_ndxs):
            conf = self.cell_confs[ndx, col_ndx]
            if conf < threshold:
                break
            values.append(self._get_cell_value_for_agg(ndx, col_ndx))
        if len(values) == 0:
            return []
        correct, pred = is_correct(self.agg_pred, values, [0])
        return pred

    def _get_cell_value_for_agg(self, row_ndx, col_ndx):
        if self.agg_pred == 0:
            return self.rows[row_ndx][col_ndx]  # get from the cell
        elif self.agg_pred != 3:
            return self.col_vals[col_ndx][row_ndx]  # get from the number converted cell
        else:
            return None

    def compute_threshold_range(self):
        self.compute_agg_pred()
        # if self.agg_pred == 0:
        #     # no aggregation
        #     return
        col_ndx = np.argmax(self.cell_confs[0])

        sorted_ndxs = np.argsort(self.cell_confs[:, col_ndx])[::-1]
        if len(self.row_gt) == 0:
            # if no cells should be aggregated, the best threshold is anything above the max conf
            self.threshold_range = (self.cell_confs[0, col_ndx], 100)
            return
        values = []
        max_conf = None
        min_conf = None
        # try:
        #     answers = [to_number(a) for a in self.answers_gt]
        # except:
        #     answers = self.answers_gt
        answers = normalize_answers(self.answers_gt)
        self.agg_answers = []
        for k, ndx in enumerate(sorted_ndxs):
            conf = self.cell_confs[ndx, col_ndx]
            values.append(self._get_cell_value_for_agg(ndx, col_ndx))
            correct, pred = is_correct(self.agg_pred, values, answers)
            self.agg_answers.append(list(pred))
            if correct:
                if max_conf is None:
                    max_conf = conf  # confidence threshold should not be higher than this
            elif max_conf is not None:
                min_conf = conf  # confidence threhold should be above this
                break
            else:
                continue
        if self.gt_agg_index == 3 and self.agg_pred == 3:
            assert max_conf is not None
        if max_conf is not None and min_conf is not None:
            self.threshold_range = (min_conf, max_conf)
        elif max_conf is not None:
            self.threshold_range = (-100, max_conf)
        else:
            self.threshold_range = None
            assert self.agg_pred == 0 or self.answers_gt not in self.agg_answers


def to_number(cell):
    return float(cell.replace(',', '').replace(' ', ''))


def get_numeric_col_vals(tbl):
    numeric_columns = np.ones(len(tbl[0]), dtype=np.bool)
    for ri, row in enumerate(tbl):
        for ci, cell in enumerate(row):
            try:
                v = to_number(cell)
            except:
                numeric_columns[ci] = False
    col_vals = [np.zeros(len(tbl), dtype=np.double) if numeric_columns[ci] else None
                for ci in range(numeric_columns.shape[0])]
    for ri, row in enumerate(tbl):
        for ci, cell in enumerate(row):
            if numeric_columns[ci]:
                col_vals[ci][ri] = to_number(cell)
    return col_vals


def avg(l):
    return sum(l)/len(l)


def is_correct(agg_index, values, answers):
    agg_ops = [None, max, min, len, sum, avg]
    if agg_index == 0:
        pred = values
    else:
        pred = [agg_ops[agg_index](values)]
    # if pred in answers and not str(pred) in normalize_answers(answers):
    #    print(f'bad answer checking?! {pred}, {answers} but {str(pred)}, {normalize_answers(answers)}')
    correct = normalize_answers(pred) == answers
    # if not correct and is_correct(agg_index, values, answers)[0]:
    #     print(f'Too strict? Counted wrong: {pred} vs {answers}')
    # if correct and not is_correct(agg_index, values, answers)[0]:
    #     print(f'Too easy?   Counted right: {pred} vs {answers}')
    return correct, pred


def accuracy_at_threshold(qinfos, threshold, *, for_agg_index=None):
    """
    Accuracy for the aggregation questions
    :param qinfos:
    :param threshold:
    :return:
    """
    correct_count = 0
    count = 0
    for qinfo in qinfos:
        if for_agg_index is not None and for_agg_index != qinfo.agg_pred:
            continue
        if qinfo.threshold_range is not None and qinfo.threshold_range[0] <= threshold <= qinfo.threshold_range[1]:
            correct_count += 1
        count += 1
    return correct_count / count


def find_best_threshold(qinfos, *, for_agg_index=None):
    candidate_thresholds = set()
    threshold_exists = 0
    count = 0
    for qinfo in qinfos:
        if for_agg_index is not None and for_agg_index != qinfo.agg_pred:
            continue
        if qinfo.threshold_range is not None:
            candidate_thresholds.add(qinfo.threshold_range[0])
            candidate_thresholds.add(qinfo.threshold_range[1])
            threshold_exists += 1
        count += 1
    print(f'threshold exists for {threshold_exists/count}')

    max_accuracy = 0
    best_threshold = None
    for threshold in candidate_thresholds:
        accuracy = accuracy_at_threshold(qinfos, threshold, for_agg_index=for_agg_index)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_threshold = threshold
    return max_accuracy, best_threshold


def main():
    opts = Options()
    fill_from_args(opts)

    id2qinfo = defaultdict(QInfo)
    for line in jsonl_lines(opts.gt):
        jobj = json.loads(line)
        id2qinfo[jobj['id']].fill_from_gt(jobj)

    sums = defaultdict(float)
    counts = defaultdict(float)
    for line in jsonl_lines(opts.agg_preds):
        jobj = json.loads(line)
        qid = jobj['id']
        qinfo = id2qinfo[qid]
        preds = np.array(jobj['predictions'], dtype=np.float32)
        predicted = np.argmax(preds)
        gt = qinfo.gt_agg_index
        qinfo.agg_pred = predicted
        qinfo.agg_confs = preds
        correct = 1 if predicted == gt else 0
        counts[f'accuracy_{gt}'] += 1
        sums[f'accuracy_{gt}'] += correct
        counts[f'accuracy'] += 1
        sums[f'accuracy'] += correct

    error_qids = set()
    for line in read_lines(opts.error_qids):
        qid = line.strip()
        error_qids.add(qid)
        if qid not in id2qinfo:
            raise ValueError

    metric_names = list(sums.keys())
    metric_names.sort()
    for n in metric_names:
        print(f'{n} = {sums[n]/counts[n]} over {counts[n]}')

    for line in jsonl_lines(opts.cell_preds):
        jobj = json.loads(line)
        qid = jobj['qid']
        cell_preds = np.array(jobj['cells'], dtype=np.float32)
        qinfo = id2qinfo[qid]
        qinfo.cell_confs = cell_preds

    if opts.lookup_preds:
        for line in jsonl_lines(opts.lookup_preds):
            jobj = json.loads(line)
            qid = jobj['qid']
            qinfo = id2qinfo[qid]
            if qinfo.compute_agg_pred() == 0:
                cell_preds = np.array(jobj['cells'], dtype=np.float32)
                qinfo.cell_confs = cell_preds

    err_analysis_count = 0  # make non-zero to show cases where no threshold is possible
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    per_agg_thresholds = np.zeros(len(agg_ops), dtype=np.float32)
    if opts.use_threshold <= -1000:
        for qinfo in id2qinfo.values():
            qinfo.compute_threshold_range()
            if qinfo.threshold_range is None and qinfo.agg_pred != 0 and err_analysis_count > 0:
                err_analysis_count -= 1
                print(f'No threshold possible: {qinfo.question}\nagg {agg_ops[qinfo.gt_agg_index]} over {qinfo.col_gt},{qinfo.row_gt} yielding {qinfo.answers_gt}')
                print(f'Predicted agg {agg_ops[qinfo.agg_pred]} over {np.argmax(qinfo.cell_confs[0])} yielding {qinfo.agg_answers}')
                print([f'{h}:{qinfo.col_vals[hi] is not None}' for hi, h in enumerate(qinfo.header)])
                for ri, row in enumerate(qinfo.rows):
                    to_show = [f'{cell}:{qinfo.cell_confs[ri,ci]}' for ci, cell in enumerate(row)]
                    print(to_show)

        max_accuracy, best_threshold = find_best_threshold(id2qinfo.values())
        print(f'can get {max_accuracy} with threshold {best_threshold}')
        print(f'    {accuracy_at_threshold(id2qinfo.values(), best_threshold-0.1)} with threshold {best_threshold - 0.1}')
        print(f'    {accuracy_at_threshold(id2qinfo.values(), best_threshold+0.1)} with threshold {best_threshold + 0.1}')

        for ai in range(0, per_agg_thresholds.shape[0]):
            acc, bt = find_best_threshold(id2qinfo.values(), for_agg_index=ai)
            print(f'for {agg_ops[ai]} can get {acc} with threshold {bt}')
            per_agg_thresholds[ai] = bt
    else:
        best_threshold = opts.use_threshold
        per_agg_thresholds[:] = opts.use_threshold

    missed_lookup = 0
    lookup = 0
    non_lookup = 0
    lookup_by_agg = 0
    pred_out = write_open(opts.output)
    for qinfo in id2qinfo.values():
        if qinfo.gt_agg_index == 0:
            lookup += 1
            if qinfo.agg_pred != 0 and qinfo.threshold_range is not None:
                #print(f'Aggregation gets right answer anyway? {qinfo.question}\nagg {agg_ops[qinfo.gt_agg_index]} over {qinfo.col_gt},{qinfo.row_gt} yielding {qinfo.answers_gt}')
                #print(f'Predicted agg {agg_ops[qinfo.agg_pred]} over {np.argmax(qinfo.cell_confs[0])} yielding {qinfo.agg_answers}')
                if qinfo.threshold_range[0] <= best_threshold <= qinfo.threshold_range[1]:
                    lookup_by_agg += 1
        else:
            non_lookup += 1
        if qinfo.gt_agg_index == 0 and qinfo.agg_pred != 0:
            missed_lookup += 1

        this_threshold = per_agg_thresholds[qinfo.agg_pred] if opts.threshold_per_agg else best_threshold
        if qinfo.gt_agg_index != 0 and qinfo.qid in error_qids:
            pred_out.write(qinfo.to_string(this_threshold)+'\n\n')

    pred_out.close()
    print(f'Lookup count = {lookup}, Non-lookup = {non_lookup}, '
          f'Lookup mispredicted as non-lookup = {missed_lookup}, but correct anyway = {lookup_by_agg}')


"""

"""
if __name__ == "__main__":
    main()
