import logging
import ujson as json
import random
from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
import os

logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.cell_sep_token = '*'  # used to use '|' but albert doesn't have it
        self.cell_value_sep_token = ':'
        self.answer_in_header = False
        self.max_cells = -1
        self.negative_sample_rate = 1.0
        self.simple_string = False
        self.per_table_negatives = -1


class RCIInst:
    __slots__ = 'id', 'text_a', 'text_b', 'label'

    def __init__(self, inst_id, text_a, text_b, label):
        self.id = inst_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def to_dict(self):
        return {'id': self.id, 'text_a': self.text_a, 'text_b': self.text_b, 'label': self.label}


class RowConvert:
    def __init__(self, config: Config):
        self.pos_count = 0
        self.all_neg_count = 0
        self.neg_count = 0
        self.unanswerable = 0
        self.multi_answer = 0
        self.single_answer = 0
        self.config = config

    def convert(self, line):
        config = self.config
        cell_sep = f' {config.cell_sep_token} '
        cell_value_sep = f' {config.cell_value_sep_token} '
        examples = []
        jobj = json.loads(line)
        qid = jobj['id']
        header = [str(h) for h in jobj['header']]
        rows = jobj['rows']
        query = jobj['question']
        target_column = jobj['target_column'] if 'target_column' in jobj else None
        target_row = jobj['target_row'] if 'target_row' in jobj else None
        target_rows = jobj['target_rows'] if 'target_rows' in jobj else None
        answers = jobj['answers'] if 'answers' in jobj else None
        table_pos = []
        table_neg = []
        if config.answer_in_header:
            rows = [header] + rows
        for ri, row in enumerate(rows):
            if target_row is not None:
                is_pos = ri == target_row
            elif target_rows is not None:
                is_pos = ri in target_rows
            elif target_column is not None:
                is_pos = row[target_column] in answers
            else:
                is_pos = any([cell in answers for cell in row])
            if is_pos or random.random() <= config.negative_sample_rate:
                if config.answer_in_header and ri == 0:
                    if config.simple_string:
                        row_rep = ' '.join(header)
                    else:
                        row_rep = 'HEADER: ' + cell_sep.join(header)
                else:
                    if config.simple_string:
                        row_rep = ' '.join(row)
                    else:
                        row_rep = cell_sep.join([h + cell_value_sep + str(c) for h, c in zip(header, row)])
                # guid should is qid + row number
                example = RCIInst(inst_id=f'{qid}:{ri}', text_a=query, text_b=row_rep, label=is_pos)
                if is_pos:
                    self.pos_count += 1
                    table_pos.append(example)
                else:
                    self.neg_count += 1
                    table_neg.append(example)
            if not is_pos:
                self.all_neg_count += 1
        if len(table_pos) == 0:
            self.unanswerable += 1
            logger.info(f'UNANSWERABLE:\n{rows}\n{answers}')
        elif len(table_pos) > 1:
            self.multi_answer += 1
        else:
            self.single_answer += 1
        if config.per_table_negatives > 0:
            table_neg = table_neg[:config.per_table_negatives]
        examples.extend(table_pos)
        examples.extend(table_neg)
        return examples


class ColumnConvert():
    def __init__(self, config: Config):
        self.pos_count = 0
        self.all_neg_count = 0
        self.neg_count = 0
        self.config = config

    def convert(self, line):
        config = self.config
        cell_sep = f' {config.cell_sep_token} '
        insts = []
        jobj = json.loads(line)
        qid = jobj['id']
        query = jobj['question']
        target_column = jobj['target_column'] if 'target_column' in jobj else None
        target_columns = jobj['target_columns'] if 'target_columns' in jobj else None
        header = jobj['header']
        rows = jobj['rows']
        answers = jobj['answers'] if 'answers' in jobj else None
        all_rows = rows
        if config.max_cells > 0:
            rows = rows[:config.max_cells]
        cols = [[str(h)] for h in header]
        for row in rows:
            for ci, cell in enumerate(row):
                cols[ci].append(str(cell))
        for ci, col in enumerate(cols):
            if target_column is not None:
                is_pos = ci == target_column
            elif target_columns is not None:
                is_pos = ci in target_columns
            else:
                is_pos = any([row[ci] in answers for row in all_rows])
                if not is_pos and config.answer_in_header:
                    is_pos = header[ci] in answers
            if is_pos or random.random() <= config.negative_sample_rate:
                insts.append(RCIInst(inst_id=f'{qid}:{ci}',
                                     text_a=query,
                                     text_b=' '.join(col) if config.simple_string else cell_sep.join(col),
                                     label=is_pos))
                if is_pos:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
            if not is_pos:
                self.all_neg_count += 1
        return insts


class Options(Config):
    def __init__(self):
        super().__init__()
        self.input_dir = ''
        self.style = 'lookup'
        self.output_dir = ''


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    for split in ['train', 'dev', 'test']:
        cols = ColumnConvert(opts)
        rows = RowConvert(opts)
        with write_open(os.path.join(opts.output_dir, split, 'row.jsonl.gz')) as rout, \
                write_open(os.path.join(opts.output_dir, split, 'col.jsonl.gz')) as cout:
            for line in jsonl_lines(os.path.join(opts.input_dir, f'{split}_{opts.style}.jsonl.gz')):
                for r in rows.convert(line):
                    rout.write(json.dumps(r.to_dict())+'\n')
                for c in cols.convert(line):
                    cout.write(json.dumps(c.to_dict())+'\n')
