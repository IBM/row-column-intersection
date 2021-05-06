import json
from tqdm import tqdm
from datasets.wikisql.wikisql_dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines
import os
from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
from datasets.wikisql.tables2agg_classify import write_agg_classify
import logging

logger = logging.getLogger(__name__)


def cell_value_is_answer(cv, answers):
    if cv in answers or cv.lower() in answers:
        return True
    try:
        cvf = float(cv)
        for ans in answers:
            try:
                if cvf == float(ans):
                    return True
            except:
                pass
    except:
        pass
    return False


def convert(queries, tables, outfile, *, skip_aggregation=True, show_aggregation=False):
    """Creates examples for the training and dev sets."""
    tid2rows = dict()
    for line in tables:
        jobj = json.loads(line)
        tid = jobj['id']
        header = jobj['header']
        rows_orig = jobj['rows']
        rows = []
        for r in rows_orig:
            rows.append([str(cv) for cv in r])
        tid2rows[tid] = [[str(h) for h in header]] + rows
    with write_open(outfile) as out:
        for qid, line in enumerate(queries):
            jobj = json.loads(line)
            agg_index = jobj['sql']['agg']
            if skip_aggregation and agg_index != 0:  # skip aggregation queries
                continue
            table_id = jobj['table_id']
            rows = tid2rows[table_id]
            qtext = jobj['question']
            target_column = jobj['sql']['sel']
            condition_columns = [colndx for colndx, comp, val in jobj['sql']['conds']]
            answers = jobj['answer']
            rowids = jobj['rowids'] if 'rowids' in jobj else None
            jobj = dict()
            jobj['id'] = f'{qid}'
            jobj['question'] = qtext
            jobj['header'] = rows[0]
            jobj['rows'] = rows[1:]
            jobj['target_column'] = target_column
            jobj['condition_columns'] = condition_columns
            jobj['table_id'] = table_id
            jobj['agg_index'] = agg_index
            if rowids is not None:
                jobj['target_rows'] = rowids
            if agg_index == 0:
                answers = [str(ans) for ans in answers]
                clean_answers = []
                for r in rows[1:]:
                    if cell_value_is_answer(r[target_column], answers):
                        clean_answers.append(r[target_column])
                if not clean_answers:
                    logger.info(f'no answers found! {answers} in {rows}')
                if len(clean_answers) != len(answers):
                    logger.info(f'answers changed from {answers} to {clean_answers}')
                jobj['answers'] = list(set(clean_answers))
            else:
                jobj['answers'] = answers
            if show_aggregation and rowids and len(rowids) > 1 and agg_index != 0:
                print(json.dumps(jobj))
            out.write(json.dumps(jobj)+'\n')


if __name__ == '__main__':
    class Options:
        def __init__(self):
            self.data_dir = ''


    opts = Options()
    fill_from_args(opts)

    for split in ['train', 'dev', 'test']:
        orig = os.path.join(opts.data_dir, f'{split}.jsonl')
        db_file = os.path.join(opts.data_dir, f'{split}.db')
        ans_file = os.path.join(opts.data_dir, f"{split}_ans.jsonl.gz")
        tbl_file = os.path.join(opts.data_dir, f"{split}.tables.jsonl")
        engine = DBEngine(db_file)
        exact_match = []
        with open(orig) as fs, write_open(ans_file) as fo:
            grades = []
            for ls in tqdm(fs, total=count_lines(orig)):
                eg = json.loads(ls)
                sql = eg['sql']
                qg = Query.from_dict(sql, ordered=False)
                gold = engine.execute_query(eg['table_id'], qg, lower=True)
                assert isinstance(gold, list)
                #if len(gold) != 1:
                #    print(f'for {sql} : {gold}')
                eg['answer'] = gold
                eg['rowids'] = engine.execute_query_rowid(eg['table_id'], qg, lower=True)
                # CONSIDER: if it is not an agg query, somehow identify the particular cell
                fo.write(json.dumps(eg)+'\n')

        convert(jsonl_lines(ans_file), jsonl_lines(tbl_file),
                os.path.join(opts.data_dir, f"{split}_agg.jsonl.gz"), skip_aggregation=False)
        convert(jsonl_lines(ans_file), jsonl_lines(tbl_file),
                os.path.join(opts.data_dir, f"{split}_lookup.jsonl.gz"), skip_aggregation=True)
        write_agg_classify(opts.data_dir, split)
