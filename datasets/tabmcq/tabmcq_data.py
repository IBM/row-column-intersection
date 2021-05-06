import logging
import os
import ujson as json
from util.line_corpus import write_open, read_open
import csv
import random
from util.args_help import fill_from_args

logger = logging.getLogger(__name__)


def convert_queries(query_tsv, table_dir, out_dir):
    # NOTE: there are some tables with a large number of rows,
    # when creating instances, we limit the total number of rows in the table
    per_question_row_limit = 50
    dev_percent = 20
    test_percent = 20

    rand = random.Random(1234)
    tid2rows = dict()
    row_search_out = write_open(os.path.join(out_dir, 'row_pseudo_docs.jsonl'))
    print(f'Table\tRow Count\tHeader')
    for subdir in ['auto', 'monarch', 'regents']:
        subdir_full = os.path.join(table_dir, subdir)
        for file in os.listdir(subdir_full):
            assert file[-4:] == '.tsv'
            table_id = subdir+'-'+file[:-4]
            all_rows = []
            with read_open(os.path.join(subdir_full, file)) as csvfile:
                for rndx, parts in enumerate(csv.reader(csvfile, doublequote=False, delimiter='\t')):
                    row = [c.strip() for c in parts]
                    all_rows.append(row)
                    if rndx > 0:
                        pdoc = dict()
                        # pdoc['contents'] = table_id + ' ' + ' | '.join([h+' : '+c for h, c in zip(all_rows[0], row)])
                        pdoc['contents'] = table_id + ' ' + ' | '.join(row)
                        pdoc['id'] = table_id + ':' + str(rndx-1)
                        row_search_out.write(json.dumps(pdoc)+'\n')
            tid2rows[table_id] = all_rows
            print(f'{table_id}\t{len(all_rows)-1}\t{all_rows[0]}')
    print('\n\n')

    split_files = []
    for split in ['dev_lookup.jsonl.gz', 'test_lookup.jsonl.gz', 'train_lookup.jsonl.gz']:
        split_files.append(write_open(os.path.join(out_dir, split)))
    question_over_row_limit_count = 0
    with read_open(query_tsv) as csvfile:
        for ndx, parts in enumerate(csv.reader(csvfile, doublequote=False, delimiter='\t')):
            if ndx == 0:
                # QUESTION        QUESTION-ALIGNMENT      CHOICE 1        CHOICE 2        CHOICE 3        CHOICE 4
                # CORRECT CHOICE  RELEVANT TABLE  RELEVANT ROW    RELEVANT COL
                continue
            if len(parts) != 10:
                print(f'bad line: {parts}')
                exit(1)
            qid = f'q{ndx}'
            qtext = parts[0].strip()
            # the part of table[relevant_row] used to construct the question
            question_alignment = [int(c.strip()) for c in parts[1].split(',')]
            choices = [c.strip() for c in parts[2:6]]
            answer = choices[int(parts[6])-1]
            table_id = parts[7]
            target_row = int(parts[8])-1  # -1 for header
            target_column = int(parts[9])
            all_rows = tid2rows[table_id]
            header = all_rows[0]
            rows = all_rows[1:]
            if target_column in question_alignment:
                question_alignment.remove(target_column)
            # t_ans = re.sub(r'\W+', '', rows[target_row][target_column].lower())
            # q_ans = re.sub(r'\W+', '', answer.lower())
            # if t_ans not in q_ans and q_ans not in t_ans:
            #    print(f'{answer} != {rows[target_row][target_column]} in ({table_id}, {target_row}, {target_column})')
            answer = rows[target_row][target_column]
            if 0 < per_question_row_limit < len(rows):
                pos_row = rows[target_row]
                neg_rows = rows[:target_row] + rows[target_row+1:]
                rand.shuffle(neg_rows)
                rows = neg_rows[:per_question_row_limit]
                target_row = rand.randint(0, len(rows)-1)
                rows[target_row] = pos_row
                question_over_row_limit_count += 1
            jobj = dict()
            jobj['id'] = qid
            jobj['question'] = qtext
            jobj['header'] = header
            jobj['rows'] = rows
            jobj['target_column'] = target_column
            jobj['answers'] = [answer]
            jobj['table_id'] = table_id
            # extra
            jobj['target_row'] = target_row
            jobj['choices'] = choices  # but note that these could fail to match any cell...
            jobj['condition_columns'] = question_alignment
            line = json.dumps(jobj) + '\n'
            # CONSIDER: this split is bad - we really should split on table rather than just randomly
            if ndx % 100 < dev_percent:
                split_files[0].write(line)
            elif ndx % 100 < dev_percent + test_percent:
                split_files[1].write(line)
            else:
                split_files[2].write(line)
    for split_file in split_files:
        split_file.close()
    print(f'{question_over_row_limit_count} questions over row limit')


if __name__ == '__main__':
    class Options:
        def __init__(self):
            self.data_dir = ''


    opts = Options()
    fill_from_args(opts)

    convert_queries(os.path.join(opts.data_dir, 'MCQs.tsv'),
                    os.path.join(opts.data_dir, 'Tables'),
                    opts.data_dir)
