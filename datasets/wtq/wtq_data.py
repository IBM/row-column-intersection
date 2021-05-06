import logging
from util.line_corpus import read_open, write_open, read_lines
from util.args_help import fill_from_args
import ujson as json
import regex as re
import os
import csv

logger = logging.getLogger(__name__)

spaces = re.compile(r'\s+', flags=re.MULTILINE)
non_word = re.compile(r'\W+', flags=re.MULTILINE)


class Options:
    def __init__(self):
        self.wtq_dir = ''
        self.id2split = ''
        self.match_cell_substring = 0.75
        self.__required_args__ = ['wtq_dir', 'id2split']


def normalize(cv):
    # just special case these
    if cv == 'Muneca brava':
        cv = 'Muñeca brava'
    elif cv == 'Costa Rican':
        cv = 'Costa Rica'

    cvr = re.sub(spaces, ' ', re.sub(non_word, ' ', cv)).strip()
    if not cvr:
        cvr = re.sub(spaces, ' ', cv).strip()
    return cvr


def tsv_unescape(part):
    fixed = part.replace('\\n', '\n').replace('\\p', '|').replace('\\\\', '\\')
    if fixed != part:
        print(f'unescaped from {part} to {fixed}')
    return fixed


def main():
    opts = Options()
    fill_from_args(opts)

    # The escaped characters include: double quote (" => \") and backslash (\ => \\).
    # Newlines are represented as quoted line breaks.
    csv_base_dir = os.path.join(opts.wtq_dir, 'csv')
    id2rows = dict()
    for dir in os.listdir(csv_base_dir):
        full_dir = os.path.join(csv_base_dir, dir)
        for file in os.listdir(full_dir):
            with read_open(os.path.join(full_dir, file)) as csvfile:
                rows = []
                for row in csv.reader(csvfile, doublequote=False, escapechar='\\'):
                   rows.append(row)
                id2rows[f'csv/{dir}/{file}'] = rows

    # List items are separated by | (e.g., when|was|taylor|swift|born|?).
    # The following characters are escaped: newline (=> \n), backslash (\ => \\), and pipe (| => \p)
    # Note that pipes become \p so that doing x.split('|') will work.
    data_dir = os.path.join(opts.wtq_dir, 'data')
    with read_open(os.path.join(opts.id2split)) as ids_file:
        id2split = json.load(ids_file)
    splits = {split_name: write_open(os.path.join(data_dir, f'{split_name}_lookup.jsonl.gz'))
              for split_name in ['train', 'dev', 'test']}

    matched_by_substring = 0
    for infile in ['training.tsv', 'pristine-seen-tables.tsv', 'pristine-unseen-tables.tsv']:
        for ndx, line in enumerate(read_lines(os.path.join(data_dir, infile))):
            parts = line.strip().split('\t')
            assert len(parts) == 4
            if ndx == 0:
                continue
            id = parts[0]
            if id not in id2split:
                continue
            split_name = id2split[id]
            query = tsv_unescape(parts[1])
            table_id = parts[2]
            answers = [tsv_unescape(p) for p in parts[3].split('|')]
            norm_answers = [normalize(answer) for answer in answers]
            all_rows = id2rows[table_id]
            header = all_rows[0]
            rows = all_rows[1:]

            # force 'answers' to contain only string equal matches to cell values
            target_columns = set()
            matched_answers = set()
            for rndx, row in enumerate(all_rows):
                for cndx, cell in enumerate(row):
                    if normalize(cell) in norm_answers:
                        target_columns.add(cndx)
                        matched_answers.add(cell)
            if opts.match_cell_substring < 1.0 and len(target_columns) == 0:
                for rndx, row in enumerate(all_rows):
                    for cndx, cell in enumerate(row):
                        ncell = normalize(cell)
                        if any([answer in ncell and len(answer)/len(ncell) >= opts.match_cell_substring
                                for answer in norm_answers]):
                            target_columns.add(cndx)
                            matched_answers.add(cell)
                if len(target_columns) > 0:
                    matched_by_substring += 1

            out = splits[split_name]
            if len(target_columns) == 0:
                print(f'{query} {answers} not found in table: \n{all_rows}')
            elif len(target_columns) > 1:
                print(f'{query} {answers} multiple columns match answer: \n{all_rows}')
            else:
                jobj = dict()
                jobj['id'] = id
                jobj['table_id'] = table_id
                jobj['question'] = query
                jobj['header'] = header
                jobj['target_column'] = list(target_columns)[0]
                answers = list(matched_answers)
                answers.sort()
                jobj['answers'] = answers
                jobj['rows'] = rows
                out.write(json.dumps(jobj) + '\n')

    print(f'matched by substring: {matched_by_substring}')
    for f in splits.values():
        f.close()


if __name__ == "__main__":
    main()
