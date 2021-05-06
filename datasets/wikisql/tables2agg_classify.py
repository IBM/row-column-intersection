import logging
import ujson as json
from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
import os

logger = logging.getLogger(__name__)


def write_agg_classify(data_dir, split, *, exclude_header=False, cell_sep_token='*'):
    with write_open(os.path.join(data_dir, f'{split}_agg_classify.jsonl.gz')) as out:
        for line in jsonl_lines(os.path.join(data_dir, f'{split}_agg.jsonl.gz')):
            jobj = json.loads(line)
            if not exclude_header:
                agg_inst = {'id': jobj['id'],
                            'text_a': jobj['question'],
                            'text_b': f' {cell_sep_token} '.join(jobj['header']),
                            'label': jobj['agg_index']}
            else:
                agg_inst = {'id': jobj['id'], 'text': jobj['question'], 'label': jobj['agg_index']}
            out.write(json.dumps(agg_inst) + '\n')


class Options:
    def __init__(self):
        self.data_dir = ''
        self.exclude_header = False
        self.cell_sep_token = '*'  # used to use '|' but albert tokenizer doesn't like it
        self.__required_args__ = ['data_dir']


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)

    for split in ['train', 'dev', 'test']:
        write_agg_classify(opts.data_dir, split, exclude_header=opts.exclude_header, cell_sep_token=opts.cell_sep_token)

