from typing import Any, List, Dict, Tuple
import torch
import numpy as np
from transformers import (
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    PreTrainedModel
)


class TableQAOptions:
    def __init__(self):
        self.model_type = 'albert'
        self.tokenizer = 'albert-base-v2'
        self.row_model = 'michaelrglass/albert-base-rci-wikisql-row'
        self.col_model = 'michaelrglass/albert-base-rci-wikisql-col'
        self.max_seq_length = 128
        self.batch_size = 16
        self.top_k = 5
        self.device = 'cuda'


class RCISystem(object):
    """
    Interactive TableQA system using the Row Column Intersection Model.
    https://www.aclweb.org/anthology/2021.naacl-main.96/
    """

    def __init__(self, opts: TableQAOptions):
        self.opts = opts
        self._device = torch.device(self.opts.device)
        # select model class
        model_classes = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
        }
        config_class, model_class, tokenizer_class = model_classes[self.opts.model_type.lower()]
        # load tokenizer and models
        self.tokenizer = tokenizer_class.from_pretrained(self.opts.tokenizer)
        self.row_model = self._load(model_class, self.opts.row_model)
        self.col_model = self._load(model_class, self.opts.col_model)

    def _load(self, model_class, model_name_or_path: str) -> PreTrainedModel:
        model = model_class.from_pretrained(model_name_or_path)
        model.to(self._device)
        model.eval()
        return model

    @staticmethod
    def row_column_strings(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[str]]:
        row_reps = []
        col_reps = []
        cols = [[str(h)] for h in header]
        for row in rows:
            row_rep = ' * '.join([h + ' : ' + str(c) for h, c in zip(header, row) if c])  # for sparse table use case
            row_reps.append(row_rep)
            for ci, cell in enumerate(row):
                if cell:  # for sparse table use case
                    cols[ci].append(str(cell))
        for col in cols:
            col_rep = ' * '.join(col)
            col_reps.append(col_rep)
        return row_reps, col_reps

    def _repr_to_input(self, query: str, reprs: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=[(query, rr) for rr in reprs],
                                                  max_length=self.opts.max_seq_length,
                                                  add_special_tokens=True, return_tensors='pt',
                                                  padding='longest', truncation=True)
        return {k: t.to(self._device) for k, t in inputs.items()}

    def _top_k_cells(self, row_logits: np.ndarray, col_logits: np.ndarray) -> List[Tuple[int, int, float]]:
        all_scores = []
        for ri, rs in enumerate(row_logits):
            for ci, cs in enumerate(col_logits):
                s = float(rs + cs)
                all_scores.append((ri, ci, s))
        all_scores.sort(key=lambda x: x[2], reverse=True)
        return all_scores[0:self.opts.top_k]

    def _batched_score(self, query, reps, model):
        logits = np.zeros(len(reps), dtype=np.float32)
        for start in range(0, len(reps), self.opts.batch_size):
            end = start + self.opts.batch_size
            with torch.no_grad():
                inputs = self._repr_to_input(query, reps[start:end])
                logits[start:end] = model(**inputs)[0].detach().cpu().numpy()[:, 1]
        return logits

    def _apply(self, query: str, header: List[str], rows: List[List[str]]) -> List[Tuple[int, int, float]]:
        row_reps, col_reps = RCISystem.row_column_strings(header, rows)
        col_logits = self._batched_score(query, col_reps, self.col_model)
        row_logits = self._batched_score(query, row_reps, self.row_model)
        return self._top_k_cells(row_logits, col_logits)

    def get_answer_columns(self, question: str, header: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Score columns (but not rows) for probability they contain the answer
        :param str question: the question
        :param List[str] header: the table header
        :param List[List[str]] rows: the table rows
        :return: Column predictions in descending score order
        """
        _, col_reps = RCISystem.row_column_strings(header, rows)
        col_logits = self._batched_score(question, col_reps, self.col_model)
        all_scores = []
        for ci, cs in enumerate(col_logits):
            all_scores.append((ci, float(cs)))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return [{'col_ndx': ci, 'confidence_score': s} for ci, s in all_scores[0:self.opts.top_k]]

    def get_answer_rows(self, question: str, header: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Scores rows (but not colums) for probability they contain the answer
        :param str question: the question
        :param List[str] header: the table header
        :param List[List[str]] rows: the table rows
        :return: Row predictions in descending score order
        """
        row_reps, _ = RCISystem.row_column_strings(header, rows)
        row_logits = self._batched_score(question, row_reps, self.row_model)
        all_scores = []
        for ri, rs in enumerate(row_logits):
            all_scores.append((ri, float(rs)))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return [{'row_ndx': ri, 'confidence_score': s} for ri, s in all_scores[0:self.opts.top_k]]

    def get_answers(self, question: str, header: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """
                Computes the answers to the question in the passage
                :param str question: the question
                :param List[str] header: the table header
                :param List[List[str]] rows: the table rows
                :return: Cell prediction answers in descending score order
                :rtype: List[Dict[str, Any]]
        """
        cells = self._apply(question, header, rows)
        # return a list of dicts
        return [{'row_ndx': ri, 'col_ndx': ci, 'confidence_score': s, 'text': rows[ri][ci]} for ri, ci, s in cells]
