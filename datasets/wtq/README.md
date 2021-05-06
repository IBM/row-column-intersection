# WikiTableQuestions data processing

* Download the WikiTableQuestions
```bash
git clone https://github.com/ppasupat/WikiTableQuestions.git
```

* Convert to standard format
```bash
python ${BASEDIR}/row-column-intersection/datasets/wtq/wtq_data.py \
--wtq_dir ${BASEDIR}/WikiTableQuestions \
--id2split ${BASEDIR}/row-column-intersection/datasets/wtq/id2split.json

mkdir -p ${BASEDIR}/datasets/wtq
mv ${BASEDIR}/WikiTableQuestions/data/*_lookup.jsonl.gz ${BASEDIR}/datasets/wtq/.
```

* tables2seq_pair

This casts the problem as independent sequence-pair classification on rows and columns.
```bash
export PYTHONPATH=whatever/row-column-intersection

python ${PYTHONPATH}/datasets/tables2seq_pair.py \
--style lookup \
--input_dir ${BASEDIR}/datasets/wtq \
--output_dir ${BASEDIR}/datasets/wtq
```

* Train the row and column models, initialized from the WikiSQL models (see [wikisql/README.md](../wikisql/README.md))
```bash
# NOTE: --train_instances = zcat ${BASEDIR}/datasets/wtq/train/col.jsonl.gz | wc -l
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=5234 --use_env --node_rank=0 \
 ${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wtq/train/col.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wtq/dev/col.jsonl.gz \
  --resume_from ${BASEDIR}/datasets/wikisql_lookup/models/col_alb  \
  --full_train_batch_size 64 --gradient_accumulation_steps 1 \
  --num_train_epochs 4 --save_per_epoch --learning_rate 1e-5 \
  --warmup_instances 0 --train_instances 5205 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wtq/models/col_alb

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=5234 --use_env --node_rank=0 \
 ${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wtq/train/row.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wtq/dev/row.jsonl.gz \
  --resume_from ${BASEDIR}/datasets/wikisql_lookup/models/row_alb \
  --full_train_batch_size 128 --gradient_accumulation_steps 4 \
  --num_train_epochs 2 --save_per_epoch --learning_rate 5e-5 \
  --warmup_instances 0 --train_instances 24572 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wtq/models/row_alb
```

* Apply the row and column models
```bash
export CUDA_VISIBLE_DEVICES=0  # or you can distributed apply too
python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path ${BASEDIR}/datasets/wtq/models/col_alb --do_lower_case \
  --input_dir ${BASEDIR}/datasets/wtq/dev/col.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wtq/apply/dev/col_alb
  
python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path ${BASEDIR}/datasets/wtq/models/row_alb --do_lower_case \
  --input_dir ${BASEDIR}/datasets/wtq/dev/row.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wtq/apply/dev/row_alb  
```

* Combine the predictions and evaluate
```bash
python ${PYTHONPATH}/tableqa/seq_pair_apply2tables.py \
--row ${BASEDIR}/datasets/wtq/apply/dev/row_alb \
--col ${BASEDIR}/datasets/wtq/apply/dev/col_alb \
--output ${BASEDIR}/datasets/wtq/apply/dev/tables_alb.jsonl.gz \
--cell_prediction_output ${BASEDIR}/datasets/wtq/apply/dev/cell_predictions_alb.jsonl

python ${PYTHONPATH}/tableqa/tableqa_eval.py \
--gt ${BASEDIR}/datasets/wtq/dev_lookup.jsonl.gz \
--preds ${BASEDIR}/datasets/wtq/apply/dev/cell_predictions_alb.jsonl

# Should give:
Answerable 0.9758064516129032 over 124
MRR cell = 0.7757383584976196, column = 0.9233783483505249, row = 0.8136382699012756
Hit@1 cell = 0.7016128897666931, column = 0.9032257795333862, row = 0.75
Hit@2 cell = 0.7983871102333069, column = 0.9274193644523621, row = 0.8306451439857483
Hit@3 cell = 0.8387096524238586, column = 0.9354838728904724, row = 0.8709677457809448
Hit@4 cell = 0.8467742204666138, column = 0.9435483813285828, row = 0.8709677457809448
Hit@5 cell = 0.8548387289047241, column = 0.9435483813285828, row = 0.8790322542190552

```