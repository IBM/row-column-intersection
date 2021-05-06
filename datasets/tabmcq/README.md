# TabMCQ data processing

* Download the TabMCQ data
```bash
mkdir ${BASEDIR}/datasets/TabMCQ
cd ${BASEDIR}/datasets/TabMCQ
wget https://ai2-public-datasets.s3.amazonaws.com/tablestore-questions/TabMCQ_v_1.0.zip
unzip TabMCQ_v_1.0.zip
```

* Convert to standard format
```bash
export PYTHONPATH=whatever/row-column-intersection

python ${PYTHONPATH}/datasets/tabmcq/tabmcq_data.py \
--data_dir ${BASEDIR}/datasets/TabMCQ
```

* tables2seq_pair

This casts the problem as independent sequence-pair classification on rows and columns.
```bash
python ${PYTHONPATH}/datasets/tables2seq_pair.py \
--style lookup \
--input_dir ${BASEDIR}/datasets/TabMCQ \
--output_dir ${BASEDIR}/datasets/TabMCQ
```

* Train models initialized from the WikiSQL models (see [wikisql/README.md](../wikisql/README.md))
```bash
# NOTE: --train_instances = zcat ${BASEDIR}/datasets/TabMCQ/train/col.jsonl.gz | wc -l

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --resume_from ${BASEDIR}/datasets/wikisql_lookup/models/col_alb \
  --train_dir ${BASEDIR}/datasets/TabMCQ/train/col.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/TabMCQ/dev/col.jsonl.gz \
  --full_train_batch_size 128 --gradient_accumulation_steps 2 \
  --num_train_epochs 3 --save_per_epoch --learning_rate 2e-5 \
  --warmup_instances 10000 --train_instances 26743 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/TabMCQ/models/col_alb
  
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --resume_from ${BASEDIR}/datasets/wikisql_lookup/models/row_alb \
  --train_dir ${BASEDIR}/datasets/TabMCQ/train/row.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/TabMCQ/dev/row.jsonl.gz \
  --full_train_batch_size 64 --gradient_accumulation_steps 1 \
  --num_train_epochs 2 --learning_rate 1e-5 \
  --warmup_instances 10000 --train_instances 241340 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/TabMCQ/models/row_alb  
```

* Apply the models
```bash
export CUDA_VISIBLE_DEVICES=0 
python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path ${BASEDIR}/datasets/TabMCQ/models/col_alb --do_lower_case \
  --input_dir ${BASEDIR}/datasets/TabMCQ/dev/col.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/TabMCQ/apply/dev/col_alb
  
python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path ${BASEDIR}/datasets/TabMCQ/models/row_alb --do_lower_case \
  --input_dir ${BASEDIR}/datasets/TabMCQ/dev/row.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/TabMCQ/apply/dev/row_alb
```

* Combine the predictions and evaluate
```bash
python ${PYTHONPATH}/tableqa/seq_pair_apply2tables.py \
--row ${BASEDIR}/datasets/TabMCQ/apply/dev/row_alb \
--col ${BASEDIR}/datasets/TabMCQ/apply/dev/col_alb \
--output ${BASEDIR}/datasets/TabMCQ/apply/dev/tables_alb.jsonl.gz \
--cell_prediction_output ${BASEDIR}/datasets/TabMCQ/apply/dev/cell_predictions_alb.jsonl

python ${PYTHONPATH}/tableqa/tableqa_eval.py \
--gt ${BASEDIR}/datasets/TabMCQ/dev_lookup.jsonl.gz \
--preds ${BASEDIR}/datasets/TabMCQ/apply/dev/cell_predictions_alb.jsonl


Answerable 1.0 over 1819
MRR cell = 0.7407179474830627, column = 0.9572288393974304, row = 0.7816454768180847
Hit@1 cell = 0.6613523960113525, column = 0.9285321831703186, row = 0.7295216917991638
Hit@2 cell = 0.7487630844116211, column = 0.9730620980262756, row = 0.7751511931419373
Hit@3 cell = 0.7888950109481812, column = 0.9868059158325195, row = 0.8015393018722534
Hit@4 cell = 0.8114348649978638, column = 0.9928532242774963, row = 0.8185816407203674
Hit@5 cell = 0.8262781500816345, column = 0.9939526915550232, row = 0.8317757248878479
```