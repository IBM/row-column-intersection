# WikiSQL-Lookup and WikiSQL (full) data processing

* Download the WikiSQL code and data
```bash
git clone https://github.com/salesforce/WikiSQL.git
cd WikiSQL
pip install -r requirements.txt
# 1.4 sqlalchemy is incompatible
pip install sqlalchemy==1.3.23
tar xvjf data.tar.bz2
export PYTHONPATH=${BASEDIR}/row-column-intersection:${BASEDIR}/WikiSQL
```
* Convert to standard format
```bash
python ${BASEDIR}/row-column-intersection/datasets/wikisql/wikisql_data.py \
--data_dir ${BASEDIR}/WikiSQL/data
```


* tables2seq_pair

This casts the problem as independent sequence-pair classification on rows and columns.
```bash
export PYTHONPATH=${BASEDIR}/row-column-intersection

python ${PYTHONPATH}/datasets/tables2seq_pair.py \
--style agg \
--input_dir ${BASEDIR}/WikiSQL/data \
--output_dir ${BASEDIR}/datasets/wikisql

python ${PYTHONPATH}/datasets/tables2seq_pair.py \
--style lookup \
--input_dir ${BASEDIR}/WikiSQL/data \
--output_dir ${BASEDIR}/datasets/wikisql_lookup

mv ${BASEDIR}/WikiSQL/data/*agg_classify.jsonl.gz ${BASEDIR}/datasets/wikisql/.
```

The commands below use multi-GPU training with DistributedDataParallel. 
You can also use single GPU training with more gradient accumulation.

# WikiSQL-Lookup

* Train the row and column models

```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql_lookup/train/col.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql_lookup/dev/col.jsonl.gz \
  --full_train_batch_size 64 --gradient_accumulation_steps 2 \
  --num_train_epochs 3 --learning_rate 2e-5 \
  --warmup_instances 100000 --train_instances 360664 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql_lookup/models/col_alb

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-base-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql_lookup/train/row.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql_lookup/dev/row.jsonl.gz \
  --full_train_batch_size 128 --gradient_accumulation_steps 4 \
  --num_train_epochs 2 --save_per_epoch --learning_rate 2e-5 \
  --warmup_instances 200000 --train_instances 957458 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql_lookup/models/row_alb

```

* Apply the row and column models

```bash
export CUDA_VISIBLE_DEVICES=0 
python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path ${BASEDIR}/datasets/wikisql_lookup/models/col_alb --do_lower_case \
  --input_dir ${BASEDIR}/datasets/wikisql_lookup/dev/col.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql_lookup/apply/dev/col_alb

python ${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path {BASEDIR}/datasets/wikisql_lookup/models/row_alb --do_lower_case \
  --input_dir {BASEDIR}/datasets/wikisql_lookup/dev/row.jsonl.gz \
  --max_seq_length 256 \
  --output_dir {BASEDIR}/datasets/wikisql_lookup/apply/dev/row_alb
```

* Merge the row and column results and evaluate
```bash
python ${PYTHONPATH}/tableqa/seq_pair_apply2tables.py \
--row ${BASEDIR}/datasets/wikisql_lookup/apply/dev/row_alb \
--col ${BASEDIR}/datasets/wikisql_lookup/apply/dev/col_alb \
--output ${BASEDIR}/datasets/wikisql_lookup/apply/dev/tables_alb.jsonl.gz \
--cell_prediction_output ${BASEDIR}/datasets/wikisql_lookup/apply/dev/cell_predictions_alb.jsonl


python ${PYTHONPATH}/tableqa/tableqa_eval.py \
--gt ${BASEDIR}/datasets/wikisql_lookup/dev_lookup.jsonl.gz \
--preds ${BASEDIR}/datasets/wikisql_lookup/apply/dev/cell_predictions_alb.jsonl


Answerable 1.0 over 6017
MRR cell = 0.9725173115730286, column = 0.9869083762168884, row = 0.9851437211036682
Hit@1 cell = 0.9577862620353699, column = 0.9790593385696411, row = 0.977895975112915
Hit@2 cell = 0.9780621528625488, column = 0.9903606176376343, row = 0.9873691201210022
Hit@3 cell = 0.9842113852500916, column = 0.9936845898628235, row = 0.9901944398880005
Hit@4 cell = 0.9888648986816406, column = 0.9963436722755432, row = 0.9921888113021851
Hit@5 cell = 0.9911916255950928, column = 0.9970085024833679, row = 0.9941831231117249
```

# WikiSQL including aggregation questions

* Train and apply the aggregation classification
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql/train_agg_classify.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql/dev_agg_classify.jsonl.gz \
  --full_train_batch_size 256 --gradient_accumulation_steps 32 \
  --num_train_epochs 2 --learning_rate 4e-5 \
  --warmup_instances 10000 --train_instances 56355 \
  --weight_decay 0.01 --max_seq_length 128 --num_labels 6 \
  --output_dir ${BASEDIR}/datasets/wikisql/models/agg_classify_alb_xxl

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --resume_from ${BASEDIR}/datasets/wikisql/models/agg_classify_alb_xxl \
  --input_dir ${BASEDIR}/datasets/wikisql/dev_agg_classify.jsonl.gz \
  --max_seq_length 128 --num_labels 6 \
  --output_dir ${BASEDIR}/datasets/wikisql/apply/dev/agg_alb_xxl
```

* Train and apply the row and column models

  * Column Representation Model
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql/train/col.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql/dev/col.jsonl.gz \
  --full_train_batch_size 128 --gradient_accumulation_steps 32 \
  --num_train_epochs 3 --save_per_epoch --learning_rate 1e-5 \
  --warmup_instances 100000 --train_instances 360664 \
  --weight_decay 0.01 --max_seq_length 256  --is_separate --save_per_epoch \
  --output_dir ${BASEDIR}/datasets/wikisql/models/col_repr_alb_xxl

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --resume_from ${BASEDIR}/datasets/wikisql/models/col_repr_alb_xxl \
  --input_dir ${BASEDIR}/datasets/wikisql/dev/col.jsonl.gz \
  --max_seq_length 256 --is_separate \
  --output_dir ${BASEDIR}/datasets/wikisql/apply/dev/col_repr_alb_xxl

```

  * Column Interaction Model
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql/train/col.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql/dev/col.jsonl.gz \
  --full_train_batch_size 128 --gradient_accumulation_steps 16 \
  --num_train_epochs 2 --save_per_epoch --learning_rate 1e-5 \
  --warmup_instances 100000 --train_instances 360664 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql/models/col_alb_xxl

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --resume_from ${BASEDIR}/datasets/wikisql/models/col_alb_xxl \
  --input_dir ${BASEDIR}/datasets/wikisql/dev/col.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql/apply/dev/col_alb_xxl
```


  * Row Interaction Model
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/classify_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
  --train_dir ${BASEDIR}/datasets/wikisql/train/row.jsonl.gz \
  --dev_dir ${BASEDIR}/datasets/wikisql/dev/row.jsonl.gz \
  --full_train_batch_size 512 --gradient_accumulation_steps 64 \
  --num_train_epochs 2 --learning_rate 1e-5 \
  --warmup_instances 200000 --train_instances 957458 \
  --weight_decay 0.01 --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql/models/row_alb_xxl

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
${PYTHONPATH}/torch_util/apply_seq_pair.py \
  --model_type albert --model_name_or_path albert-xxlarge-v2 --do_lower_case \
 --resume_from ${BASEDIR}/datasets/wikisql/models/row_alb_xxl \
  --input_dir ${BASEDIR}/datasets/wikisql/dev/row.jsonl.gz \
  --max_seq_length 256 \
  --output_dir ${BASEDIR}/datasets/wikisql/apply/dev/row_alb_xxl

```

* Get TAPAS gold standard file for comparison

wget command to get NSM gold json files is from:
https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/aws_setup.sh
```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lwLH4-5FRZzM9JVicy3TH6Al11bRalyg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lwLH4-5FRZzM9JVicy3TH6Al11bRalyg" -O wikisql.zip && rm -rf /tmp/cookies.txt

jar -xvf wikisql.zip
ls wikisql/raw_input/*_gold.json
mv wikisql/raw_input/*_gold.json ${BASEDIR}/datasets/wikisql
```

* Aggregation Integration
```bash
python ${PYTHONPATH}/tableqa/seq_pair_apply2tables.py \
--row ${BASEDIR}/datasets/wikisql/apply/dev/row_alb_xxl \
--col ${BASEDIR}/datasets/wikisql/apply/dev/col_alb_xxl \
--gt ${BASEDIR}/datasets/wikisql/dev_agg.jsonl.gz \
--output ${BASEDIR}/datasets/wikisql/apply/dev/tables_alb_xxl.jsonl.gz


python ${PYTHONPATH}/tableqa/aggregation_integration.py \
--agg_preds ${BASEDIR}/datasets/wikisql/apply/dev/agg_alb_xxl \
--cell_preds ${BASEDIR}/datasets/wikisql/apply/dev/tables_alb_xxl.jsonl.gz \
--gt ${BASEDIR}/datasets/wikisql/dev_agg.jsonl.gz \
--prediction_file ${BASEDIR}/datasets/wikisql/apply/dev/aggregation_integration_alb_xxl.jsonl


python ${PYTHONPATH}/tableqa/tapas_eval.py \
--reference_file ${BASEDIR}/datasets/wikisql/dev_gold.json \
--prediction_file ${BASEDIR}/datasets/wikisql/apply/dev/aggregation_integration_alb_xxl.jsonl

```