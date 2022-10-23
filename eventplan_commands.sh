#!/bin/bash
set -e
# @@@@@@@@@@@@@@@@@@@ event plan @@@@@@@@@@@@@@@@@@@
# =============================== train ====================
python tasks/event-plan/train.py --data_dir=datasets/event-plan/roc-stories\
 --learning_rate=1e-4 --train_batch_size=1024 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-plan --model_name event-plan-bart --experiment_name=event-plan-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=512 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-to-events-Seq2seq --experiment_name=leading-to-events-Seq2seq-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=512 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=10 --model_name_or_path=resources/external_models/gpt2 \
 --output_dir=output/generation_models --model_name leading-to-events-gpt2 --experiment_name=leading-to-events-gpt2-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=5 --accum_batches_args=1 --num_sanity_val_steps=0 \
 --save_on_train_epoch_end

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=128 --eval_batch_size=10 --model_name_or_path=resources/external_models/t5-base \
 --output_dir=output/generation_models --model_name leading-to-events-t5 --experiment_name=leading-to-events-t5-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=5 --accum_batches_args=1 --num_sanity_val_steps=0

# =============================== test ====================
python tasks/event-plan/test.py --data_dir=datasets/event-plan/roc-stories\
  --eval_batch_size=512 --model_name_or_path=output/event-plan/event-plan-bart-roc-stories/best_tfmr \
  --output_dir=output/event-plan --model_name event-plan-bart --experiment_name=event-plan-bart-roc-stories\
  --test_event_infix=_event

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories \
 --eval_batch_size=512 --model_name_or_path=output/generation_models/leading-to-events-Seq2seq-roc-stories/best_tfmr \
 --output_dir=output/generation_models --model_name leading-to-events-Seq2seq --experiment_name=leading-to-events-Seq2seq-roc-stories \
 --test_event_infix=_event --remain_sp_tokens --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories \
 --eval_batch_size=512 --model_name_or_path=output/generation_models/leading-to-events-bart-roc-stories/best_tfmr \
 --output_dir=output/generation_models --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-roc-stories \
 --test_event_infix=_event --remain_sp_tokens --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories \
 --eval_batch_size=1 --model_name_or_path=output/generation_models/leading-to-events-gpt2-roc-stories/best_tfmr \
 --output_dir=output/generation_models --model_name leading-to-events-gpt2 --experiment_name=leading-to-events-gpt2-roc-stories \
 --test_event_infix=_event --remain_sp_tokens --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories \
 --eval_batch_size=512 --model_name_or_path=output/generation_models/leading-to-events-t5-roc-stories/best_tfmr \
 --output_dir=output/generation_models --model_name leading-to-events-t5 --experiment_name=leading-to-events-t5-roc-stories \
 --test_event_infix=_event --remain_sp_tokens --accumulate_grad_batches=1

# =============================== predict ====================
python tasks/event-plan/predict.py --data_dir=datasets/event-plan/roc-stories\
  --eval_batch_size=512 --model_name_or_path=output/event-plan/event-plan-bart-roc-stories/best_tfmr \
  --output_dir=output/event-plan --model_name event-plan-bart --experiment_name=event-plan-bart-roc-stories\
  --test_event_infix=_event

# @@@@@@@@@@@@@@@@@@@ story generation @@@@@@@@@@@@@@@@@@@
# =============================== train ====================
# leading -------------------------------
python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-Seq2seq --experiment_name=leading-Seq2seq-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-bart --experiment_name=leading-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=68 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-hint --experiment_name=leading-hint-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=10 --model_name_or_path=resources/external_models/t5-base \
 --output_dir=output/generation_models --model_name leading-t5 --experiment_name=leading-t5-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0 \
 --save_on_train_epoch_end

# leading plus event -------------------------------
python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

 python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

python tasks/generation_models/train.py --data_dir=datasets/generation_models/roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=10 --model_name_or_path=resources/external_models/t5-base \
 --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=5 --accum_batches_args=1  --num_sanity_val_steps=0 \
 --save_on_train_epoch_end
 # =============================== test ====================
 # leading -------------------------------
 python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-Seq2seq --experiment_name=leading-Seq2seq-roc-stories\
  --accumulate_grad_batches=1

 python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-bart --experiment_name=leading-bart-roc-stories\
  --accumulate_grad_batches=1

 python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-hint --experiment_name=leading-hint-roc-stories\
  --accumulate_grad_batches=1

 python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-t5 --experiment_name=leading-t5-roc-stories\
  --accumulate_grad_batches=1

# leading plus event -------------------------------
# Seq2seq ----
python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
  --test_event_infix=_epb_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
  --test_event_infix=_predicted_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
  --test_event_infix=_Seq2seq_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
  --test_event_infix=_bart_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-Seq2seq-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-Seq2seq --experiment_name=leading-plus-event-Seq2seq-roc-stories\
  --test_event_infix=_gpt2_event --accumulate_grad_batches=1

# bart ----
python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_epb_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_predicted_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_Seq2seq_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_bart_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_gpt2_event --accumulate_grad_batches=1

# hint ----
python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_epb_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_predicted_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_Seq2seq_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_bart_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_gpt2_event --accumulate_grad_batches=1

# t5 ----
python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
  --test_event_infix=_epb_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
  --test_event_infix=_predicted_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
  --test_event_infix=_Seq2seq_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
  --test_event_infix=_bart_event --accumulate_grad_batches=1

python tasks/generation_models/test.py --data_dir=datasets/generation_models/roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-t5-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-t5 --experiment_name=leading-plus-event-t5-roc-stories\
  --test_event_infix=_gpt2_event --accumulate_grad_batches=1

# @@@@@@@@@@@@@@@@@@@ generation for verb event @@@@@@@@@@@@@@@@@@@
# =============================== train ====================
# leading -------------------------------
python tasks/generation_models/train.py --data_dir=datasets/generation_models/verb-roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-bart --experiment_name=leading-bart-verb-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1  --num_sanity_val_steps=0

# leading plus event -------------------------------
python tasks/generation_models/train.py --data_dir=datasets/generation_models/verb-roc-stories\
 --learning_rate=1e-4 --train_batch_size=24 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-verb-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=3  --num_sanity_val_steps=0

# leading-to-events ----------------------
python tasks/generation_models/train.py --data_dir=datasets/generation_models/verb-roc-stories\
 --learning_rate=1e-4 --train_batch_size=64 --eval_batch_size=100 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/generation_models --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-verb-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=100 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=0

 # =============================== test ====================
 # leading -------------------------------
 python tasks/generation_models/test.py --data_dir=datasets/generation_models/verb-roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-bart-verb-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-bart --experiment_name=leading-bart-verb-roc-stories\
  ---accumulate_grad_batches=1

# leading plus event -------------------------------
python tasks/generation_models/test.py --data_dir=datasets/generation_models/verb-roc-stories\
  --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-plus-event-bart-verb-roc-stories/best_tfmr \
  --output_dir=output/generation_models --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-verb-roc-stories\
  --test_event_infix=_bart_event --accumulate_grad_batches=1

# leading-to-events ----------------------
python tasks/generation_models/test.py --data_dir=datasets/generation_models/verb-roc-stories \
 --eval_batch_size=24 --model_name_or_path=output/generation_models/leading-to-events-bart-verb-roc-stories/best_tfmr \
 --output_dir=output/generation_models --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-verb-roc-stories \
 --test_event_infix=_event --remain_sp_tokens --accumulate_grad_batches=1
