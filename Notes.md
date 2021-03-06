## CnDbpedia Chinese
CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/book_review/train.tsv \
    --dev_path ./datasets/book_review/dev.tsv \
    --test_path ./datasets/book_review/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
    > ./outputs/kbert_bookreview_CnDbpedia.log

## HowNet Chinese
CUDA_VISIBLE_DEVICES='1' python -u run_kbert_cls.py \
  --pretrained_model_path ./models/google_model.bin \
  --config_path ./models/google_config.json \
  --vocab_path ./models/google_vocab.txt \
  --train_path ./datasets/book_review/train.tsv \
  --dev_path ./datasets/book_review/dev.tsv \
  --test_path ./datasets/book_review/test.tsv \
  --epochs_num 5 --report_steps 100 --batch_size 8 \
  --kg_name HowNet \
  --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin


## Ai Paper English
CUDA_VISIBLE_DEVICES='1' python -u run_kbert_cls.py \
  --pretrained_model_path ./models/google_model_en_uncased_base.bin \
  --config_path ./models/google_config.json \
  --vocab_path ./models/google_vocab_en.txt \
  --train_path ./datasets/vivek/train_new.tsv \
  --dev_path ./datasets/vivek/dev_new.tsv \
  --test_path ./datasets/vivek/test_new.tsv \
  --epochs_num 20 --report_steps 100 --batch_size 32 --workers_num 8 \
  --kg_name HowNet_newt \
  --output_model_path ./outputs/kbert_vivek_vm_kg_pt.bin


## Ai Paper English (No Visible Matrix)
CUDA_VISIBLE_DEVICES='1' python -u run_kbert_cls.py \
  --pretrained_model_path ./models/google_model_en_uncased_base.bin \
  --config_path ./models/google_config.json \
  --vocab_path ./models/google_vocab_en.txt \
  --train_path ./datasets/vivek/train_new.tsv \
  --dev_path ./datasets/vivek/dev_new.tsv \
  --test_path ./datasets/vivek/test_new.tsv \
  --epochs_num 20 --report_steps 100 --batch_size 48 --workers_num 8 \
  --kg_name none \
  --no_vm \
  --output_model_path ./outputs/kbert_vivek_pt.bin


## Ai Paper English (No Visible Matrix and No Pretrained)
CUDA_VISIBLE_DEVICES='1' python -u run_kbert_cls.py \
  --config_path ./models/google_config.json \
  --vocab_path ./models/google_vocab_en.txt \
  --train_path ./datasets/vivek/train_new.tsv \
  --dev_path ./datasets/vivek/dev_new.tsv \
  --test_path ./datasets/vivek/test_new.tsv \
  --epochs_num 5 --report_steps 100 --batch_size 48 --workers_num 8 \
  --kg_name none \
  --no_vm \
  --output_model_path ./outputs/kbert_vivek.bin
