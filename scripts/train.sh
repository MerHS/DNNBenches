python bench.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --do_train \
    --fp16 \
    --seed 42 \
    --preprocessing_num_workers 8 \
    --output_dir test-clm 
