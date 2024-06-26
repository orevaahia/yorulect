cat $0
echo "--------------------"
export PYTHONPATH=$PWD
DIALECTS=("std" "ife" "ilaje")
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
num_gpus=2


for dial in "${DIALECTS[@]}";do
    dataset="data/aligned_speech/${dial}"
    output_dir="results/mms/${dial}_${current_datetime}"

    run_name=$(echo "$output_dir" | awk -F'/' '{print $NF}')

    echo 'Finding free port'
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    torchrun \
        --nproc_per_node $num_gpus --master_port $PORT \
        src/speech/finetune_mms_asr.py \
        --dataset_dir ${dataset} \
        --model_name_or_path "facebook/mms-300m" \
        --output_dir ${output_dir} \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --eval_strategy epoch \
        --learning_rate 1e-3 \
        --text_column_name "transcription" \
        --length_column_name "input_length" \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --save_total_limit 2 \
        --target_language "yor" \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
        --fp16 \
        --do_train \
        --do_eval \
        --do_predict \
        --report_to "wandb" \
        --run_name $run_name \
        --cache_dir "cache" \
        --load_best_model_at_end True \
        --metric_for_best_model "wer" \
        --greater_is_better False \
        --seed 42

done