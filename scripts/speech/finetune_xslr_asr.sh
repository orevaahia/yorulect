cat $0
echo "--------------------"
export PYTHONPATH=$PWD
DIALECTS=("std" "ife" "ilaje")
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
num_gpus=2


for dial in "${DIALECTS[@]}";do
    dataset="data/aligned_speech/${dial}"
    output_dir="results/xlsr/${dial}_${current_datetime}"

    run_name=$(echo "$output_dir" | awk -F'/' '{print $NF}')

    echo 'Finding free port'
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    python src/speech/finetune_xslr.py \
        --dataset_dir ${dataset} \
        --model_name_or_path "facebook/wav2vec2-xls-r-1b" \
        --output_dir ${output_dir} \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --eval_strategy epoch \
        --learning_rate 3e-4 \
        --text_column_name "transcription" \
        --logging_steps 50 \
        --layerdrop "0.0" \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --eval_metrics wer cer \
        --save_total_limit 2 \
        --mask_time_prob "0.3" \
        --mask_time_length "10" \
        --mask_feature_prob "0.1" \
        --mask_feature_length "64" \
        --freeze_feature_encoder \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
        --max_duration_in_seconds "20" \
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
        --seed 42 \
        --warmup_ratio 0.1 \
        --gradient_checkpointing

done