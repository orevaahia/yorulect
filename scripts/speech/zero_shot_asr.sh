cat $0
echo "--------------------"

export PYTHONPATH=$PWD

DIALECTS=("ilaje" "ife" "std")

model_name="facebook/mms-1b-all"
#model_name="openai/whisper-large-v2"


for dial in "${DIALECTS[@]}";do
        dataset="data/aligned_speech/${dial}/test/"
        output_dir="output_dir/${dial}"
        mkdir -p ${output_dir}

        python src/evaluate_mms_asr.py \
                --model_name $model_name \
                --language "yor" \
                --dataset ${dataset} \
                --batch_size 16 \
                --output_dir ${output_dir} \
                --cache_dir "cache"
        echo "Finished with ${dial}"

done
