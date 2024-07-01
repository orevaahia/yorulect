cat $0
echo "--------------------"

export PYTHONPATH=$PWD
data_set="data/splits"
cache_dir="cache"
output_dir="finetuned_mt"
num_gpus=2
model_name_stem="yo_nllb"

# this command uses default parameters to train best model
# train each dialect as a separate language
for dialect in Ife Ijebu Ilaje Standard
do 
    echo "Finetuning on $dialect"
    python src/mt/finetune_mt.py \
        --num_gpus $num_gpus \
        --dataset_dir $data_set \
        --model_output_dir $output_dir \
        --checkpoint "facebook/nllb-200-distilled-600M" \
        --single_dialect $dialect \
        --trained_model_name "${model_name_stem}_${dialect}" \
        --cache_dir $cache_dir \ 
done

