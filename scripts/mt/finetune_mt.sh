cat $0
echo "--------------------"

export PYTHONPATH=$PWD
data_set="data/splits"
cache_dir="cache"
output_dir="finetuned_mt"
num_gpus=2

# run `src/mt/finetune_mt.py --help` to see full options
# this command uses default parameters to train our best model
python src/mt/finetune_mt.py \
    --num_gpus $num_gpus \
    --dataset_dir $data_set \
    --model_output_dir $output_dir \
    --cache_dir $cache_dir \
    --checkpoint "facebook/nllb-200-distilled-600M" \
