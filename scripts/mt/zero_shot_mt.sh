cat $0
echo "--------------------"

export PYTHONPATH=$PWD
test_set="data/splits/test.csv"
cache_dir="/data/dabagyan/"
output_dir="mt_results"

python src/mt/zero_shot_mt_eval.py \
    --model_name "facebook/nllb-200-distilled-600M" \
    --test_set $test_set \
    --cache_dir $cache_dir \
    --output_dir $output_dir