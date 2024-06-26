cat $0
echo "--------------------"

export PYTHONPATH=$PWD
test_set="data/splits/test.csv"
cache_dir="cache"
output_dir="mt_results"

python src/mt/zero_shot_lm_eval.py \
    --model_name "bigscience/mt0-xxl" \
    --test_set $test_set \
    --cache_dir $cache_dir \
    --output_dir $output_dir