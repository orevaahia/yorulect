cat $0
echo "--------------------"

export PYTHONPATH=$PWD
test_set="data/splits/test.csv"
cache_dir="cache"
output_dir="mt_results"

# make sure $GOOGLE_CLOUD_PROJECT and GOOGLE_API_KEY environment variables are set
python src/mt/zero_shot_gmnmt_eval.py \
    --project_id $GOOGLE_CLOUD_PROJECT \
    --test_set $test_set \
    --cache_dir $cache_dir \
    --output_dir $output_dir