# training scripts of DocRepair model

echo "host name is: `hostname`, on GPU: $2"
echo "wording directory is: `pwd`"
echo "bash command for this execuation is: `echo $0 $@`"

source # activate conda environment

script_dir=`dirname $0`
main_dir=$script_dir/../
data_dir= # insert the directory storing preprocessed data
working_dir=$main_dir/$1
consis_dir= # <directory of good-translation-wrong-in-context>/consistency_testsets/scoring_data/

mkdir -p $working_dir

# variables (toolkits; source and target language)
. $main_dir/vars
# change the code respository to 2-way branch
nematus_home= # directory of code respository of 2-way branch

devices=$2 # GPU ID

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --datasets $data_dir/train.lc.bpe.gp.$mt \
               $data_dir/train.lc.bpe.gp.$tgt \
    --f_datasets $data_dir/fine_tune/train.lc.bpe.gp.$mt \
                 $data_dir/fine_tune/train.lc.bpe.gp.$tgt \
    --dictionaries $data_dir/train.lc.bpe.gp.$tgt.json \
                   $data_dir/train.lc.bpe.gp.$tgt.json \
    --save_freq 4500 \
    --model $working_dir/model \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size 512 \
    --state_size 512 \
    --tie_decoder_embeddings \
    --tie_encoder_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule transformer \
    --warmup_steps 8000 \
    --maxlen 500 \
    --batch_size 50 \
    --token_batch_size 15000 \
    --valid_datasets $data_dir/dev.lc.bpe.gp.$mt \
                     $data_dir/dev.lc.bpe.gp.$tgt \
    --valid_deixis_datasets $consis_dir/deixis_dev.lc.bpe.gp.$mt \
                            $consis_dir/deixis_dev.lc.bpe.gp.$tgt \
    --valid_cohesion_datasets $consis_dir/lex_cohesion_dev.lc.bpe.gp.$mt \
                              $consis_dir/lex_cohesion_dev.lc.bpe.gp.$tgt \
    --valid_batch_size 25 \
    --valid_token_batch_size 2000 \
    --valid_freq 4500 \
    --valid_script $script_dir/validate.sh \
    --valid_consis_script $script_dir/validate_consis.sh \
    --disp_freq 1000 \
    --summary_freq 0 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 500 \
    --normalization_alpha 0.6 \
    --max_tokens_per_device 2000 \
    --data_mode monlig \
    --chunk_size 2500000 \
    --f_ratio $3 \
    --patience 999999 \


