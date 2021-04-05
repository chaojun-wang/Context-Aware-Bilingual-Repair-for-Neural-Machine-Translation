#!/bin/sh

# evaluate BLEU score on general test set
echo "evaluate general test set"
. vars

working_dir=$1
script_dir=scripts
data_dir= # insert the directory storing preprocessed data

devices=3 # GPU device number 

test=test.lc.bpe.gp.$src
test_p=test.lc.bpe.gp.$mt
ref=test.lc.gp.single.$tgt
model=$working_dir/${2:-model.best-valid-script}


mkdir $working_dir/$2.$3.$4.$5
# decode
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
	     -m $model \
	     -i $data_dir/$test \
       -ip $data_dir/$test_p \
	     -o $working_dir/$2.$3.$4.$5/$test.output.dev \
	     -k ${3:-4} \
	     -n 0.6 \
	     -b 50 \
       -cp ${4:-0} \
       -cw ${5:-probabilities} \


awk -F' _eos ' '{print $4}' $working_dir/$2.$3.$4.$5/$test.output.dev > $working_dir/$2.$3.$4.$5/$test.output.single.dev

# postprocess
$script_dir/postprocess_tokenized.sh < $working_dir/$2.$3.$4.$5/$test.output.single.dev > $working_dir/$2.$3.$4/$test.output.tokenized.dev

# evaluate with tokenized BLEU
echo "tokenized BLEU"
$nematus_home/data/multi-bleu.perl $data_dir/$ref < $working_dir/$2.$3.$4.$5/$test.output.tokenized.dev



# evaluate consistency test accuracy
echo "evaluate consistency test"
script_dir=`dirname $0`
main_dir= # insert the director of good-translation-wrong-in-context respository
data_dir=$main_dir/consistency_testsets/scoring_data
working_dir=$1

devices=0
mkdir $working_dir/$2

for i in deixis_test lex_cohesion_test \
ellipsis_vp \
ellipsis_infl ;do
     defix=lc.bpe.gp
     test_s=$i.$defix.$src
     test_m=$i.$defix.$mt
     test_t=$i.$defix.$tgt
     model=$working_dir/${2:-model.best-valid-script}

     # decode
     CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/score.py \
          -m $model \
          -s $data_dir/$test_s \
          -mt $data_dir/$test_m \
          -t $data_dir/$test_t \
          -o $working_dir/$2/$test_s.score \
          -n 0.6 \
          -b 500 \

     python $script_dir/evaluate_consistency.py \
--repo-dir $main_dir --test $i --scores $working_dir/$2/$test_s.score;

done

