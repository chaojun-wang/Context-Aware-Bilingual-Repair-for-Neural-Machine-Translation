#!/bin/sh
# this script evaluates contrastive test set accuracy

scores=$1
testset_name=$2

script_dir=`dirname $0`
main_dir= # insert the director of good-translation-wrong-in-context respository


# evaluate accuracy and write to standard output (used by training scripts)
python $script_dir/evaluate_consistency.py \
--repo-dir $main_dir --test $testset_name --scores $scores


