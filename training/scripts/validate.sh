#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the general test set
# using tokenised BLEU.

translations=$1

script_dir=`dirname $0`
main_dir=$script_dir/../
data_dir= # insert the directory storing preprocessed data

# language-independent variables, toolkit locations
. $main_dir/vars

dev_prefix=dev.lc.gp.single 
ref=$dev_prefix.$tgt

# evaluate translations and write BLEU score to standard output (for
# use by training scripts)
single_translation=$(mktemp)
awk -F' _eos ' '{print $4}' $translations > $single_translation
$script_dir/postprocess_tokenized.sh < $single_translation | \
    $nematus_home/data/multi-bleu.perl $data_dir/$ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
rm $single_translation

