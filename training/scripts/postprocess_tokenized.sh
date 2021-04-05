# postprocess translations by removing out BPE segmentation.

script_dir=`dirname $0`
main_dir=$script_dir/../

# variables (toolkits; source and target language)
. $main_dir/vars

sed -r 's/ `//g'
