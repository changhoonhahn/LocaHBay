# script for running adcg on file 

# BTLS 
dirtub="/Users/ChangHoon/data/locahbay/smlm/bundled_tubes_long_seq/"
fimage=$dirtub"sequence/11929.tif"
output=$dirtub"adcg/11929.dat"

python adcg.py $fimage "BTLS" 5 $output 
