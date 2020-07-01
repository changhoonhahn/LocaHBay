# script for running adcg on file 

repodir="/Users/ChangHoon/projects/sparseBayes/"
# BTLS 
#dirtub="/Users/ChangHoon/data/locahbay/smlm/bundled_tubes_long_seq/"
#fimage=$dirtub"sequence/11929.tif"
#output=$dirtub"adcg/11929.dat"

#python adcg.py $fimage "BTLS" 5 $output 

# mock data  
dirtub=$repodir"dat/alpha/hd_lsnr/"
for i in {0..24}; do
    fimage=$dirtub"$i.dat"
    output=$dirtub"adcg.$i.dat"
    python ${repodir}run/adcg.py $fimage "mock_alpha" 5 $output 
done

