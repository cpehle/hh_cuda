#!/bin/bash
#for Tstart in `seq 280000 2000 300000`; do
Tstart=280000
let 'sim_time = Tstart + 10000'

sim_time=4000000
h=0.02
Nneur=100
rate=185.0
w_n=1.3
w_p1=1.99
w_p2=2.05
w_p_nbund=1
seed_nbund=100
seed=100
Ie=5.27
let "startFl = (Tstart / 2000) - 1"

path='/media/data/'
bin_path='./'
par_fname=$bin_path/nn_params_$Nneur.csv
let "Ntotal = Nneur*w_p_nbund*seed_nbund"

fname=$path/N_${Nneur}_rate_${rate}_w_n_${w_n}_Ie_${Ie}_

# fname_=$path/N_${Nneur}_rate_${rate}_w_n_${w_n}_Ie_${Ie}
# fname=${fname_}_startFl_${startFl}

mkdir $fname
for j in `seq $seed $[$seed_nbund + $seed - 1]`; do
	mkdir $fname/seed\_$j
done

#let "ivp_n = sim_time / 350 - 1"
##for j in `seq 0 $ivp_n`; do
#for j in `seq 0 300`; do
#  mkdir $fname/$j
#done

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 0-00:10
#SBATCH --error=slurm-%J-err.txt
#SBATCH --output=slurm-%J.txt

# r_str="$bin_path/hh-cuda $sim_time $h $Ntotal $w_p_nbund $Nneur $fname $seed $rate $w_p1 $w_p2 $w_n $par_fname $Ie 0.01 0 ${fname_}/$startFl $Tstart"
r_str="$bin_path/hh-cuda $sim_time $h $Ntotal $w_p_nbund $Nneur $fname $seed $rate $w_p1 $w_p2 $w_n $par_fname $Ie 0.01 0"

echo $r_str > $fname/params
$r_str >> $fname/params
#done