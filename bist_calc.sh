#!/bin/bash

sim_time=10000
h=0.02
Nneur=100
rate=185.0
w_n=1.3
w_p1=1.95
w_p2=2.10
w_p_nbund=15
seed_nbund=1

seed=0
Ie=5.27

path='./'
bin_path='./'
par_fname=$bin_path/nn_params_$Nneur.csv
let "Ntotal = Nneur*w_p_nbund*seed_nbund"

fname=$path/N_${Nneur}_rate_${rate}_w_n_${w_n}_Ie_${Ie}_h_${h}
mkdir $fname
for j in `seq $seed $[$seed_nbund + $seed - 1]`; do
	mkdir $fname/seed\_$j
done

#let "ivp_n = sim_time / 250 - 1"
#for j in `seq 0 $ivp_n`; do
#  mkdir $fname/$j
#done

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 0-00:10
#SBATCH --error=slurm-%J-err.txt
#SBATCH --output=slurm-%J.txt

r_str="$bin_path/hh-cuda $sim_time $h $Ntotal $w_p_nbund $Nneur $fname $seed $rate $w_p1 $w_p2 $w_n $par_fname $Ie 0.01 0"
echo $r_str > $fname/params
$r_str >> $fname/params
