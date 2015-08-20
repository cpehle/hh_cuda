#!/bin/bash

sim_time=10000
h=0.1
Nneur=100
rate=$1
w_n=1.3
w_p1=2.00
w_p2=2.15
w_p_nbund=15
seed_nbund=10

seed=0
Ie=5.26

path='/home/esir_p/1'
bin_path='/home/esir_p/1'
par_fname="$bin_path/nn_params_$Nneur.csv"
let "Ntotal = Nneur*w_p_nbund*seed_nbund"

Ie_arr=`seq 5.26 0.01 5.28`
for Ie_ in $Ie_arr; do
	fname=$path/N\_$Nneur\_rate_$rate\_w_n_$w_n\_Ie\_$Ie_
	mkdir $fname
	for j in `seq $seed $[$seed_nbund + $seed - 1]`; do
		mkdir $fname/seed\_$j
	done
done

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 0-00:10          
#SBATCH --error=slurm-%J-err.txt
#SBATCH --output=slurm-%J.txt

r_str="$bin_path/hh-cuda $sim_time $h $Ntotal $w_p_nbund $Nneur $path $seed $rate $w_p1 $w_p2 $w_n $par_fname $Ie 0.01 0"
echo $r_str
srun $r_str 
