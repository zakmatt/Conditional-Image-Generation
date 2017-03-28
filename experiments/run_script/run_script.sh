#!/bin/bash
#PBS -l nodes=1:ppn=2 -l mem=6gb
#PBS -q normale
#PBS -l walltime=168:00:00
#PBS -N mzak-ift6266
#PBS -o /home2/ift6ed68/logs.out
#PBS -M zak.matthew@yahoo.com
#PBS -m abe
#PBS -r n
#PBS -j oe
#PBS -V
#--------------------------------------------------------------
module add python/3.5.1
module add CUDA/7.5

source /home2/ift6ed68/pyt3.5/bin/activate
cd /home2/ift6ed68/Conditional-Image-Generation/
THEANO_FLAGS='device=cuda,floatX=float32' python experiments/train.py -t experiments/dataset/training_dataset -v experiments/dataset/validation_dataset -s experiments/