#!/bin/sh
#Set the job name (for your reference)
#PBS -N DL_A1
### Set the project name
#PBS -P col764.aib242289.course
### Request email when job begins and ends
### PBS -m bea
### Specify email address to use for notification.
###P BS -M aib242289@iitd.ac.in
### chunk specific resources ###(select=5:ncpus=4:mpiprocs=4:ngpus=2:mem=2GB::centos=skylake etc.)
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=01:00:00
#PBS -o outfile
#PBS -e errors

###PBS -l software=INTEL_PARALLEL_STUDIO

echo "==============================="
module load apps/pytorch/1.10.0/gpu/intelpython3.7
echo $PBS_JOBID
cd ~/AIL721/Assignment1/Q1/
python shallow_pytorch.py 
python shallow_numpy.py
echo "Q1 Completed"
cd ~/AIL721/Assignment1/Q2.1/
python assi_LR1_quest.py
echo "Q2.1 Completed"
cd ~/AIL721/Assignment1/Q2.2/
python train.py --n_epochs=100
echo "Q2.2 Completed"
cd ~/AIL721/Assignment1/Q3/
python mnist.py
echo "Q3 Completed"
#job execution command
echo "completed"