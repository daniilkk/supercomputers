#!/bin/bash
#SBATCH --job-name=hw_2_1.2           # Название задачи
#SBATCH --error=output/hw_2_1.2-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/hw_2_1.2-%j.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=16          # Требуемое кол-во CPU
#SBATCH --constraint="type_a|type_b|type_c|type_d"
#SBATCH --time=2

export OMP_NUM_THREADS=1
./build/hw_2_1.2 < ./input/hw_2_1.2.txt

export OMP_NUM_THREADS=2 
./build/hw_2_1.2 < ./input/hw_2_1.2.txt

export OMP_NUM_THREADS=4
./build/hw_2_1.2 < ./input/hw_2_1.2.txt

export OMP_NUM_THREADS=8
./build/hw_2_1.2 < ./input/hw_2_1.2.txt

export OMP_NUM_THREADS=16
./build/hw_2_1.2 < ./input/hw_2_1.2.txt
