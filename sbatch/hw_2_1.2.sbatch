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

for n_threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$n_threads
    ./build/hw_2_1.2 < ./input/hw_2_1.2.txt
done
