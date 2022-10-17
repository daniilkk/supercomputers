#!/bin/bash
#SBATCH --job-name=hw_1            # Название задачи
#SBATCH --error=output/hw_1-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/hw_1-%j.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1          # Требуемое кол-во CPU
#SBATCH --constraint=type_d

#source /home/dkrasilnikov/supercomputers/venv/bin/activate
module load Python
for i in $(seq 100 100 1000); do
	python3 ./python/hw_1.py < ./input/${i}_5_random.txt
done
