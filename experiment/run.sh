#! /bin/bash 

source ~/.bashrc 

conda activate katfish

date 

for text_type in "essay" "poetry" "abstract"
do 
    for seed in 42 43 44 45 46
    do
        python main.py --text_type $text_type --seed $seed
    done 
done

date 

rm -rf __pycache__