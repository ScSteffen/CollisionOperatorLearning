## vanila DeepONet wo/ bias
python3 trainer_tb.py wo_bias --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias no --epochs 10000 --lambda 0.1

## vanila DeepONet w/ bias
python3 trainer_tb.py w_bias --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias vanila --epochs 10000 --lambda 0.1

## (soft constraint) DeepONet with additional orthogonal loss
python3 trainer_tb.py soft_lamb --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias vanila --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 trainer_tb.py hard_gram --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias no --use_gram --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with special bias
python3 trainer_tb.py hard_special --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias depend --epochs 10000 --lambda 0.1
