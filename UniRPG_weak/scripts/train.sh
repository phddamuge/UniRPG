PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/trainer.py --bart_name plm/bart-base --save_model_path checkpoint/bart-base --n_epochs 30 --batch_size 32 --max_length 50 --max_len_a 0 --num_beams 4 --lr 1e-4 --weight_decay 1e-2 --blr 1e-4 --b_weight_decay 1e-2 --seed 345 --gradient_accumulation 4 --add_structure 1
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/trainer.py --bart_name plm/bart-base --save_model_path checkpoint/bart-base --n_epochs 30 --batch_size 128 --max_length 50 --max_len_a 0 --num_beams 4 --lr 5e-5 --weight_decay 1e-2 --blr 5e-5 --b_weight_decay 1e-2 --seed 345 --gradient_accumulation 1 --add_structure 0
