python3 tag_op/trainer.py --bart_name plm/bart-large --save_model_path checkpoint/bart-large --n_epochs 30 --batch_size 8 --max_length 50 --max_len_a 0 --num_beams 4 --lr 1e-4 --weight_decay 1e-2 --blr 1e-4 --b_weight_decay 1e-2 --seed 543 --gradient_accumulation 32 --add_structure 1
