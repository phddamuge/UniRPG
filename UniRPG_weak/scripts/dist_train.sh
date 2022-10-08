CUDA_VISIBLE_DEVICES=3,4 PYTHONPATH=$PYTHONPATH:tag_op python3 -m torch.distributed.launch --nproc_per_node=2 tag_op/dist_trainer.py --bart_name plm/bart-base --save_model_path checkpoint/bart-base --n_epochs 30 --batch_size 32 --max_length 50 --max_len_a 0 --num_beams 4 --lr 1e-4 --weight_decay 1e-2 --blr 1e-4 --b_weight_decay 1e-2 --seed 345 --gradient_accumulation 4 --add_structure 0 --port 54343
# PYTHONPATH=$PYTHONPATH:tag_op