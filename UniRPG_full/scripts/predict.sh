# CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-large --model_path checkpoint/bart-large/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_0/best_SequenceGeneratorModel_em_2022-02-05-19-13-52-706393 --num_beams 4 --add_structure 0 --max_length 50 --max_len_a 0 --batch_size 16

# CUDA_VISIBLE_DEVICES=5 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-base --model_path checkpoint/bart-base/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1/best_SequenceGeneratorModel_global_f1_2022-05-30-22-02-05-694424 --num_beams 4 --add_structure 1 --max_length 50 --max_len_a 0 --batch_size 16

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-large --model_path checkpoint/bart-large/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1/best_SequenceGeneratorModel_global_f1_2022-06-02-12-08-09-208541 --num_beams 4 --add_structure 1 --max_length 50 --max_len_a 0 --batch_size 16

# CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-large --model_path checkpoint/bart-large/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1/best_SequenceGeneratorModel_global_f1_2022-06-22-12-15-19-639172 --num_beams 4 --add_structure 1 --max_length 50 --max_len_a 0 --batch_size 16

# CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-large --model_path checkpoint/bart-large/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1/best_SequenceGeneratorModel_global_f1_2022-06-22-10-11-42-650583 --num_beams 4 --add_structure 1 --max_length 50 --max_len_a 0 --batch_size 16

