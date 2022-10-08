PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/tester.py --bart_name plm/bart-large --model_path checkpoint/bart-large/batch_size_256_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1_sample_False/best_SequenceGeneratorModel_global_f1_2022-06-05-23-01-48-547338 --num_beams 4  --add_structure 1 --max_length 50 --max_len_a 0


