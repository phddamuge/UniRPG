# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_dev.pkl --logical_form_path checkpoint/bart-base/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_0_sample_False/predict_grammar_dev.json --mode dev

# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_dev.pkl --logical_form_path checkpoint/bart-large/batch_size_256_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1_sample_False/predict_grammar_dev.json --mode dev

# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_test.pkl --logical_form_path checkpoint/bart-base/batch_size_128_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1_sample_False/predict_grammar_test.json --mode test

# CUDA_VISIBLE_DEVICES=5YTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_dev.pkl --logical_form_path checkpoint/bart-large/batch_size_256_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1_sample_False/predict_grammar_dev.json --mode dev

CUDA_VISIBLE_DEVICES=5YTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_test.pkl --logical_form_path checkpoint/bart-large/ensemble/tat_pred_ensemble_test.json --mode test

# CUDA_VISIBLE_DEVICES=5YTHONPATH=$PYTHONPATH:tag_op python3 tag_op/executor.py --data_path tag_op/cache/tagop_roberta_cached_test.pkl --logical_form_path checkpoint/bart-large/batch_size_256_lr_0.0001_blr_0.0001_wd_0.01_bwd_0.01_beams_4_epoch_30_maxlength_50_add_structure_1_sample_False/predict_grammar_test.json --mode test