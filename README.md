# Text-to-3D codes for finetuning and evaluation
This folder contains codes for Text-to-3D experiments in terms of finetuning and evaluation. 

## Finetuning

```
### Finetune Shap-E

# first git clone https://github.com/openai/shap-e, and install with pip install -e .
# move finetune_shapE.py and corresponding files in example_material to shap-e directory
# need to modify --latent_code_path to the directory you store the shap-E latent code .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/ShapELatentCode_zips)
# you may also change the caption and train/val set file path in line #48~49, #68~69
```
python finetune_shapE.py --gpus 4 --batch_size 16 --save_name 'shapE_bs16_lr1e5' --latent_code_path './Cap3D_latentcodes'
```


# Basic Script
python evaluate.py --fid --eval_size 2000 --gt_dir test_gt_images_2k/ --pred_dir your_prediction/ --test_uid_path ./example_material/test_uids_2k.pkl --caption_path ./example_material/Cap3D_automated_Objaverse.csv
python evaluate.py --clip_score_precision --eval_size 2000 --gt_dir test_gt_images_2k/ --pred_dir your_prediction/ --test_uid_path ./example_material/test_uids_2k.pkl --caption_path ./example_material/Cap3D_automated_Objaverse.csv

# For Point-E, Shap-E set eval_size as 2000 (test_gt_images_2k.zip), for DreamField, DreamFusion, 3DFuse set eval_size as 300 (test_gt_images_300.zip)
# Must select only one of --fid or --clip_score_precision in a given call unless have multiple GPUs available. 
```
