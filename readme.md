# CNN + MFCC Baseline (Speech)


This repository contains a simple, end-to-end baseline pipeline for a speech classification task using MFCC features and a CNN.


## Structure
- data_loader.py # load audio files and labels
- preprocess.py # extract MFCC and utilities (pad/trim)
- model_baseline_A.py # CNN model (PyTorch)
- train.py # training script
- evaluate.py # evaluation script (accuracy + confusion matrix PDF)
- utils.py # helper functions (save/load)
- requirements.txt


## Dataset layout (expected)
Place your audio files with this structure:
## Quick start
1. Install dependencies:
`pip install -r requirements.txt`
2. Edit `DATA_DIR` path in `train.py` and `evaluate.py` (or pass as arg)
3. Train:
`python train.py --data_dir ./data --epochs 10 --batch_size 32`
4. Evaluate:
`python evaluate.py --data_dir ./data --model_path ./results/best_model.pt`


If you don't have a dataset yet, the scripts include an option `--make_dummy` to generate a tiny synthetic dataset for testing the pipeline.