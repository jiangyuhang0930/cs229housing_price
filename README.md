# cs229housing_price

Github Repository for CS229 Fall 2022 Project - Housing Price Prediction

## Datasets
- Datasets extracted from Kaggle, specified in `data_links.txt`. After extracting the data, place it in the same folder as rest of the code. Rename the Ames file to `ames.csv` and rename the Melbourne file to `melb_data.csv`

## Main Files:
- `data_generator.py`: takes in `.csv` files, cleans it, and generates dataframe
- `my_models.py`: list of models used for this study
- `train_predict.py`: contains functions for training and testing a model

## Run Main Experiment:
- Run command `python3 main.py` for the main experiments excluding transfer learning

## Run Transfer Learning:
- Run all cells in the notebook `process_data.ipynb`, then run `python3 transfer_learning.py`
