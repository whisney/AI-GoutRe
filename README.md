# AI-GoutRe
This is the repository for the paper "Development and validation of an artificial intelligence model to predict gout recurrence in hospitalized patients: a real-world, retrospective, and prospective multicentre cohort study in China".

![GoutRe](/pic.png)

## Requirements
* python 3.6+
* pytorch 1.5+
* pandas
* scikit-learn
* scikit-feature
* xgboost
* lightgbm
* catboost

## Model training
For training classification models, you can run:
```
python train_models.py
```

## Model evaluation
To evaluate the models, you can run:
```
python compute_metrics.py
```
