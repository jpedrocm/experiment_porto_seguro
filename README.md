# Project 2

![python](https://img.shields.io/badge/python-2.7-blue.svg)
![status](https://img.shields.io/badge/status-in%20progress-yellow.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the second project for the Machine Learning class.

## Description

The goal of this project is to explore the [Porto Seguro's Safe Driver Prediction Challenge](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/) released on Kaggle in November of 2017. We aim to simply experiment main monolythic models, find their best hyperparameter configurations and features combination through k-fold cross-validation to get a good final solution.

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 2.7.15
* [NumPy](http://www.numpy.org/) >= 1.15.1
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.19.2


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the training and test datasets from the [Porto Seguro's data page](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)
* Place the CSV's in the data/ folder

### Reproducing

* Enter into the code/ folder in your local repository
* Run the experiment to produce every ensemble's predictions
* Edit the training configuration file in the configs/training/ folder
* Generate training results
```
python generate_training_results.py
```
* Edit the training configuration file in the configs/test/ folder
* Generate test submission files
```
python generate_test_submission.py [-s SUBMISSION_FILENAME]
```


## Project Structure

    .            
    ├── code                               # Code files
    |   ├── generate_training_results.py
    |   └── generate_test_submission.py
    ├── configs                            # Models' configurations
    |   ├── training
    |   └── test
    ├── data                               # Dataset files
    ├── results                            # Training results
    ├── submissions                        # Test submission files
    ├── LICENSE.md
    └── README.md

## Authors

* [jpedrocm](https://github.com/jpedrocm)
* Flávio Filho

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.