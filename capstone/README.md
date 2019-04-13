# Capstone project

## About
Capstone project for Udacity's machine-learning engineer nanodegree.
A thorough description of the project is available through the 
[capstone report](https://github.com/slangenbach/udacity-ml-nanodegree/blob/master/capstone/capstone_proposal.pdf).
All code necessary to reproduce the results is available trough 
[Jupyter Notebooks](https://github.com/slangenbach/udacity-ml-nanodegree/blob/master/capstone/capstone.ipynb).

## Prerequisites
* [Anaconda Python](https://www.anaconda.com/distribution/) 2018.12+ with Python 3
* [Kaggle](https://www.kaggle.com) account

## Setup
The following commands assume you work in an Unix-like 
(Linux, macOS, [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)) environment  

Clone this repository to your local workstation
```
git clone https://github.com/slangenbach/udacity-ml-nanodegree.git
cd udacity-ml-nanodegree/capstone
```
Create a dedicated Python 3 environment using conda
```
conda env create -f conda.yml
```
Activate your newly created environment
```
conda activate <name of your environment>
```
Create directories required for the project
```
mkdir data image model
```
Download data from Kaggle (Make sure you have 
[provided your API-Credentials to Kaggle's CLI](https://github.com/Kaggle/kaggle-api#api-credentials))
```
kaggle competitions download santander-customer-transaction-prediction -p ./data/
```
[Unzip](https://linux.die.net/man/1/unzip) all downloaded files
```
cd ./data/
unzip '*.zip'
```
Run Jupyter Lab (or Notebook) to reproduce results
```
cd ..
jupyter lab
```