# Text processing and classification using Logistic Regression
This repository contains the code for basic text processing and building a binary text classifier using Logistic Regression

### Installation

**Python Version: 3.8.10**

The Python 3.8 version is supported for the execution of this project.
1. Running the codes locally: Python 3.8
    - Create a virtual environment: In the terminal or command prompt, use the following command to create a new virtual environment:

        - For Windows:
            `py -3.8 -m venv env`

        - For macOS and Linux:
            `python3 -m venv venv`
        This command creates a new virtual environment named "venv" in the current directory.

    - Activate the virtual environment:

        - For Windows (Command Prompt):
            `venv\Scripts\activate`


        - For macOS and Linux (Terminal) :
            `source venv/bin/activate`
    
Once the virtual environment is activated, you should see the environment name (e.g., (venv)) in the terminal or command prompt.

- Install project requirements: With the virtual environment active, you can now install the required packages for your project. Typically, the required packages are listed in a requirements.txt file. Make sure you have the requirements.txt file for your project.

2. Setup the pip and installing dependencies in Virtual environment

    - Upgrade pip
        `python -m pip install --upgrade pip`

    - To install the requirements, use the following command:
        `pip install -r requirements.txt`


### Dataset
The dataset is a custom dataset where reviews about an app are taken from app store and the reviews are classified either as positive or negative

### Train the model
To train the model run:
```buildoutcfg
python Engine.py --file_name Canva_reviews.xlsx --vectorizer bowb --output_name binary_count_vect
```
Here we can use 4 types of vectorizers:
* Bag of Words - `bow`
* Binary Bag of Words - `bowb`
* N-grams - `ng`
* TF-IDF - `tf`

### Predictions
To make prediction on a new review `Its the worst app ever I save my design lts not save`,  run:
```buildoutcfg
python predict.py --text 'Its the worst app ever I save my design lts not save' --model_name binary_count_vect
```
Here `binary_count_vect` is the file name used to save the model and the vectorizer during the training phase

### Note on NLTK Package:
For installing NLTK, use the command `pip install nltk` <br />
After downloading, the NLTK corpus has to be downloaded <br />
Run `import nltk` followed by `nltk.download()` in jupyter notebook <br />
This will open a separate window where you can donwnload the necessary packages <br />
For this project, you will need the following packages:<br />
<ol>
<li>punkt</li>
<li>stopwords</li>
<li>wordnet</li>
</ol>
