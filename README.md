# mlopsproject_nlpwithdisastertweets
Mlops_2025_finalproject_nlpwithdisastertweets prediction
# Environment preparation
GitHub Codespaces

Step 1: Download and install the Anaconda distribution of Python

~wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh~

Step 2: Update existing packages
sudo apt update

Step 3: Install Docker and Docker Compose - (Using Docker inside Codespaces)

Step 4: Run Docker
docker run hello-world

disaster_tweet_classification/
│
├── data/
│   ├── raw/                         # Original dataset (downloaded CSVs)
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/                   # Cleaned & preprocessed data (optional)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA notebook
│   └── 02_preprocessing.ipynb       # Text cleaning, tokenizing, vectorizing
│
├── scripts/
│   ├── preprocessing.py             # Modularized version of preprocessing logic
│   ├── train_baseline.py            # Train model + save vectorizer/model
│   └── predict.py                   # Inference using saved model/vectorizer
│
├── models/
│   ├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│   └── baseline_model.pkl           # Saved trained model
│
├── mlruns/                          # MLflow local experiment tracking
│
├── requirements.txt                 # Project dependencies
├── Makefile                         # Commands for setup, linting, training etc.
├── .gitignore                       # Ignore models/, mlruns/, etc.
├── README.md                        # Project overview
└── setup.py (optional)              # If you package your code


Next Steps After EDA
Create feature engineering module (feature_engineering.py)

Build baseline model (Logistic Regression with TF-IDF)

Track it with MLflow

Compare with BERT model

Containerize and deploy with FastAPI

Monitor with Evidently AI

Deploy infra with Terraform on Azure

 **********
Run the script with PYTHONPATH=.
From the root of your project (where the scripts/ folder is located), run:

PYTHONPATH=. python scripts/train_baseline.py
***********