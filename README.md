# Credit Card Fraud Detection — MLOps

An end-to-end machine learning project for credit card fraud detection, built with an MLOps workflow for data versioning, reproducible pipelines, experiment tracking, model evaluation, and API serving.

## Overview

This project focuses on building a reproducible machine learning workflow rather than only training a model in a notebook.

The pipeline covers:

* Data ingestion
* Data preprocessing and transformation
* Feature engineering
* Model training
* Model evaluation
* Experiment tracking with MLflow
* Data and pipeline versioning with DVC
* Remote storage using AWS S3
* Model serving through a Flask API
* Project testing and environment validation
* CI/CD setup using GitHub Actions

The main DVC pipeline is:

```text
Data Ingestion
      |
      v
Data Transformation
      |
      v
Model Training
      |
      v
Model Evaluation
```

The pipeline is defined in `dvc.yaml` and locked using `dvc.lock`.

## Project Structure

```text
Credit-Card-Fraud-Detection-_-MLOps/
|
├── .dvc/                       # DVC configuration
├── .github/
│   └── workflows/              # GitHub Actions workflows
|
├── artifacts/                  # Generated artifacts
├── data/                       # Dataset and DVC-tracked data
├── docs/                       # Documentation
├── flask_app/                  # Flask application
├── local_s3/                   # Local storage used during development
├── logs/                       # Application and pipeline logs
├── models/                     # Trained models
├── notebooks/                  # Exploratory notebooks
├── references/                 # Reference files
├── reports/                    # Reports and evaluation results
|
├── src/
│   └── components/
│       ├── model_trainer.py    # Model training logic
│       ├── preprocessing.py    # Data preprocessing
│       ├── train_model.py      # Training entry point
│       └── visualize.py        # Visualization utilities
|
├── .dvcignore
├── .gitignore
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # Locked pipeline state
├── params.yaml                # Pipeline/model parameters
├── requirements.txt           # Python dependencies
├── setup.py                   # Package configuration
├── Makefile                   # Project commands
├── test_environment.py        # Environment validation
├── tox.ini                    # Testing configuration
├── LICENSE
└── README.md
```

## Tech Stack

* Python 3.10
* Pandas
* NumPy
* Scikit-learn
* DVC
* dvc-s3
* AWS S3
* MLflow
* Flask
* Git
* GitHub Actions
* Tox

## DVC

DVC is used to track datasets and manage the machine learning pipeline without storing large data files directly in Git.

The project uses an AWS S3 bucket as the DVC remote.

```text
DVC remote: s3://credit-demo123
```

Useful commands:

```bash
# Check configured remotes
dvc remote list

# Pull data from the remote
dvc pull

# Reproduce the pipeline
dvc repro

# Push tracked data to the remote
dvc push

# Display the pipeline
dvc dag
```

The current pipeline contains:

```text
data_ingestion
       |
       v
data_transformation
       |
       v
model_training
       |
       v
model_evaluation
```

## MLflow

MLflow is used for tracking machine learning experiments and storing information about model runs.

It can be used to track:

* Parameters
* Metrics
* Model results
* Experiments
* Artifacts

During local development, the project uses an MLflow database and local tracking setup.

## Flask API

The trained model is exposed through a Flask application located in:

```text
flask_app/
```

The application currently provides:

### Health check

```http
GET /health
```

Used to check whether the API is running.

### Prediction

```http
POST /predict
```

Used to send transaction data to the model and receive a fraud prediction.

The Flask application can be tested by importing the application:

```bash
python -c "from flask_app.app import app; print(app.url_map)"
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection-_-MLOps.git
cd Credit-Card-Fraud-Detection-_-MLOps
```

### 2. Create the environment

Using Conda:

```bash
conda create -n atlas python=3.10
conda activate atlas
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install DVC S3 support if required:

```bash
pip install dvc-s3
```

## AWS Configuration

The project uses AWS S3 as the DVC remote.

Configure AWS credentials using the AWS CLI:

```bash
aws configure
```

Verify the AWS configuration:

```bash
aws sts get-caller-identity
```

Check access to the DVC bucket:

```bash
aws s3 ls s3://credit-demo123
```

AWS credentials should never be committed to the repository.

## Running the Pipeline

After configuring DVC and AWS:

```bash
dvc pull
```

To reproduce the complete pipeline:

```bash
dvc repro
```

To push updated DVC data to S3:

```bash
dvc push
```

## Running the Flask Application

Start the application using Flask:

```bash
python -m flask --app flask_app.app run
```

The API will be available locally.

Health endpoint:

```text
/health
```

Prediction endpoint:

```text
/predict
```

## Testing

Python source files can be checked for syntax errors with:

```bash
python -m compileall src flask_app
```

The project also includes:

```text
test_environment.py
tox.ini
```

for environment validation and testing configuration.

## CI/CD

GitHub Actions is being used as part of the project's CI/CD setup.

Workflow files are stored in:

```text
.github/workflows/
```

The CI/CD pipeline can be extended to automatically:

1. Install project dependencies
2. Run tests
3. Validate the ML pipeline
4. Build the application
5. Deploy the model/API

## Current MLOps Workflow

```text
                 GitHub
                    |
                    v
             Source Control
                    |
                    v
              DVC Pipeline
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Ingest   Transform   Train
                              |
                              v
                         Evaluation
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
               MLflow                  AWS S3
             Experiments             DVC Storage
                  |                       |
                  +-----------+-----------+
                              |
                              v
                         Flask API
                              |
                              v
                         Prediction
```

## Future Improvements

* Complete CI/CD automation
* Dockerize the application
* Add automated model validation
* Add model registry
* Add Prometheus monitoring
* Add Grafana dashboards
* Add data drift detection
* Add model performance monitoring
* Add automated retraining
* Add automated recovery/self-healing workflows
* Deploy the API to a cloud environment

## Author

Pujan Pandey

Machine Learning and MLOps Developer

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
# Credit Card Fraud Detection — MLOps

An end-to-end machine learning project for credit card fraud detection, built with an MLOps workflow for data versioning, reproducible pipelines, experiment tracking, model evaluation, and API serving.

## Overview

This project focuses on building a reproducible machine learning workflow rather than only training a model in a notebook.

The pipeline covers:

* Data ingestion
* Data preprocessing and transformation
* Feature engineering
* Model training
* Model evaluation
* Experiment tracking with MLflow
* Data and pipeline versioning with DVC
* Remote storage using AWS S3
* Model serving through a Flask API
* Project testing and environment validation
* CI/CD setup using GitHub Actions

The main DVC pipeline is:

```text
Data Ingestion
      |
      v
Data Transformation
      |
      v
Model Training
      |
      v
Model Evaluation
```

The pipeline is defined in `dvc.yaml` and locked using `dvc.lock`.

## Project Structure

```text
Credit-Card-Fraud-Detection-_-MLOps/
|
├── .dvc/                       # DVC configuration
├── .github/
│   └── workflows/              # GitHub Actions workflows
|
├── artifacts/                  # Generated artifacts
├── data/                       # Dataset and DVC-tracked data
├── docs/                       # Documentation
├── flask_app/                  # Flask application
├── local_s3/                   # Local storage used during development
├── logs/                       # Application and pipeline logs
├── models/                     # Trained models
├── notebooks/                  # Exploratory notebooks
├── references/                 # Reference files
├── reports/                    # Reports and evaluation results
|
├── src/
│   └── components/
│       ├── model_trainer.py    # Model training logic
│       ├── preprocessing.py    # Data preprocessing
│       ├── train_model.py      # Training entry point
│       └── visualize.py        # Visualization utilities
|
├── .dvcignore
├── .gitignore
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # Locked pipeline state
├── params.yaml                # Pipeline/model parameters
├── requirements.txt           # Python dependencies
├── setup.py                   # Package configuration
├── Makefile                   # Project commands
├── test_environment.py        # Environment validation
├── tox.ini                    # Testing configuration
├── LICENSE
└── README.md
```

## Tech Stack

* Python 3.10
* Pandas
* NumPy
* Scikit-learn
* DVC
* dvc-s3
* AWS S3
* MLflow
* Flask
* Git
* GitHub Actions
* Tox

## DVC

DVC is used to track datasets and manage the machine learning pipeline without storing large data files directly in Git.

The project uses an AWS S3 bucket as the DVC remote.

```text
DVC remote: s3://credit-demo123
```

Useful commands:

```bash
# Check configured remotes
dvc remote list

# Pull data from the remote
dvc pull

# Reproduce the pipeline
dvc repro

# Push tracked data to the remote
dvc push

# Display the pipeline
dvc dag
```

The current pipeline contains:

```text
data_ingestion
       |
       v
data_transformation
       |
       v
model_training
       |
       v
model_evaluation
```

## MLflow

MLflow is used for tracking machine learning experiments and storing information about model runs.

It can be used to track:

* Parameters
* Metrics
* Model results
* Experiments
* Artifacts

During local development, the project uses an MLflow database and local tracking setup.

## Flask API

The trained model is exposed through a Flask application located in:

```text
flask_app/
```

The application currently provides:

### Health check

```http
GET /health
```

Used to check whether the API is running.

### Prediction

```http
POST /predict
```

Used to send transaction data to the model and receive a fraud prediction.

The Flask application can be tested by importing the application:

```bash
python -c "from flask_app.app import app; print(app.url_map)"
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection-_-MLOps.git
cd Credit-Card-Fraud-Detection-_-MLOps
```

### 2. Create the environment

Using Conda:

```bash
conda create -n atlas python=3.10
conda activate atlas
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install DVC S3 support if required:

```bash
pip install dvc-s3
```

## AWS Configuration

The project uses AWS S3 as the DVC remote.

Configure AWS credentials using the AWS CLI:

```bash
aws configure
```

Verify the AWS configuration:

```bash
aws sts get-caller-identity
```

Check access to the DVC bucket:

```bash
aws s3 ls s3://credit-demo123
```

AWS credentials should never be committed to the repository.

## Running the Pipeline

After configuring DVC and AWS:

```bash
dvc pull
```

To reproduce the complete pipeline:

```bash
dvc repro
```

To push updated DVC data to S3:

```bash
dvc push
```

## Running the Flask Application

Start the application using Flask:

```bash
python -m flask --app flask_app.app run
```

The API will be available locally.

Health endpoint:

```text
/health
```

Prediction endpoint:

```text
/predict
```

## Testing

Python source files can be checked for syntax errors with:

```bash
python -m compileall src flask_app
```

The project also includes:

```text
test_environment.py
tox.ini
```

for environment validation and testing configuration.

## CI/CD

GitHub Actions is being used as part of the project's CI/CD setup.

Workflow files are stored in:

```text
.github/workflows/
```

The CI/CD pipeline can be extended to automatically:

1. Install project dependencies
2. Run tests
3. Validate the ML pipeline
4. Build the application
5. Deploy the model/API

## Current MLOps Workflow

```text
                 GitHub
                    |
                    v
             Source Control
                    |
                    v
              DVC Pipeline
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Ingest   Transform   Train
                              |
                              v
                         Evaluation
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
               MLflow                  AWS S3
             Experiments             DVC Storage
                  |                       |
                  +-----------+-----------+
                              |
                              v
                         Flask API
                              |
                              v
                         Prediction
```

## Future Improvements

* Complete CI/CD automation
* Dockerize the application
* Add automated model validation
* Add model registry
* Add Prometheus monitoring
* Add Grafana dashboards
* Add data drift detection
* Add model performance monitoring
* Add automated retraining
* Add automated recovery/self-healing workflows
* Deploy the API to a cloud environment

## Author

Pujan Pandey

Machine Learning and MLOps Developer

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
