# 💳 Credit Card Fraud Detection — End-to-End MLOps

> An end-to-end Machine Learning and MLOps project for detecting fraudulent credit card transactions, with reproducible data pipelines, experiment tracking, model versioning, API deployment, and cloud-based data versioning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple?logo=dvc)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![AWS S3](https://img.shields.io/badge/AWS-S3-orange?logo=amazon-aws)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black?logo=github-actions)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project demonstrates how to build and manage a **production-oriented machine learning workflow** rather than simply training a model inside a notebook.

The system covers the complete ML lifecycle:

```text
Raw Data
   │
   ▼
Data Ingestion
   │
   ▼
Data Transformation
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ├──────────────► MLflow Experiment Tracking
   │
   └──────────────► DVC Pipeline & Versioning
                         │
                         ▼
                       AWS S3
                         │
                         ▼
                  Versioned Artifacts
                         │
                         ▼
                    Flask REST API
```

The goal is to create a **reproducible, maintainable, and deployable fraud detection pipeline** using modern MLOps practices.

---

# 🎯 Project Objectives

* Build an end-to-end fraud detection ML pipeline
* Separate data ingestion, transformation, training, and evaluation
* Version datasets and ML artifacts using DVC
* Store DVC remote data in AWS S3
* Track experiments and model metrics with MLflow
* Expose the trained model through a Flask REST API
* Make the pipeline reproducible
* Prepare the project for CI/CD automation
* Maintain a clean and modular ML project structure

---

# 🧠 Machine Learning Pipeline

The project uses a structured DVC pipeline consisting of the following stages:

```text
data_ingestion
      │
      ▼
data_transformation
      │
      ▼
model_training
      │
      ▼
model_evaluation
```

The pipeline is defined in:

```text
dvc.yaml
```

Pipeline state and dependency information are stored in:

```text
dvc.lock
```

You can visualize the pipeline using:

```bash
dvc dag
```

---

# 📊 Dataset

The project focuses on **credit card transaction fraud detection**.

The dataset contains transaction-related information used to identify whether a transaction is legitimate or fraudulent.

The data pipeline handles tasks such as:

* Data ingestion
* Data cleaning
* Feature engineering
* Data transformation
* Model preparation
* Training/evaluation

Large datasets are intentionally kept outside Git and managed using **DVC**.

---

# 🔄 Data Version Control with DVC

DVC is used to version and reproduce the project's data pipeline.

Instead of committing large datasets directly to Git, DVC tracks the data and stores the actual data remotely.

### DVC Remote

The project uses an AWS S3 remote:

```text
s3://credit-demo123
```

The remote is configured as the default DVC storage.

Push tracked data:

```bash
dvc push
```

Pull tracked data:

```bash
dvc pull
```

Reproduce the pipeline:

```bash
dvc repro
```

Visualize the pipeline:

```bash
dvc dag
```

This allows the project to maintain **data versioning alongside Git code versioning**.

---

# ☁️ AWS S3

AWS S3 is used as the remote storage backend for DVC.

```text
Local Machine
      │
      ▼
     DVC
      │
      ▼
   dvc-s3
      │
      ▼
   AWS S3
      │
      ▼
credit-demo123
```

The project successfully pushes DVC-tracked data to the configured S3 remote.

> 🔐 AWS credentials should never be committed to Git. Configure them through AWS CLI, environment variables, or another secure credential-management mechanism.

---

# 🧪 MLflow Experiment Tracking

MLflow is used for experiment tracking and ML lifecycle management.

The project can track information such as:

* Model parameters
* Evaluation metrics
* Experiments
* Model runs
* Training results
* Model artifacts

The local MLflow database is used during development.

MLflow helps compare different model experiments and identify the best-performing configuration.

---

# 🚀 Flask REST API

The trained model is exposed through a Flask application.

The application is located in:

```text
flask_app/
```

### Available endpoints

#### Health Check

```http
GET /health
```

Used to verify that the API is running.

#### Fraud Prediction

```http
POST /predict
```

Used to send transaction information to the trained model and receive a prediction.

Example API structure:

```text
Client
   │
   │ POST /predict
   ▼
Flask API
   │
   ▼
Preprocessing
   │
   ▼
Trained Model
   │
   ▼
Fraud Prediction
```

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection-_-MLOps/
│
├── .dvc/                       # DVC configuration
├── .github/
│   └── workflows/              # GitHub Actions workflows
│
├── artifacts/                  # Generated pipeline artifacts
├── data/                       # Dataset files / DVC-tracked data
├── docs/                       # Project documentation
├── flask_app/                  # Flask REST API
├── local_s3/                   # Local S3-style storage used during development
├── logs/                       # Application and pipeline logs
├── models/                     # Trained ML models
├── notebooks/                  # Exploratory analysis and experiments
├── references/                 # Reference materials
├── reports/                    # Evaluation reports and outputs
│
├── src/
│   └── components/             # Core ML pipeline components
│       ├── model_trainer.py
│       ├── preprocessing.py
│       ├── train_model.py
│       └── visualize.py
│
├── src.egg-info/               # Python package metadata
│
├── .dvcignore                  # DVC ignore rules
├── .gitignore                  # Git ignore rules
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # Locked pipeline dependencies
├── params.yaml                 # ML pipeline parameters
├── requirements.txt            # Python dependencies
├── setup.py                    # Package configuration
├── Makefile                    # Project automation commands
├── test_environment.py         # Environment validation
├── tox.ini                     # Testing configuration
├── LICENSE                     # Project license
└── README.md                   # Project documentation
```

---

# 🛠️ Tech Stack

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| **Python**         | Core programming language   |
| **Pandas**         | Data processing             |
| **NumPy**          | Numerical computation       |
| **Scikit-learn**   | Machine learning            |
| **DVC**            | Data & pipeline versioning  |
| **AWS S3**         | Remote DVC storage          |
| **MLflow**         | Experiment tracking         |
| **Flask**          | REST API                    |
| **Git**            | Source code version control |
| **GitHub**         | Repository hosting          |
| **GitHub Actions** | CI/CD automation            |
| **PyYAML**         | Configuration management    |
| **Pytest/Tox**     | Testing                     |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection-_-MLOps.git

cd Credit-Card-Fraud-Detection-_-MLOps
```

## 2. Create a virtual environment

Using Conda:

```bash
conda create -n atlas python=3.10
conda activate atlas
```

Or using Python virtual environments:

```bash
python -m venv .venv
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure DVC S3 support is installed:

```bash
pip install dvc-s3
```

---

# 🔐 AWS Configuration

Configure AWS credentials using the AWS CLI:

```bash
aws configure
```

Then test:

```bash
aws sts get-caller-identity
```

Verify access to the configured S3 bucket:

```bash
aws s3 ls s3://credit-demo123
```

> Never commit AWS credentials, `.env` files containing secrets, or access keys to GitHub.

---

# 📦 DVC Setup

Check configured remotes:

```bash
dvc remote list
```

Pull the dataset/artifacts:

```bash
dvc pull
```

Run the complete pipeline:

```bash
dvc repro
```

Push updated DVC data:

```bash
dvc push
```

---

# 🧪 Running MLflow

Start the MLflow server according to your local configuration.

For example:

```bash
mlflow ui
```

Then open the MLflow interface locally to inspect experiments and runs.

---

# 🌐 Running the Flask API

Start the Flask application according to the project's Flask configuration.

For example:

```bash
python -m flask --app flask_app.app run
```

The API can then be accessed locally.

Health check:

```http
GET /health
```

Prediction endpoint:

```http
POST /predict
```

---

# 🔍 Code Validation

Compile the Python source files:

```bash
python -m compileall src flask_app
```

Run environment tests:

```bash
python test_environment.py
```

---

# 🔁 Reproducibility

One of the main goals of this project is reproducibility.

The combination of:

```text
Git
 +
DVC
 +
dvc.lock
 +
params.yaml
 +
MLflow
 +
AWS S3
```

allows the ML workflow to be reproduced and tracked across different development environments.

A simplified workflow is:

```text
Developer changes code
        │
        ▼
      Git
        │
        ▼
    DVC Pipeline
        │
        ▼
   Train Model
        │
        ▼
     MLflow
        │
        ▼
 Evaluate Model
        │
        ▼
    Version Data
        │
        ▼
      AWS S3
```

---

# 🚧 Future Improvements

Planned improvements include:

* [ ] Complete GitHub Actions CI/CD pipeline
* [ ] Automated model testing
* [ ] Automated model validation
* [ ] Docker containerization
* [ ] Model registry integration
* [ ] Production MLflow tracking server
* [ ] Automated model deployment
* [ ] Prometheus monitoring
* [ ] Grafana dashboards
* [ ] Data drift detection
* [ ] Model performance monitoring
* [ ] Automated retraining
* [ ] Self-healing MLOps workflows

---

# 📈 MLOps Architecture

```text
                    ┌───────────────────┐
                    │     GitHub        │
                    │  Source Control   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ GitHub Actions    │
                    │     CI / CD       │
                    └─────────┬─────────┘
                              │
                              ▼
┌─────────────┐       ┌───────────────────┐
│    Data     │──────►│    DVC Pipeline   │
└─────────────┘       └─────────┬─────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
       Data Ingestion    Transformation      Model Training
                                                    │
                                                    ▼
                                            Model Evaluation
                                                    │
                          ┌─────────────────────────┤
                          ▼                         ▼
                    ┌───────────┐             ┌───────────┐
                    │  MLflow   │             │ AWS S3    │
                    │ Tracking  │             │ DVC Data  │
                    └───────────┘             └───────────┘
                                                    │
                                                    ▼
                                            ┌─────────────┐
                                            │ Flask API   │
                                            └──────┬──────┘
                                                   │
                                                   ▼
                                             Predictions
```

---

# 💡 Key Learning Outcomes

This project demonstrates practical experience with:

* End-to-end ML pipelines
* Modular ML project architecture
* Data version control
* ML experiment tracking
* Cloud-based artifact/data storage
* REST API deployment
* Reproducible ML workflows
* Git-based development
* CI/CD preparation
* Production-oriented MLOps practices

---

# 👨‍💻 Author

**Pujan Pandey**

Machine Learning / AI & MLOps Enthusiast

GitHub: `https://github.com/<your-username>`

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

⭐ If you find this project useful, consider giving the repository a star!
