

# Sagemaker Fraud Detection

This repository contains code and workflows for detecting fraudulent transactions using both supervised and unsupervised machine learning models on AWS SageMaker. The project demonstrates how to use AWS SageMaker to automate the end-to-end process of training and deploying fraud detection models.

## Project Overview

The main goal of this project is to build an automated pipeline that detects fraud in financial transactions. The repository showcases the implementation of supervised and unsupervised machine learning algorithms for identifying fraudulent patterns, with a focus on scalability and performance using AWS SageMaker.

Key features of the project:
- Data preprocessing and feature engineering for fraud detection.
- Training supervised and unsupervised models using AWS SageMaker.
- Deployment of the best-performing models with real-time predictions.
- Monitoring and visualization of model performance using Streamlit.

## Requirements

To run this project, you need the following dependencies installed:

- Python 3.8+
- Boto3 (AWS SDK for Python)
- AWS SageMaker Python SDK
- Scikit-learn
- Streamlit

You can install the required dependencies with the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```
git clone https://github.com/Jencheng1/SagemakerFraud.git
cd SagemakerFraud
```

2. Run the Python script for fraud detection:

```
python sagemaker.streamlit.supervised.unsupervised.fraud.ml.py
```

This script will:
- Preprocess the transaction data.
- Train supervised and unsupervised models using AWS SageMaker.
- Monitor model performance through Streamlit’s interactive dashboard.

## Data

The data for fraud detection should be provided in CSV format and uploaded to an S3 bucket. The script will automatically load the data from S3, preprocess it, and start the machine learning pipeline using SageMaker.

## Credit

This project uses **AWS SageMaker** to build and deploy fraud detection models. AWS SageMaker offers a scalable environment for training machine learning models, making it ideal for detecting anomalies in large-scale transaction datasets.

## License

This project is licensed under the MIT License - see the LICENSE file at the following link for details: https://www.mit.edu/~amini/LICENSE.md.

