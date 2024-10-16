from tkinter.messagebox import YES
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
import flask
import boto3

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import plotly
import plotly.express as px
import json # for graph plotting in website

import sys
import time
import requests
from st_aggrid import AgGrid
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from numpy import radians, cos, sin




def Get_Training_Data(s3_bucket,file_name):
    import boto3
    import numpy as np 
    import pandas as pd
#from package import config
    
    instance_type = 'ml.m5.large'

    session = boto3.Session()
    print(session)

    s3 = session.resource('s3', region_name='us-east-1')
    print(s3)
  
    object = s3.Object(s3_bucket,file_name)
    print(s3_bucket)
    print(file_name)
    print(object)
    download=object.download_file(file_name)
    print(download)

    data = pd.read_csv(file_name, delimiter=',')
    return data




with st.sidebar:
    st.header('Input training data and output S3 for Fraud AI Model Training')
    with st.form(key='training_form'):
        s3_bucket=st.text_input(label='Please Enter S3 Bucket Name of Training data')
        file_name=st.text_input(label='Please Enter Training data file name')
        s3_bucket_for_model=st.text_input(label='Please Enter S3 Bucket Name for storing Fraud AI Model')
        submit_button = st.form_submit_button(label='Submit')

if submit_button:

    import numpy as np 
    import pandas as pd

    data= Get_Training_Data(s3_bucket,file_name)
    st.title('Investigate and process the training data')

    nonfrauds, frauds = data.groupby('Class').size()
    st.write('Number of frauds: ', frauds)
    st.write('Number of non-frauds: ', nonfrauds)
    st.write('Percentage of fradulent data:', 100.*frauds/(frauds + nonfrauds))
    
    feature_columns = data.columns[:-1]
    label_column = data.columns[-1]

    features = data[feature_columns].values.astype('float32')
    labels = (data[label_column].values).astype('float32')

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42)

    import os
    import sagemaker
    #from package import config

    session = sagemaker.Session()
    bucket = s3_bucket
    prefix = 'fraud-classifier'

    from sagemaker import RandomCutForest
    st.title('Unsupervised Learning')
    st.image("un-super.jfif")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("ai.png")
  

# specify general training job information
    rcf = RandomCutForest(role='arn:aws:iam::022830946521:role/service-role/AmazonSageMaker-ExecutionRole-20231110T154223',
                      train_instance_count=1,
                      train_instance_type='ml.m5.large',
                      data_location='s3://{}/{}/'.format(bucket, prefix),
                      output_path='s3://{}/{}/output'.format(bucket, prefix),
                      base_job_name="{}-rcf".format('fraud-hackathon'),
                      num_samples_per_tree=512,
                      num_trees=50)
    with st.spinner("Waiting:Starting AI training with unsupervised learning for your data..."):
        train_result=rcf.fit(rcf.record_set(X_train))

                  # Example process with progress bar
    st.subheader('Progress of  AI training with unsupervised learning for your data...')
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 100  # Example total steps for the process

    for step in range(total_steps):
    # Simulate a step in the process
        time.sleep(0.1)  # Adjust time per step to your actual process
        # Calculate progress percentage
        percent_complete = int((step + 1) / total_steps * 100)
        # Update progress bar and status text
        progress_bar.progress(percent_complete)
        status_text.text(f'Progress: {percent_complete}%')

    st.success('Training job for unsupervised learning Fraud AI model  has been completed successfully!')
    st.image("un_super_m.JPG")


                  # Example process with progress bar
    st.subheader('Deploying AI unsupervised learning model endpoint for evalution ...')
    with st.spinner("Waiting:Deploying the trained unsupervised learning Fraud AI model..."):
        instance_type = 'ml.m5.large'
        import uuid
        unique_id = uuid.uuid4()
        base_id='fraud-hackathon'
        model_name=f"{base_id}-{unique_id}-rcf"
        endpoint_name=f"{base_id}-{unique_id}-rcf-ep"
        end_point_results=rcf_predictor = rcf.deploy(
        model_name=model_name,
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type=instance_type)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 100  # Example total steps for the process

    for step in range(total_steps):
    # Simulate a step in the process
        time.sleep(0.1)  # Adjust time per step to your actual process
        # Calculate progress percentage
        percent_complete = int((step + 1) / total_steps * 100)
        # Update progress bar and status text
        progress_bar.progress(percent_complete)
        status_text.text(f'Progress: {percent_complete}%')
    st.success('Deploying the trained unsupervised learning Fraud AI model has been completed successfully' )
    st.image("un_super_ep.JPG")
   

    from sagemaker.predictor import csv_serializer, json_deserializer

    rcf_predictor.content_type = 'text/csv'
    rcf_predictor.serializer = csv_serializer
    rcf_predictor.accept = 'application/json'
    rcf_predictor.deserializer = json_deserializer

    def predict_rcf(current_predictor, data, rows=500):
        split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
        predictions = []
        for array in split_array:
            array_preds = [s['score'] for s in current_predictor.predict(array)['scores']]
            predictions.append(array_preds)

        return np.concatenate([np.array(batch) for batch in predictions])

    positives = X_test[y_test == 1]
    positives_scores = predict_rcf(rcf_predictor, positives)

    negatives = X_test[y_test == 0]
    negatives_scores = predict_rcf(rcf_predictor, negatives)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(color_codes=True)
    st.title('Test Unsupervised Learning(Random Cut Forest)')


    sns.distplot(positives_scores, label='fraud', bins=20)
    sns.distplot(negatives_scores, label='not-fraud', bins=20)
    plt.legend()
    st.pyplot(plt)
    
    st.markdown('''
    
    - With the unsupervised model deployed, let's see how it performs in terms of separating fraudulent from legitimate transactions., 

    - The unsupervised model already can achieve some **separation** between the classes, with **higher anomaly scores being correlated to fraud**.
        
''')
    import io
    import sklearn
    from sklearn.datasets import dump_svmlight_file   

    buf = io.BytesIO()

    sklearn.datasets.dump_svmlight_file(X_train, y_train, buf)
    buf.seek(0);

    import boto3
    import os
    key = 'fraud-dataset'
    subdir = 'base'
    bucket = s3_bucket
    prefix = 'fraud-classifier'
    session = boto3.Session()
    print(session)

    s3 = session.resource('s3', region_name='us-east-1')
    print(s3)

    session.resource('s3', region_name='us-east-1').Bucket(bucket).Object(os.path.join(prefix, 'train', subdir, key)).upload_fileobj(buf)

    s3_train_data = 's3://{}/{}/train/{}/{}'.format(bucket, prefix, subdir, key)
    print('Uploaded training data location: {}'.format(s3_train_data))

    output_location = 's3://{}/{}/output'.format(bucket, prefix)
    print('Training artifacts will be uploaded to: {}'.format(output_location))

   
    import boto3
    from sagemaker.amazon.amazon_estimator import get_image_uri


    container = get_image_uri(boto3.Session().region_name, 'xgboost', repo_version='0.90-2')

    from math import sqrt

# Because the data set is so highly skewed, we set the scale position weight conservatively,
# as sqrt(num_nonfraud/num_fraud).
# Other recommendations for the scale_pos_weight are setting it to (num_nonfraud/num_fraud).
    scale_pos_weight = sqrt(np.count_nonzero(y_train==0)/np.count_nonzero(y_train))
    print(scale_pos_weight)
    hyperparams = {
        "max_depth":5,
        "subsample":0.8,
        "num_round":100,
        "eta":0.2,
        "gamma":4,
        "min_child_weight":6,
        "silent":0,
        "objective":'binary:logistic',
        "eval_metric":'auc',
        "scale_pos_weight": scale_pos_weight
    }



    import uuid
    unique_id = uuid.uuid4()
    base_id='fraud-hackathon'
    base_job_name=f"{base_id}-{unique_id}-xgb"
    instance_type = 'ml.m5.large'
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    boto3_session = boto3.Session()
    print(boto3_session)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)
    print(sagemaker_session)

    clf = sagemaker.estimator.Estimator(container,
                                    role='arn:aws:iam::022830946521:role/service-role/AmazonSageMaker-ExecutionRole-20231110T154223',
                                    hyperparameters=hyperparams,
                                    train_instance_count=1, 
                                    train_instance_type=instance_type,
                                    output_path=output_location,
                                    sagemaker_session=sagemaker_session,
                                    base_job_name=base_job_name)

    st.title('Supervised Learning')
    st.image("supervise.png")
    st.image("super.jfif")
   
    st.markdown('''
    
    - Once we have gathered an adequate amount of **labeled** training data, we can use a **supervised learning** algorithm that discovers relationships between the **features** and the **dependent class**.
    
    - We will use **Gradient Boosted Trees** as our model, as they have a proven track record, are highly **scalable** and can deal with missing data, reducing the need to pre-process datasets.🐿💨
''')
    
    with st.spinner("Waiting:Training the Fraud AI model with Supervised Learning ..."):
        clf.fit({'train': s3_train_data})
    st.subheader('Progress of training the Fraud AI model with Supervised Learning...')
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 100  # Example total steps for the process

    for step in range(total_steps):
    # Simulate a step in the process
        time.sleep(0.1)  # Adjust time per step to your actual process
        # Calculate progress percentage
        percent_complete = int((step + 1) / total_steps * 100)
        # Update progress bar and status text
        progress_bar.progress(percent_complete)
        status_text.text(f'Progress: {percent_complete}%')
    st.success('Training the Fraud AI model with Supervised Learning has been completed successfully' )
    st.image("super_m.JPG")


    from sagemaker.predictor import csv_serializer
    import uuid
    unique_id = uuid.uuid4()
    base_id='fraud-hackathon'
    model_name=f"{base_id}-{unique_id}-xgb"
    endpoint_name=f"{base_id}-{unique_id}-xgb-ep"
    with st.spinner("Waiting:Deploying the Fraud AI model with Supervised Learning ..."):
        predictor = clf.deploy(initial_instance_count=1,
                        model_name=model_name,
                        endpoint_name=endpoint_name,
                        instance_type=instance_type,
                        serializer=csv_serializer,
                        deserializer=None,
                        content_type='text/csv')
    st.subheader('Progress of deploying the Fraud AI model with Supervised Learning...')
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 100  # Example total steps for the process

    for step in range(total_steps):
    # Simulate a step in the process
        time.sleep(0.1)  # Adjust time per step to your actual process
        # Calculate progress percentage
        percent_complete = int((step + 1) / total_steps * 100)
        # Update progress bar and status text
        progress_bar.progress(percent_complete)
        status_text.text(f'Progress: {percent_complete}%')
                        
    st.success('Deploying the Fraud AI model with Supervised Learning has been completed successfully' )
    st.image("super_ep.JPG")

    def predict(current_predictor, data, rows=500):
        split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
        predictions = ''
        for array in split_array:
            predictions = ','.join([predictions, current_predictor.predict(array).decode('utf-8')])

        return np.fromstring(predictions[1:], sep=',')

    raw_preds = predict(predictor, X_test)
    st.title('Evaluation for Supervised Learning')

    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

# scikit-learn expects 0/1 predictions, so we threshold our raw predictions
    y_preds = np.where(raw_preds > 0.5, 1, 0)

    st.markdown('''
    
    - We will use a few measures from the scikit-learn package to evaluate the performance of our model. 
    
    - When dealing with an **imbalanced dataset**, we need to choose metrics that take into account the **frequency** of each class in the data.

    - Two such metrics are the **balanced accuracy score**, and **Cohen's Kappa**.

    - We can already see that our model performs **very well** in terms of both metrics,
    
    - Cohen's Kappa scores above 0.8 are generally very **favorable**.🐿💨
''')
   
    balanced_accuracy_score=balanced_accuracy_score(y_test, y_preds)
    Cohen=cohen_kappa_score(y_test, y_preds)
    balanced_text=f"**Balanced accuracy = {balanced_accuracy_score}**"
    Cohen_text=f"**Cohen's Kappa = {Cohen}**"
    st.markdown(balanced_text)
    st.markdown(Cohen_text)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(y_true, y_predicted):

        cm  = confusion_matrix(y_true, y_predicted)
        # Get the per-class normalized value for each cell
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # We color each cell according to its normalized value, annotate with exact counts.
        ax = sns.heatmap(cm_norm, annot=cm, fmt="d")
        ax.set(xticklabels=["non-fraud", "fraud"], yticklabels=["non-fraud", "fraud"])
        ax.set_ylim([0,2])
        plt.title('Confusion Matrix')
        plt.ylabel('Real Classes')
        plt.xlabel('Predicted Classes')
        #plt.show()
        st.pyplot(plt)
    plot_confusion_matrix(y_test, y_preds)

    st.markdown('''
    
    - Apart from single-value metrics, it's also useful to look at metrics that indicate performance per class. 
    
    - A **confusion matrix**, and **per-class precision**, **recall** and **f1-score** can also provide more information about the model's **performance**. 
''')

    st.title('Classification Report  Metrics for Supervised Learning')
    from sklearn.metrics import classification_report

    report=classification_report(
    y_test, y_preds, target_names=['non-fraud', 'fraud'],output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


    




