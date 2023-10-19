import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
DATASET INFORMATION 
There are two main folders of data:
    1. Claim prediction data
    2. Fraud detection data

Claim prediction data is the final dataset that the discriminator model will utilize to predict if a patient will claim insurance for their condition or not
This dataset will mostly be used for the final stage of testing the model

Fraud detection data is the dataset that the generator model will utilize to generate synthetic data for the discriminator model to train on
The training data has been split into 3 main cateogries, inpatient, outpatient and beneficiary data.
A) Inpatient Data
This data provides insights about the claims filed for those patients who are admitted in the hospitals. It also provides additional details like their admission and discharge dates and admit d diagnosis code.
B) Outpatient Data
This data provides details about the claims filed for those patients who visit hospitals and not admitted in it.
C) Beneficiary Details Data
This data contains beneficiary KYC details like health conditions,regioregion they belong to etc.

The labels for the fraud detection, however, have not been categorized into the above categories. They are provided in the form of a binary variable, 
where 1 indicates fraud and 0 indicates no fraud.
Healthcare fraud and abuse take many forms. Some of the most common types of frauds by providers are:
a) Billing for services that were not provided.
b) Duplicate submission of a claim for the same service.
c) Misrepresenting the service provided.
d) Charging for a more complex or expensive service than was actually provided.
e) Billing for a covered service when the service actually provided was not covered.
** NOTE: The labels for the test set are not provided, rather it can be assumed that all the providers (UID) given in the file are all potential fraudsters.

Idea on preprocessing training data
1) Combine inpatient and outpatient data with label for inpatient = 1, outpatient = 0
2) Add label from training data indicating if claim was potentially fraudulent
3)

"""

# Reading in datasets
train_inpatient = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Train_Inpatientdata-1542865627584.csv')
train_outpatient = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Train_Outpatientdata-1542865627584.csv')
train_beneficiary = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Train_Beneficiarydata-1542865627584.csv')
train_label = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Train-1542865627584.csv')

test_inpatient = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Test_Inpatientdata-1542969243754.csv')
test_outpatient = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Test_Outpatientdata-1542969243754.csv')
test_beneficiary = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Test_Beneficiarydata-1542969243754.csv')
test_label = pd.read_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\data\fraud_detection_data\Test-1542969243754.csv')

# Combine training datasets
train_inpatient['inpatient'] = 1
train_outpatient['inpatient'] = 0
train = pd.concat([train_inpatient, train_outpatient], axis=0, sort=False)
train.merge(train_label, on='Provider', how='left')

# Combine test datasets
test_inpatient['inpatient'] = 1
test_outpatient['inpatient'] = 0
test = pd.concat([test_inpatient, test_outpatient], axis=0, sort=False)
test_label['PotentialFraud'] = 1
test['PotentialFraud'] = 1

for (i,row) in test.iterrows():
    if row['Provider'] not in test_label['Provider']:
        row['PotentialFraud'] = 0

#Export train and test datasets
train.to_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\processed_data\train.csv')
test.to_csv(r'C:\Users\aengu\OneDrive\Desktop\school_stuff\Y4S1\DSA4262\medical-insurance-fraud\processed_data\test.csv')