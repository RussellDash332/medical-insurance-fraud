import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
** NOTE: The labels for the test set are not provided, rather it can be assumed that all the providers (UID) given in the file are all potential fraudsters

Idea on preprocessing training data
1) Combine inpatient and outpatient data with label for inpatient = 1, outpatient = 0
2) Add label from training data indicating if claim was potentially fraudulent
3)

"""

# This code combines Inpatient, Outpatient, Beneficiary and Provider data together (Dataset 1,2,3)

# Reading in datasets
train_inpatient = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Train_Inpatientdata.csv'))
train_outpatient = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Train_Outpatientdata.csv'))
train_beneficiary = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Train_Beneficiarydata.csv'))
train_label = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Train.csv'))

test_inpatient = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Test_Inpatientdata.csv'))
test_outpatient = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Test_Outpatientdata.csv'))
test_beneficiary = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Test_Beneficiarydata.csv'))
test_label = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraud_detection_data', 'Test.csv'))

## Replacing 2 with 0 for chronic conditions ,that means chroniv condition No is 0 and yes is 1
train_beneficiary = train_beneficiary.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

train_beneficiary = train_beneficiary.replace({'RenalDiseaseIndicator': 'Y'}, 1)

test_beneficiary = test_beneficiary.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

test_beneficiary = test_beneficiary.replace({'RenalDiseaseIndicator': 'Y'}, 1)

# Combine training datasets 
train_inpatient['inpatient'] = 1
train_outpatient['inpatient'] = 0
#train = pd.concat([train_inpatient, train_outpatient], axis=0, sort=False)
train = pd.merge(train_outpatient,train_inpatient,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode','inpatient'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode','inpatient']
                              ,how='outer')

train_alldata=pd.merge(train,train_beneficiary,left_on='BeneID',right_on='BeneID',how='inner')

#Combining test data
test_inpatient['inpatient'] = 1
test_outpatient['inpatient'] = 0
test = pd.merge(test_outpatient,test_inpatient,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode']
                              ,how='outer')

test_alldata = pd.merge(test,test_beneficiary,left_on='BeneID',right_on='BeneID',how='inner')

# Merging with Provider
train_allprovider = pd.merge(train_label,train_alldata,on='Provider')
test_allprovider = pd.merge(test_label,test_alldata,on='Provider')

# Removing unecessary columns
train_allprovider = train_allprovider.drop(['NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'State', 'County', 'AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'DiagnosisGroupCode'],axis=1)

test_allprovider = test_allprovider.drop(['NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'State', 'County', 'AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'DiagnosisGroupCode'],axis=1)

# Add code to combine with remaining datasets here

# Export train and test datasets
#train_allprovider.to_csv(os.path.join(os.getcwd(), 'processed_data', 'train.csv'))
#test_allprovider.to_csv(os.path.join(os.getcwd(), 'processed_data', 'test.csv'))
