import os
import random
import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from tabular_lm.float_helpers import create_tab_text

def main():
    file = os.path.splitext(os.path.basename(__file__))[0]
    path = "../data"
    df_train = pd.read_csv((os.path.join(path, 'source', 'train.csv')))
    df_test = pd.read_csv((os.path.join(path, 'sorce', 'test.csv')))


    print('training size =', len(df_train), ', testing size =', len(df_test))

    # Make descriptive column names for comorbidity features
    eci = [col for col in df_train.columns if col.startswith('eci')]
    cci = [col for col in df_train.columns if col.startswith('cci')]
    comorbidity_names = {
        eci[0]: 'Arrhythmia', eci[1]: 'Valvular Disease', eci[2]: 'Pulmonary Hypertension',
        eci[3]: 'Hypertension Without Complications', eci[4]: 'Hypertension With Complications',
        eci[5]: 'Neurodegenerative Disease', eci[6]: 'Hypothyroidism', eci[7]: 'Lymphoma',
        eci[8]: 'Coagulopathy', eci[9]: 'Obesity', eci[10]: 'Weight Loss',
        eci[11]: 'Fluids and Electrolyte Disorders', eci[12]: 'Anemia From Blood Loss',
        eci[13]: 'Anemia From Other Causes', eci[14]: 'Alcohol Abuse', eci[15]: 'Drug Abuse',
        eci[16]: 'Psychosis', eci[17]: 'Depression',
        cci[0]: 'Myocardial Infarction', cci[1]: 'Congestive Heart Failure',
        cci[2]: 'Peripheral Vascular Disease', cci[3]: 'Stroke', cci[4]: 'Dementia',
        cci[5]: 'Pulmonary Disease', cci[6]: 'Rheumatic Disease', cci[7]: 'Peptic Ulcer Disease',
        cci[8]: 'Mild Liver Disease', cci[9]: 'Diabetes Mellitus without Complications',
        cci[10]: 'Diabetes Mellitus with Complications', cci[11]: 'Hemiplegia',
        cci[12]: 'Moderate to Severe Chronic Kidney Disease', cci[13]: 'Cancer without Metastasis', cci[14]: 'Moderate to Severe Liver Disease',
        cci[15]: 'Cancer with Metastasis', cci[16]: 'HIV/AIDS'
    }

    # Change all column names starting with cci to more readable names
    df_train.rename(columns=comorbidity_names, inplace=True)
    df_test.rename(columns=comorbidity_names, inplace=True)

    # Create a text column for the CCI, containing a list of comorbidities taken from the CCI columns
    df_train['Comorbidities'] = df_train[comorbidity_names.values()].apply(lambda x: ', '.join(x[x==1].index), axis=1)
    df_test['Comorbidities'] = df_test[comorbidity_names.values()].apply(lambda x: ', '.join(x[x==1].index), axis=1)

    # Now do the same as above for chiefcom
    chiefcom = [col for col in df_train.columns if col.startswith('chiefcom_')]
    chiefcom_names = {
        chiefcom[0]: 'Chest Pain', chiefcom[1]: 'Abdominal Pain', chiefcom[2]: 'Headache',
        chiefcom[3]: 'Shortness of Breath', chiefcom[4]: 'Back Pain', chiefcom[5]: 'Cough',
        chiefcom[6]: 'Nausea/Vomiting', chiefcom[7]: 'Fever/Chills', chiefcom[8]: 'Syncope',
        chiefcom[9]: 'Dizziness'
    }

    # Change all column names starting with chiefcom to more readable names
    df_train.rename(columns=chiefcom_names, inplace=True)
    df_test.rename(columns=chiefcom_names, inplace=True)

    # Create a text column for the chiefcom, containing a list of chief complaints taken from the chiefcom columns
    df_train['Chief Complaint(s)'] = df_train[chiefcom_names.values()].apply(lambda x: ', '.join(x[x==1].index), axis=1)
    df_test['Chief Complaint(s)'] = df_test[chiefcom_names.values()].apply(lambda x: ', '.join(x[x==1].index), axis=1)

    # Create 3 new columns for the number of presentations (ED, hospital, ICU) in the last 30, 90 and 365
    presentations_names = ["n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d"]

    # Create a text column for the number of ED visits, hospitalizations, and ICU stays in the past 30 days of the form "ED: 1, Hosp: 2, ICU: 3"
    df_train['ED, Hospital and ICU stays in Past 30 Days'] = df_train[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[0], x[3], x[6]), axis=1)
    df_test['ED, Hospital and ICU stays in Past 30 Days'] = df_test[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[0], x[3], x[6]), axis=1)

    # Create a text column for the number of ED visits, hospitalizations, and ICU stays in the past 90 days of the form "ED: 1, Hosp: 2, ICU: 3"
    df_train['ED, Hospital and ICU stays in Past 90 Days'] = df_train[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[1], x[4], x[7]), axis=1)
    df_test['ED, Hospital and ICU stays in Past 90 Days'] = df_test[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[1], x[4], x[7]), axis=1)

    # Create a text column for the number of ED visits, hospitalizations, and ICU stays in the past 365 days of the form "ED: 1, Hosp: 2, ICU: 3"
    df_train['ED, Hospital and ICU stays in Past 365 Days'] = df_train[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[2], x[5], x[8]), axis=1)
    df_test['ED, Hospital and ICU stays in Past 365 Days'] = df_test[presentations_names].apply(lambda x: 'ED: {}, Hosp: {}, ICU: {}'.format(x[2], x[5], x[8]), axis=1)

    # Change the following column names to more readable names:
    # "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity", "age", "gender"
    t1_names = {
        "triage_temperature": "Triage Temperature", "triage_heartrate": "Triage Heart Rate", "triage_resprate": "Triage Respiratory Rate",
        "triage_o2sat": "Triage Oxygen Saturation", "triage_sbp": "Triage Systolic Blood Pressure", "triage_dbp": "Triage Diastolic Blood Pressure",
        "triage_pain": "Triage Pain", "triage_acuity": "Triage Acuity", "age": "Age", "gender": "Sex", "chiefcomplaint": "Chief Complaints (free text)",
        "Chief Complaint(s)": "Chief Complaints (categorical)"
    }

    # Change the above column names to more readable names
    df_train.rename(columns=t1_names, inplace=True)
    df_test.rename(columns=t1_names, inplace=True)

    # Change the following column names to more readable names:
    # "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"
    t3_names = {
        "ed_temperature_last": "Temperature", "ed_heartrate_last": "Heart Rate", "ed_resprate_last": "Respiratory Rate",
        "ed_o2sat_last": "Oxygen Saturation", "ed_sbp_last": "Systolic Blood Pressure", "ed_dbp_last": "Diastolic Blood Pressure",
        "ed_los": "Length of Stay", "n_med": "Number of Medications", "n_medrecon": "Number of Medication Reconciliations"
    }

    # Change the above column names to more readable names
    df_train.rename(columns=t3_names, inplace=True)
    df_test.rename(columns=t3_names, inplace=True)

    # Now collate the new column names into a list
    variables = ['Comorbidities', 'ED, Hospital and ICU stays in Past 30 Days',
                'ED, Hospital and ICU stays in Past 90 Days', 'ED, Hospital and ICU stays in Past 365 Days',
                ]

    # Add the other columns to the list
    variables_t1 = variables + list(t1_names.values())

    outcome = "outcome_hospitalization"


    df_train_t1 = create_tab_text(df_train, variables_t1, outcome)
    df_test_t1 = create_tab_text(df_test, variables_t1, outcome, norm_dataframe=df_train)

    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': df_train_t1,
        'test': df_test_t1
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_1_{file}'))


    outcome = "outcome_critical"

    df_train_t2 = create_tab_text(df_train, variables_t1, outcome)
    df_test_t2 = create_tab_text(df_test, variables_t1, outcome, norm_dataframe=df_train)

    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': df_train_t2,
        'test': df_test_t2
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_2_{file}'))


    outcome = "outcome_ed_revisit_3d"
    variables_t3 = ["Age", "Sex"] + variables + list(t3_names.values()) + ["Chief Complaints (free text)", "Chief Complaints (categorical)"]


    train_temp = df_train[(df_train['outcome_hospitalization'] == False)]
    test_temp = df_test[(df_test['outcome_hospitalization'] == False)]

    df_train_t3 = create_tab_text(train_temp, variables_t3, outcome)
    df_test_t3 = create_tab_text(test_temp, variables_t3, outcome, norm_dataframe=train_temp)

    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': df_train_t3,
        'test': df_test_t3
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_3_{file}'))

if __name__ == '__main__':
    main()