import os
import random
import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset


def main():
    file = os.path.splitext(os.path.basename(__file__))[0]
    path = "../data"
    df_train = pd.read_csv((os.path.join(path, 'source', 'train.csv')))
    df_test = pd.read_csv((os.path.join(path, 'sorce', 'test.csv')))

    # Round all the numeric columns to 1 decimal place
    df_train = df_train.round(1)
    df_test = df_test.round(1)

    random_seed=0

    random.seed(random_seed)
    np.random.seed(random_seed)


    print('training size =', len(df_train), ', testing size =', len(df_test))

    # TASK 1 #
    variable = ["age", "gender", 
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_temperature", "triage_heartrate", "triage_resprate", 
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
            "chiefcom_dizziness", 
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV", 
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"]

    outcome = "outcome_hospitalization"

    # For the columns in the variable list, convert to string and replace the missing values with the word "missing"
    for col in variable:
        df_train[col] = df_train[col].astype(str)
        df_train[col] = df_train[col].replace("nan", "missing")
        df_test[col] = df_test[col].replace("nan", "missing")
        df_test[col] = df_test[col].astype(str)

    # Select the columns of interest and convert each row into a concatenated string of the form "column_name1: value1, column_name2: value2, ..."
    df_train['text'] = df_train[variable].apply(lambda x: '; '.join([f'{col}: {val}' for col, val in zip(variable, x)]), axis=1)
    df_test['text'] = df_test[variable].apply(lambda x: '; '.join([f'{col}: {val}' for col, val in zip(variable, x)]), axis=1)

    # Change the outcome column to "Yes" or "No"
    df_train[outcome] = df_train[outcome].apply(lambda x: 'Yes' if x else 'No')
    df_test[outcome] = df_test[outcome].apply(lambda x: 'Yes' if x else 'No')

    df_train_t1 = df_train[['text', outcome]]
    df_test_t1 = df_test[['text', outcome]]

    df_train_t1['text'] = 'Will this patient be admitted to hospital from the emergency department: ' + df_train_t1['text']
    df_test_t1['text'] = 'Will this patient be admitted to hospital from the emergency department: ' + df_test_t1['text']

    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            df_train_t1.rename(columns={outcome: 'label'})
            ),
        'test': Dataset.from_pandas(
            df_test_t1.rename(columns={outcome: 'label'})
            ),
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_1_{file}'))

    # TASK 2 #
    outcome = "outcome_critical"

    df_train[outcome] = df_train[outcome].apply(lambda x: 'Yes' if x else 'No')
    df_test[outcome] = df_test[outcome].apply(lambda x: 'Yes' if x else 'No')

    df_train_t2 = df_train[['text', outcome]]
    df_test_t2 = df_test[['text', outcome]]

    df_train_t2['text'] = 'Will this patient have a critical outcome (death or be admission to ICU within 12 hours): ' + df_train_t2['text']
    df_test_t2['text'] = 'Will this patient have a critical outcome (death or be admission to ICU within 12 hours): ' + df_test_t2['text']
    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            df_train_t2.rename(columns={outcome: 'label'})
            ),
        'test': Dataset.from_pandas(
            df_test_t2.rename(columns={outcome: 'label'})
            ),
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_2_{file}'))

    # TASK 3 #
    
    df_train = df_train[(df_train['outcome_hospitalization'] == "No")]
    df_test = df_test[(df_test['outcome_hospitalization'] == "No")].reset_index()

    new_variable = ["age", "gender", 
                
                "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
                "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
                
                "triage_pain", "triage_acuity",
                
                "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", 
                "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
                "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
                "chiefcom_dizziness",
                
                "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", 
                "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", 
                "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", 
                "cci_HIV",
                
                "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2",  
                "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
                "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", 
                "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
                
                "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", 
                "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"]

    outcome = "outcome_ed_revisit_3d"

    vars = [var for var in new_variable if var not in variable]

    for col in vars:
        df_train[col] = df_train[col].astype(str)
        df_train[col] = df_train[col].replace("nan", "missing")
        df_test[col] = df_test[col].replace("nan", "missing")
        df_test[col] = df_test[col].astype(str)

    # Select the columns of interest and convert each row into a concatenated string of the form "column_name1: value1, column_name2: value2, ..."
    df_train['text'] = df_train[new_variable].apply(lambda x: '; '.join([f'{col}: {val}' for col, val in zip(new_variable, x)]), axis=1)
    df_test['text'] = df_test[new_variable].apply(lambda x: '; '.join([f'{col}: {val}' for col, val in zip(new_variable, x)]), axis=1)

    # Change the outcome column to "Yes" or "No"
    df_train[outcome] = df_train[outcome].apply(lambda x: 'Yes' if x else 'No')
    df_test[outcome] = df_test[outcome].apply(lambda x: 'Yes' if x else 'No')

    df_train_t3 = df_train[['text', outcome]]
    df_test_t3 = df_test[['text', outcome]]

    df_train_t3['text'] = 'Will this patient have a representation to the emergency department within 3 days: ' + df_train_t3['text']
    df_test_t3['text'] = 'Will this patient have a representation to the emergency department within 3 days: ' + df_test_t3['text']

    # Create a DatasetDict object containing the train and test datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            df_train_t3.rename(columns={outcome: 'label'})
            ),
        'test': Dataset.from_pandas(
            df_test_t3.rename(columns={outcome: 'label'})
            ),
    })

    # Save the DatasetDict object to disk
    dataset_dict.save_to_disk(os.path.join(path, "mimic_iv", f'mimic_iv_task_3_{file}'))


if __name__ == '__main__':
    main()