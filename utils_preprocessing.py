import math
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from datetime import datetime
from datetime import timedelta
from dateutil import relativedelta

import utils

from sklearn.preprocessing import MinMaxScaler

'''
Preprocess BMI data
'''
def preprocess_bmi(df_to_process, data_dir):
    df = df_to_process.copy()

    # delete column Height_in_Meters if exists  
    if 'Height_in_Meters' in df.columns:
        df.drop(columns='Height_in_Meters', inplace=True)

    # Read data from raw FVC CSV file
    data_file = f'{data_dir}/PROACT_VITALSIGNS.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')


    # Convert Weigth to KG for those with units in Pounds
    # get rows with the column Height not NaN
    df_weigth = df_raw.loc[(df_raw['Weight_Units']=='Pounds')].copy()
    df_weigth['Weight_in_kg'] = np.round((df_weigth['Weight'] * 0.453592), 1) # constant to convert
    df_raw.loc[(df_weigth.index), 'Weight'] = df_weigth['Weight_in_kg']


    # Get info about subject's Height and update on Static CSV file
    # Note: some heigths were measured in Inches and will be converted to Centimeters.
    
    # 1) Create Height_in_Centimeters column, converting those heights in INCHES to centimeters

    # get rows with the column Height not NaN
    df_heights = df_raw.loc[(df_raw['Height'].isnull()==False), ['subject_id', 'Height', 'Height_Units']].copy()

    # convert Inches to Centimeters
    df_inches = df_heights[(df_heights['Height_Units']=='Inches')].copy()
    df_inches['Height_in_Centimeters'] = np.round((df_inches['Height'] * 2.54), 1) # constant to convert
    df_heights.loc[(df_inches.index), 'Height_in_Centimeters'] = np.round(df_inches['Height_in_Centimeters'],0)

    # create a column to represent the final Height to be used (in Meters and Centimeters)
    df_heights['Height_in_Centimeters'] = np.round(df_heights['Height'], 0)
    df_heights['Height_in_Meters'] = np.round( (df_heights['Height'] / 100), 2)

    # group by subject_ID and keep the first non-NaN value for each column (using the first() method)
    df_heights = df_heights.groupby(['subject_id']).first().reset_index()

    #delete rows that have height <= 0.80 centimeters (possible measurement errors) (n=175)
    to_delete = df_heights.loc[(df_heights.Height_in_Meters) <= 0.80].copy()
    print(f'{utils.get_quantity_of_rows(to_delete)} samples will be deleted due to Height < 0.80')
    df_heights.drop(index=to_delete.index, inplace=True)

    # delete unnecessary columns
    df_heights.drop(
        columns=[
            'Height', 
            'Height_Units',
            'Height_in_Centimeters'
        ], 
        inplace=True
    )

    # 2) Add subjects' Heights in the Static data file, according to those obtained from
    # Vital Signs data file
    # create the height column on STATIC data, with default value = NaN
    df = utils.join_datasets_by_key(df_main=df, df_to_join=df_heights, key_name='subject_id', how='left')


    # Get longitudinal information about the subjects' weigth, aiming calculate the
    # Weight data file containing weigths history, Delta's, and BMI (Body Mass Index)

    # 1) Create Weigth_in_Kilograms column, converting those weigths in POUNDS to kilograms
    # get rows with the column Weigth not NaN
    df_weigths = df_raw.loc[(df_raw['Weight'].isnull()==False), ['subject_id', 'Vital_Signs_Delta',
                                                                'Weight', 'Weight_Units']].copy()

    # rename Vital_Signs_Delta column to Delta
    df_weigths.rename(columns={'Vital_Signs_Delta': 'Delta'}, inplace=True)

    # create a column to represent the final Height to be used
    df_weigths['Weigth_in_Kilograms'] = df_weigths['Weight']

    # convert Inches to Centimeters
    df_pounds = df_weigths[(df_weigths['Weight_Units']=='Pounds')].copy()
    df_pounds['Weigth_in_Kilograms'] = np.round((df_pounds['Weight'] / 2.2046), 1) # constant to convert

    # update dataset with weigths
    df_weigths.loc[(df_pounds.index), 'Weigth_in_Kilograms'] = df_pounds['Weigth_in_Kilograms']

    # delete rows that have weigth <= 25 kg (possible measurement errors) (n=165)
    to_delete = df_weigths.loc[(df_weigths.Weigth_in_Kilograms) < 25].copy()
    print(f'{utils.get_quantity_of_rows(to_delete)} samples will be deleted due to Weigth < 25 kg')
    df_weigths.drop(index=to_delete.index, inplace=True)

    # delete unnecessary columns
    df_weigths.drop(columns=['Weight', 'Weight_Units'], inplace=True)


    # Join the 2 datasets to calcute ALSFRS Delta from Symptoms_Onset
    # get only columns subject_id and Symptoms_Onset_Delta
    df_patients = df[[
        'subject_id', 
        'Symptoms_Onset_Delta',
        'Height_in_Meters',
    ]].copy()
    df_temp = utils.join_datasets_by_key(df_main=df_weigths, df_to_join=df_patients, key_name='subject_id')


    # Create Delta_from_Symptoms columns (in  days and  months )
    #calculate in DAYS
    df_temp['Delta_from_Symptoms_Onset_in_Days'] = df_temp.Delta + np.abs(df_temp.Symptoms_Onset_Delta)
    #calculate in MONTHS
    df_temp['Delta_from_Symptoms_Onset'] = np.NaN
    in_months = df_temp['Delta_from_Symptoms_Onset_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    df_temp.loc[df_temp.index,'Delta_from_Symptoms_Onset'] = in_months

    # Drop rows with NaN in Delta_from_Symptoms_Onset column
    df_temp.dropna(subset=['Delta_from_Symptoms_Onset'], inplace=True)
    df_temp.dropna(subset=['Height_in_Meters'], inplace=True)

    # Delete some unnecessary columns
    cols_to_remove = [
        'Delta', 
        'Symptoms_Onset_Delta', 
    ]
    df_temp.drop(columns=cols_to_remove, inplace=True)


    # Calculate the subjects' BMI (Body Mass Index)
    # Formula: BMI = Weigth (kg) / Height2 (m)
    # get ONLY rows with information about WEIGHT and HEIGTH
    df_bmi = df_temp.loc[(df_temp['Weigth_in_Kilograms'].isnull()==False) & (df_temp['Height_in_Meters'].isnull()==False)].copy()

    # calculate the BMI for each row
    df_bmi['BMI'] = df_bmi.apply( lambda x: utils.calculate_BMI(x['Weigth_in_Kilograms'], x['Height_in_Meters']), 
                                axis=1) 

    # create columns in the main data-frame to store the calculated values
    df_temp['BMI'] = np.NaN

    # store the calculated values in the main data-frame
    df_temp.loc[(df_bmi.index), 'BMI'] = df_bmi['BMI']


    # Solve the problem were rows have duplicated Delta_from_Symptoms_Onset
    #sort rows by 'subject_id', 'Delta_from_Symptoms_Onset', and 'Slope'
    df_remove_duplicated = df_temp.sort_values(['subject_id', 'Delta_from_Symptoms_Onset', 'Weigth_in_Kilograms'])
    #get only the last row for each 'subject_id' and'Delta_from_Symptoms_Onset'
    df_remove_duplicated = df_remove_duplicated.groupby(['subject_id', 'Delta_from_Symptoms_Onset']).last().reset_index()
    df_raw = df_remove_duplicated.copy()

    #
    return df, df_raw


'''
Preprocess SVC data
'''
def preprocess_svc(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read data from raw FVC CSV file
    data_file = f'{data_dir}/PROACT_SVC.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')

    # Check if exist rows with NaN on Slow_vital_Capacity_Delta column
    # and, if so, DELETE them
    to_delete = df_raw.loc[(df_raw.Slow_vital_Capacity_Delta.isnull())]
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)


    # The value for SVC is present in 3 columns (Subject_Liters_Trial_1, Trial_2,
    # and Trial_3).
    # A new column will be created and set to the max value present in these 3 columns.
    cols = ['Subject_Liters_Trial_1', 'Subject_Liters__Trial_2_', 'Subject_Liters__Trial_3_']

    # create a new DF with these columns values
    df_max_svc = df_raw[cols].copy()

    # replace all NaN values to -1
    df_max_svc.fillna(-1.0, inplace=True)

    # get the max value present in columns Subject_Liters_Trial_1, Subject_Liters_Trial_2, and Subject_Liters_Trial_3
    df_max_svc['max'] = df_max_svc[cols].values.max(axis=1)

    #replace -1 values with NaN
    df_max_svc['max'].replace(-1.0, np.NaN, inplace=True)

    # create the column SVC in df_raw and set to max value calculated
    df_raw['SVC'] = df_max_svc['max']


    # A new column will be created to represent the percentage of normal, following the same logic of the previous step
    # The value is present in 3 columns (pct_of_Normal_Trial_1, Trial_2,
    # and Trial_3).
    cols = ['pct_of_Normal_Trial_1', 'pct_of_Normal_Trial_2', 'pct_of_Normal_Trial_3']

    # create a new DF with these columns values
    df_max_svc = df_raw[cols].copy()

    # replace all NaN values to -1
    df_max_svc.fillna(-1.0, inplace=True)

    # get the max value present in columns 
    df_max_svc['max'] = df_max_svc[cols].values.max(axis=1)

    #replace -1 values with NaN
    df_max_svc['max'].replace(-1.0, np.NaN, inplace=True)

    df_max_svc

    # create the column FVC in df_raw and set to max value calculated
    df_raw['SVC_Perc_of_Normal'] = df_max_svc['max']


    # Check if exist rows with NaN on SVC column and, if so, DELETE them
    to_delete = df_raw.loc[(df_raw.SVC.isnull() == True)]
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete,verbose=False)
    
    # Join the 2 datasets to calcute ALSFRS Delta from Symptoms_Onset
    # get only columns subject_id and Symptoms_Onset_Delta
    df_patients = df[[
        'subject_id', 
        'Symptoms_Onset_Delta'
    ]].copy()
    df_raw = utils.join_datasets_by_key(df_main=df_raw, df_to_join=df_patients, key_name='subject_id')


    # Create Delta_from_Symptoms columns (in  days and  months)
    #calculate in DAYS
    df_raw['Delta_from_Symptoms_Onset_in_Days'] = df_raw.Slow_vital_Capacity_Delta + np.abs(df_raw.Symptoms_Onset_Delta)
    #calculate in MONTHS
    df_raw['Delta_from_Symptoms_Onset'] = np.NaN
    in_months = df_raw['Delta_from_Symptoms_Onset_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    df_raw.loc[df_raw.index,'Delta_from_Symptoms_Onset'] = in_months


    # Calculate percentage of normal for those with SVC_Perc_of_Normal = NaN,
    # subject_normal <> NaN and SVC <> NaN
    to_calculate = df_raw.loc[
         (df_raw.SVC_Perc_of_Normal.isnull())
        &(df_raw.Subject_Normal.isnull() == False)
        &(df_raw.SVC.isnull() == False)
    ].copy()
    df_raw.loc[to_calculate.index, 'SVC_Perc_of_Normal'] = np.round(df_raw['SVC']/df_raw['Subject_Normal']*100, 0)


    # Delete some unnecessary columns
    cols_to_remove = ['Slow_vital_Capacity_Delta', 
                    'Subject_Liters_Trial_1', 
                    'pct_of_Normal_Trial_1', 
                    'Subject_Liters__Trial_2_', 
                    'pct_of_Normal_Trial_2', 
                    'Subject_Liters__Trial_3_', 
                    'pct_of_Normal_Trial_3',
                    'Slow_Vital_Capacity_Units',
                    'Symptoms_Onset_Delta', 
                    'Subject_Normal',
                    ]
    df_raw.drop(columns=cols_to_remove, inplace=True)


    # Drop rows with NaN values in Delta_from_Symptoms_Onset and SVC_Perc_of_Normal
    df_raw.dropna(subset=['Delta_from_Symptoms_Onset'], inplace=True)
    df_raw.dropna(subset=['SVC_Perc_of_Normal'], inplace=True)


    # Solve the problem were rows have duplicated Delta_from_Symptoms_Onset
    # sort rows by 'subject_id', 'Delta_from_Symptoms_Onset', and 'Slope'
    df_remove_duplicated = df_raw.sort_values(['subject_id', 
                                        'Delta_from_Symptoms_Onset_in_Days', 
                                        'SVC'])   
    # get only the last row for each 'subject_id' and'Delta_from_Symptoms_Onset'
    df_remove_duplicated = df_remove_duplicated.groupby(['subject_id', 
                                                        'Delta_from_Symptoms_Onset'
                                                        ]).last().reset_index()

    df_raw = df_remove_duplicated.copy()
    

    #
    return df_raw


'''
Preprocess FVC data
'''
def preprocess_fvc(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read data from raw FVC CSV file
    data_file = f'{data_dir}/PROACT_FVC.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')

    # Check if exist rows with NaN on Forced_Vital_Capacity_Delta column
    # and, if so, DELETE them
    to_delete = df_raw.loc[(df_raw.Forced_Vital_Capacity_Delta.isnull())]
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)
    

    # The value for FVC is present in 3 columns (Subject_Liters_Trial_1, Trial_2, and Trial_3).
    # A new column will be created and set to the max value present in these 3 columns.
    # Another column will be created to represent the percentage of normal, following the same logic
    cols = ['Subject_Liters_Trial_1', 'Subject_Liters_Trial_2', 'Subject_Liters_Trial_3']

    # create a new DF with these columns values
    df_max_fvc = df_raw[cols].copy()

    # replace all NaN values to -1
    df_max_fvc.fillna(-1.0, inplace=True)

    # get the max value present in columns Subject_Liters_Trial_1, Subject_Liters_Trial_2, and Subject_Liters_Trial_3
    df_max_fvc['max'] = df_max_fvc[cols].values.max(axis=1)

    #replace -1 values with NaN
    df_max_fvc['max'].replace(-1.0, np.NaN, inplace=True)

    # create the column FVC in df_raw and set to max value calculated
    df_raw['FVC'] = df_max_fvc['max']



    # Another column will be created to represent the percentage of normal, following the same logic of the previous step
    # The value is present in 3 columns (pct_of_Normal_Trial_1, Trial_2, and Trial_3).
    cols = ['pct_of_Normal_Trial_1', 'pct_of_Normal_Trial_2', 'pct_of_Normal_Trial_3']

    # create a new DF with these columns values
    df_max_fvc = df_raw[cols].copy()

    # replace all NaN values to -1
    df_max_fvc.fillna(-1.0, inplace=True)

    # get the max value present in columns 
    df_max_fvc['max'] = df_max_fvc[cols].values.max(axis=1)

    #replace -1 values with NaN
    df_max_fvc['max'].replace(-1.0, np.NaN, inplace=True)

    df_max_fvc

    # create the column FVC in df_raw and set to max value calculated
    df_raw['FVC_Perc_of_Normal'] = df_max_fvc['max']


    # Check if exist rows with NaN on FVC column and, if so, DELETE them
    to_delete = df_raw.loc[(df_raw.FVC.isnull() == True)]
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)


    # Join the 2 datasets to calcute ALSFRS Delta from Symptoms_Onset
    # get only columns subject_id and Symptoms_Onset_Delta
    df_patients = df[[
        'subject_id', 
        'Symptoms_Onset_Delta'
    ]].copy()
    df_raw = utils.join_datasets_by_key(df_main=df_raw, df_to_join=df_patients, key_name='subject_id')


    # Create Delta_from_Symptoms columns (in  days and  months)
    #calculate in DAYS
    df_raw['Delta_from_Symptoms_Onset_in_Days'] = df_raw.Forced_Vital_Capacity_Delta + np.abs(df_raw.Symptoms_Onset_Delta)
    #calculate in MONTHS
    df_raw['Delta_from_Symptoms_Onset'] = np.NaN
    in_months = df_raw['Delta_from_Symptoms_Onset_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    df_raw.loc[df_raw.index,'Delta_from_Symptoms_Onset'] = in_months
    
    
    # Calculate percentage of normal for those with FVC_Perc_of_Normal = NaN,
    # subject_normal <> NaN and FVC <> NaN
    to_calculate = df_raw.loc[
         (df_raw.FVC_Perc_of_Normal.isnull())
        &(df_raw.subject_normal.isnull() == False)
        &(df_raw.FVC.isnull() == False)
    ].copy()
    df_raw.loc[to_calculate.index, 'FVC_Perc_of_Normal'] = np.round(df_raw['FVC']/df_raw['subject_normal']*100, 0)
    

    # Delete some unnecessary columns
    cols_to_remove = ['Forced_Vital_Capacity_Delta', 
                    'Subject_Liters_Trial_1', 
                    'pct_of_Normal_Trial_1', 
                    'Subject_Liters_Trial_2', 
                    'pct_of_Normal_Trial_2', 
                    'Subject_Liters_Trial_3', 
                    'pct_of_Normal_Trial_3',
                    'Forced_Vital_Capacity_Units',
                    'Symptoms_Onset_Delta', 
                    'subject_normal',
                    ]

    df_raw.drop(
        columns=cols_to_remove, 
        inplace=True
    )


    # Drop rows with NaN values in the columns Delta_from_Symptoms_Onset and FVC_Perc_of_Normal
    df_raw.dropna(subset=['Delta_from_Symptoms_Onset'], inplace=True)
    df_raw.dropna(subset=['FVC_Perc_of_Normal'], inplace=True)    


    # Solve the problem were rows have duplicated Delta_from_Symptoms_Onset
    # sort rows by 'subject_id', 'Delta_from_Symptoms_Onset', and 'Slope'
    df_remove_duplicated = df_raw.sort_values(['subject_id', 
                                        'Delta_from_Symptoms_Onset_in_Days', 
                                        'FVC'])

    # get only the last row for each 'subject_id' and'Delta_from_Symptoms_Onset'
    df_remove_duplicated = df_remove_duplicated.groupby(['subject_id', 
                                                        'Delta_from_Symptoms_Onset'
                                                        ]).last().reset_index()

    df_raw = df_remove_duplicated.copy()

    #
    return df_raw

'''
Preprocess ALSFRS data
'''
def preprocess_alsfrs(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read data from the raw ALSFRS CSV file
    data_file = f'{data_dir}/PROACT_ALSFRS.csv'
    df_raw = pd.read_csv(data_file, delimiter=',', 
                        dtype={'Mode_of_Administration': 'str', 
                                'ALSFRS_Responded_By': 'str'}
                        )

    # Drop rows with NaN value in the ALSFRS_Delta column
    to_delete = df_raw.loc[(df_raw.ALSFRS_Delta.isnull())].copy()
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)

    # Remove unnecessary columns
    cols_to_remove = ['Mode_of_Administration','ALSFRS_Responded_By']
    df_raw.drop(columns=cols_to_remove, inplace=True)


    # Create the Q5 column to compute the correct value for Question-5,
    # which is divided in a) and b) subparts
    df_raw['Q5_Cutting'] = np.NaN
    df_raw['Patient_with_Gastrostomy'] = np.NaN

    # without Gastrostomy
    df_raw.loc[(df_raw['Q5a_Cutting_without_Gastrostomy'].isnull()==False), 'Q5_Cutting'] = df_raw['Q5a_Cutting_without_Gastrostomy']
    df_raw.loc[(df_raw['Q5a_Cutting_without_Gastrostomy'].isnull()==False), 'Patient_with_Gastrostomy'] = False

    # with Gastrostomy
    df_raw.loc[(df_raw['Q5b_Cutting_with_Gastrostomy'].isnull()==False), 'Q5_Cutting'] = df_raw['Q5b_Cutting_with_Gastrostomy']
    df_raw.loc[(df_raw['Q5b_Cutting_with_Gastrostomy'].isnull()==False), 'Patient_with_Gastrostomy'] = True

    #drop columns Q5a and Q5b
    df_raw.drop(columns=['Q5a_Cutting_without_Gastrostomy', 'Q5b_Cutting_with_Gastrostomy'], axis=1, inplace=True)


    # Join question 10 from ALSFRS with question 10 (dyspnea) from ALSFRS-R
    to_update = df_raw.loc[
        (df_raw.Q10_Respiratory.isnull())
        &(df_raw.R_1_Dyspnea.isnull()==False)
    ].copy()
    # set ALSFRS-R Q10_Respiratory equal to R_1_Dyspnea 
    df_raw.loc[to_update.index, 'Q10_Respiratory'] = df_raw['R_1_Dyspnea']


    # Delete rows having NaN in the Q1 to Q10 columns
    to_delete = df_raw.loc[
          (df_raw.Q1_Speech.isnull())
        | (df_raw.Q2_Salivation.isnull())
        | (df_raw.Q3_Swallowing.isnull())
        | (df_raw.Q4_Handwriting.isnull())
        | (df_raw.Q5_Cutting.isnull())
        | (df_raw.Q6_Dressing_and_Hygiene.isnull())
        | (df_raw.Q7_Turning_in_Bed.isnull())
        | (df_raw.Q8_Walking.isnull())
        | (df_raw.Q9_Climbing_Stairs.isnull())
        | (df_raw.Q10_Respiratory.isnull())
    ].copy()
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)


    # Get only necessary columns
    df_raw = df_raw[[
            'subject_id',
            'Q1_Speech',
            'Q2_Salivation',
            'Q3_Swallowing',
            'Q4_Handwriting',
            'Q5_Cutting',
            'Q6_Dressing_and_Hygiene',
            'Q7_Turning_in_Bed',
            'Q8_Walking',
            'Q9_Climbing_Stairs',
            'Q10_Respiratory',
            'ALSFRS_Delta',
            # 'ALSFRS_Total',
            'Patient_with_Gastrostomy',
            ]].copy()
    

    # Join the 2 datasets to calcute ALSFRS Delta from Symptoms_Onset
    # get only columns subject_id and Symptoms_Onset_Delta
    df_patients = df[[
        'subject_id', 
        'Symptoms_Onset_Delta'
    ]].copy()

    df_raw = utils.join_datasets_by_key(
        df_main=df_raw, 
        df_to_join=df_patients, 
        key_name='subject_id', 
        raise_error=True
    )

    # Create Delta_from_Symptoms_Onset columns (in  days and months)
    #calculate in DAYS
    df_raw['Delta_from_Symptoms_Onset_in_Days'] = df_raw.ALSFRS_Delta + np.abs(df_raw.Symptoms_Onset_Delta)

    #calculate in MONTHS
    df_raw['Delta_from_Symptoms_Onset'] = np.NaN
    in_months = df_raw['Delta_from_Symptoms_Onset_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    df_raw.loc[df_raw.index,'Delta_from_Symptoms_Onset'] = in_months

    # Delete some unnecessary columns and
    to_delete = [
        'Symptoms_Onset_Delta', 
    ]
    df_raw.drop(
        columns=to_delete, 
        inplace=True
    )

    # Delete rows that have Delta_from_Symptoms_Onset = NaN
    to_delete = df_raw.loc[
        (df_raw.Delta_from_Symptoms_Onset.isnull()==True)
    ]
    df_raw = utils.remove_rows(df=df_raw, to_delete=to_delete, verbose=False)


    # Solve the problem were rows have duplicated Delta_from_Symptoms_Onset
    # sort rows by 'subject_id', 'Delta_from_Symptoms_Onset'
    df_remove_duplicated = df_raw.sort_values(['subject_id', 'Delta_from_Symptoms_Onset_in_Days'])
    
    #get only the last row for each 'subject_id' and'Delta_from_Symptoms_Onset'
    df_remove_duplicated = df_remove_duplicated.groupby(['subject_id', 'Delta_from_Symptoms_Onset']).last().reset_index()
    
    df_raw = df_remove_duplicated.copy()


    # =======================================================================
    # Calculate the Slopes from Onset for each row for ALSFRS
    # =======================================================================
    
    # Sort the data by subject_id and Delta_from_Symptoms_Onset_in_Days columns
    df_raw.sort_values(by=['subject_id','Delta_from_Symptoms_Onset_in_Days'], inplace=True)
    
    # Create Slopes for each ALSFRS question (1−10)
    cols_to_create = [
        'Q1_Speech', 
        'Q2_Salivation',       
        'Q3_Swallowing', 
        'Q4_Handwriting', 
        'Q5_Cutting',
        'Q6_Dressing_and_Hygiene', 
        'Q7_Turning_in_Bed', 
        'Q8_Walking', 
        'Q9_Climbing_Stairs',
        'Q10_Respiratory', 
    ]
    max_score = 4

    for col in cols_to_create:
        #define name of the column to create
        c = f'Slope_from_Onset_{col}'
        #create the column
        df_raw[c] = np.NaN
        # calculate the slope = (Max_SubScore - SubScore)/Delta_from_Symptoms_Onset
        slope = (max_score - df_raw[col]) / df_raw['Delta_from_Symptoms_Onset']
        # round to 2 decimal places
        df_raw[c] = np.round(slope, 2) 


    # =======================================================================
    # Create Regions-Involved Groups and set their values
    # =======================================================================
    # Their values can be TRUE or FALSE, being TRUE will indicate the lost of at least 1 point in the maximum region score:
    # - Bulbar     = questions 1–3     Max. region score = 12
    # - Upper Limb = questions 4–5     Max. region score = 8 
    # - Lower Limb = questions 8–9     Max. region score = 8 
    # - Respiratory= question  10      Max. region score = 4
    #
    #      Reference: Manera et. al. (2019)

    # create Region_Involved_Bulbar column
    df_raw['Region_Involved_Bulbar'] = \
        ((df_raw['Q1_Speech'] + df_raw['Q2_Salivation'] + df_raw['Q3_Swallowing']) < 12) * 1.0

    # create Region_Involved_Upper_Limb column
    df_raw['Region_Involved_Upper_Limb'] = \
        ((df_raw['Q4_Handwriting'] + df_raw['Q5_Cutting'] ) < 8) * 1.0

    # create Region_Involved_Lower_Limb column
    df_raw['Region_Involved_Lower_Limb'] = \
        ((df_raw['Q8_Walking'] + df_raw['Q9_Climbing_Stairs'] ) < 8) * 1.0

    # create Region_Involved_Respiratory column
    df_raw['Region_Involved_Respiratory'] = ((df_raw['Q10_Respiratory']) < 4) * 1.0

    # Create Qty-Regions-Involved summing all 4 groups (Bulbar, Upper Limb, Lower Limb, Respiratory)

    # create column
    df_raw['Qty_Regions_Involved'] = np.NaN

    # sum the 4 groups
    df_raw['Qty_Regions_Involved'] = \
        df_raw['Region_Involved_Bulbar'] + df_raw['Region_Involved_Upper_Limb']  \
        + df_raw['Region_Involved_Lower_Limb'] + df_raw['Region_Involved_Respiratory'] 




    #
    return df_raw



'''
Preprocess EL_ESCORIAL data
'''
def preprocess_el_escorial(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read RAW El Escorial Criteria CSV file and show some stats
    data_file = f'{data_dir}/PROACT_ELESCORIAL.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')

    # Remove unnecessary column delta_days
    df_raw.drop(columns=['delta_days'], inplace=True)

    # Rename el_escorial column to El_Escorial
    df_raw.rename(columns={"el_escorial": "El_Escorial"}, inplace=True)

    # Join the 2 datasets
    df = utils.join_datasets_by_key(df_main=df, df_to_join=df_raw, key_name='subject_id', how='left')



    #
    return df

'''
Preprocess RILUZOLE data
'''
def preprocess_riluzole(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read RAW Riluzole CSV file 
    data_file = f'{data_dir}/PROACT_RILUZOLE.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')

    # Rename some columns
    df_raw.rename(
        columns={
            "Subject_used_Riluzole": "Riluzole", 
            "Riluzole_use_Delta": "Riluzole_Delta", 
        }, 
        inplace=True
    )

    # Join the 2 datasets
    df = utils.join_datasets_by_key(df_main=df, df_to_join=df_raw, key_name='subject_id', how='left')

    # Replace NaN values in Riluzole column to "No" and Riluzole-Delta to "0"
    df.loc[(df.Riluzole.isnull()==True), 'Riluzole'] = 'No'
    df.loc[(df.Riluzole_Delta.isnull()==True), 'Riluzole_Delta'] = 0.0

    # Convert Riluzole column to Boolean datatype
    df.Riluzole = df.Riluzole.map( {'Yes':True ,'No':False}) 

    # Drop unnecessary columns
    df.drop(
        columns=[
            'Riluzole_Delta', 
        ], 
        inplace=True,
    )

    #
    return df

'''
Preprocess DEATH data
'''
def preprocess_death_data(df_to_process, data_dir):
    df = df_to_process.copy()

    # Read DeathData CSV file 
    data_file = f'{data_dir}/PROACT_DEATHDATA.csv'
    df_raw = pd.read_csv(data_file, delimiter=',')

    # Join the 2 datasets (renaming some columns)
    df = utils.join_datasets_by_key(df_main=df, df_to_join=df_raw, key_name='subject_id', how='left', 
                                    raise_error=True)
    # rename columns
    df.rename(
        columns={
            "Subject_Died": "Event_Dead", 
            "Death_Days": "Event_Dead_Delta"
        }, 
        inplace=True
    )


    # Convert event_dead column to Boolean datatype, converting NaN values to False (not dead)
    df.Event_Dead = df.Event_Dead.map(
        {
            'Yes' : True,
            'No'  : False, 
            np.NaN: False,
        }
    )


    # Create column Event_Dead_Delta_from_Onset with the same value of
    # the Last_Visit_Delta_from_Onset column for those samples with
    # Event_Dead = False (not died)
    df_event_dead_false = df.loc[(df.Event_Dead==False)].copy()
    df_event_dead_false['Event_Dead_Time_from_Onset'] = df_event_dead_false['Last_Visit_from_Onset']
    df.loc[df_event_dead_false.index,
        'Event_Dead_Time_from_Onset'] = df_event_dead_false['Event_Dead_Time_from_Onset']


    # Update column Event_Dead_Delta_from_Onset with the same value of
    # the Last_Visit_Delta_from_Onset column for those samples with 
    # Event_Dead = True (died) and Event_Dead_Delta = NaN
    df_event_dead_true = df.loc[
        (df.Event_Dead==True)
       &(df.Event_Dead_Delta.isnull() )
    ].copy()
    df.loc[df_event_dead_true.index,
        'Event_Dead_Time_from_Onset'] = df['Last_Visit_from_Onset']


    # Calculate column Event_Dead_Time_from_Onset (months) for those samples
    # with Event_Dead = True (died) and Event_Dead_Delta <> NaN
    df_to_update = df.loc[
        (df.Event_Dead==True)
       &(df.Event_Dead_Delta.isnull()==False )
    ].copy()

    df.loc[df_to_update.index,
      'Event_Dead_Time_from_Onset_in_days'] = df['Event_Dead_Delta'] + np.abs(df.Symptoms_Onset_Delta)

    in_months = df['Event_Dead_Time_from_Onset_in_days'].apply( lambda x: utils.calculate_months_from_days(x)) 

    df.loc[df_to_update.index,
        'Event_Dead_Time_from_Onset'] = in_months


    # Delete unncessary columns
    to_delete = [
        'Event_Dead_Delta',
        'Last_Visit_Delta',
        'Last_Visit_from_Onset',
        'Event_Dead_Time_from_Onset_in_days',
    ]

    df.drop(
        columns=to_delete, 
        inplace=True,
    )

    #
    return df



'''
Preprocess LAST_VISIT
'''
def preprocess_last_visit(df_to_process, data_dir):
    df = df_to_process.copy()

    # Get information about the Last Visit Delta registered for each patient
    # This information will be used to check Valid Uncensored and Censored patients

    data_files = [
        ['PROACT_ALSFRS'    , 'ALSFRS_Delta'],
        ['PROACT_FVC'       , 'Forced_Vital_Capacity_Delta'],
        ['PROACT_DEATHDATA' , 'Death_Days'],
        ['PROACT_LABS'      , 'Laboratory_Delta'],
        ['PROACT_RILUZOLE'  , 'Riluzole_use_Delta'],
        ['PROACT_SVC'       , 'Slow_vital_Capacity_Delta'],
        ['PROACT_VITALSIGNS', 'Vital_Signs_Delta'],  
        #
        ['PROACT_ALSHISTORY',       'Subject_ALS_History_Delta'],
        ['PROACT_DEMOGRAPHICS',     'Demographics_Delta'],
        ['PROACT_ELESCORIAL',       'delta_days'],
        ['PROACT_FAMILYHISTORY',    'Family_History_Delta'],
        ['PROACT_HANDGRIPSTRENGTH', 'MS_Delta'],
        ['PROACT_MUSCLESTRENGTH',   'MS_Delta'],
        ['PROACT_TREATMENT',        'Treatment_Group_Delta'],
        #start and end deltas
        ['PROACT_ADVERSEEVENTS',    'Start_Date_Delta'], #
        ['PROACT_ADVERSEEVENTS',    'End_Date_Delta'], #Start_Date_Delta
        ['PROACT_CONMEDS',          'Start_Delta'], #,
        ['PROACT_CONMEDS',          'Stop_Delta'], #Start_Delta,

    ]

    df_last_visit = pd.DataFrame(data=[], columns=['subject_id', 'Delta', 'Biomarker'])

    for data_file, col_delta in data_files:
        print(f' - Get Last_Visit registered in {data_file}')
        #set the name of CSV file
        csv_file = f'{data_dir}/{data_file}.csv'

        #read data and show some info
        df_raw = pd.read_csv(csv_file, 
                            delimiter=',', 
                            usecols=['subject_id', col_delta] #read only columns subject_id and delta
                            )
        
        
        #rename column Delta to standardize
        df_raw.rename(columns={col_delta: "Last_Visit_Delta"}, inplace=True)

        #sort data by subject_d and Delta in ascending order    
        df_raw.sort_values(['subject_id', 'Last_Visit_Delta'])
        
        #group by subject_id and get max Delta for each of them
        df_grouped = df_raw.groupby('subject_id').max()
        #reset index to re-organize columns index (solve problem of subject_id become the index of the DF)
        df_grouped.reset_index(inplace=True)
        
        #create column to represent the biomarker source of information
        df_grouped['Biomarker'] = data_file
        
        #concatenate data into "df_last_visit" dataFrame
        if df_last_visit.shape[0]==0:
            df_last_visit = df_grouped.copy()
        else:   
            df_last_visit = pd.concat([df_last_visit, df_grouped], ignore_index=True)


    #drop rows with NaN values    
    df_last_visit.dropna(inplace=True)    

    #sort data by subject_d and Delta in ascending order    
    df_last_visit.sort_values(['subject_id', 'Last_Visit_Delta'])

    #group by subject_id and get max Delta for each of them
    df_last_visit = df_last_visit.groupby('subject_id').max()
    df_last_visit.reset_index(inplace=True)


    # Join the Patients and Last_Visit dataFrames
    df_to_join = df_last_visit[['subject_id', 'Last_Visit_Delta']].copy()
    
    df = utils.join_datasets_by_key(df_main=df, 
                                    df_to_join=df_to_join, 
                                    key_name='subject_id', 
                                    how='left')

    
    # Calculate Last Visit in months from symptoms onset
    df['Last_Visit_from_Onset_in_Days'] = np.abs(df.Last_Visit_Delta) + np.abs(df.Symptoms_Onset_Delta)
    
    last_visit_in_months = df['Last_Visit_from_Onset_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    
    df.loc[df.index,'Last_Visit_from_Onset'] = last_visit_in_months
    
    # Drop irrelevant columns   
    irrelevant_cols = [
        'Last_Visit_from_Onset_in_Days', 
    ]

    df.drop(
        columns=irrelevant_cols, 
        inplace=True,
    )

    #
    return df






'''
Preprocess AGE_AT_ONSET
'''
def preprocess_age_at_onset(df_to_process):
    df = df_to_process.copy()


    # Create the Age_at_Onset column
    # Calculation based on difference between the Age (at trial entrance) and the
    # Symptoms_Onset_Delta columns
    df['Age_at_Onset'] = np.NaN
    
    
    # get only rows with values in Age and Symptoms_Onset_Delta columns
    df_calc_age_onset = df.loc[(df.Age.isnull()==False) & (df.Symptoms_Onset_Delta.isnull()==False)].copy()
    
    
    # calculate the age at symptoms onset
    ages_calculated = df_calc_age_onset.apply( 
        lambda x: utils.calculate_age_from_onset_delta(
            x['Age'], 
            x['Symptoms_Onset_Delta']), 
        axis=1
    ) 
    
    #update samples with the calculated Age_at_Onset
    df.loc[df_calc_age_onset.index,'Age_at_Onset'] = ages_calculated
        

    #define a dictionary with age ranges
    age_ranges = {
        '0-39' : [0, 39],
        '40-49': [40, 49],
        '50-59': [50, 59],
        '60-69': [60, 69],
        '70+'  : [70, 999],
    }


    # Create Age_Range column and set its value       
    df['Age_Range_at_Onset'] = np.NAN
    
    for key, value in age_ranges.items():
        label = key
        min_age = value[0]
        max_age = value[1] + 1
        indices = df[(df['Age_at_Onset'] >= min_age) & (df['Age_at_Onset'] < max_age)]
        df.loc[indices.index, 'Age_Range_at_Onset'] = label
    
    # Drop irrelevant columns   
    irrelevant_cols = [
        'Age_at_Onset', 
    ]

    df.drop(
        columns=irrelevant_cols, 
        inplace=True,
    )


    # rename columns
    df.rename(
        columns={
            'Age_Range_at_Onset': 'Age_at_Onset', 
        }, 
        inplace=True
    )    

    return df
    


'''
Preprocess DIAGNOSIS_DELAY
'''
def preprocess_diagnosis_delay(df_to_process):
    df = df_to_process.copy()

    # Calculate Diagnosis_Delay in months
    df['Diagnosis_Delay_in_Days'] = np.abs(df.Symptoms_Onset_Delta) - np.abs(df.Diagnosis_Delta)
    diagnosis_delay_in_months = df['Diagnosis_Delay_in_Days'].apply( lambda x: utils.calculate_months_from_days(x)) 
    df.loc[df.index,'Diagnosis_Delay'] = diagnosis_delay_in_months

    # # Codify Diagnosis_Delay
    # # - Long    : >  18 months     
    # # - Average  : >   8 and <= 18 months  
    # # - Short     : <=  8 months   
    # to_update = df.loc[(df.Diagnosis_Delay > 18)]
    # df.loc[to_update.index, 'Diagnosis_Delay_Coded'] = 'Long'
    
    # to_update = df.loc[(
    #     (df.Diagnosis_Delay > 8)
    #     &(df.Diagnosis_Delay <= 18)
    # )]
    # df.loc[to_update.index, 'Diagnosis_Delay_Coded'] = 'Average'
    
    # to_update = df.loc[(df.Diagnosis_Delay <= 8)]
    # df.loc[to_update.index, 'Diagnosis_Delay_Coded'] = 'Short'


    # Drop irrelevant columns   
    irrelevant_cols = [
        'Diagnosis_Delay_in_Days', 
    ]

    df.drop(
        columns=irrelevant_cols, 
        inplace=True,
    )

    return df

'''
Preprocess ALS HISTORY
'''
def preprocess_als_history(df):

    df_als_history = df.copy()

    # Create a new column called site_onset with the aim of standardize its values
    # to [Limb, Bulbar, Spine, Limb and Bulbar, and Other]
    df_als_history['site_onset'] = np.NaN
    df_als_history.head(3)


    # Set value to NaN for any columns having values different of 1
    columns = [
        'Site_of_Onset___Bulbar',
        'Site_of_Onset___Limb',
        'Site_of_Onset___Limb_and_Bulbar',
        'Site_of_Onset___Other',
        'Site_of_Onset___Other_Specify',
        'Site_of_Onset___Spine'
        ]
        
    for col in columns:
        df_als_history.loc[(df_als_history[col] != 1), col] = np.NaN


    # ===================================================================
    # 1) Set column site_onset for the BULBAR onset samples
    # ===================================================================
    # Update samples with Site_of_Onset = "Onset: Bulbar"
    df_als_history.loc[(df_als_history['Site_of_Onset']=='Onset: Bulbar'), 'site_onset'] = 'Bulbar'

    # Update samples with Site_of_Onset___Bulbar = 1
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Bulbar']==1)
    &(df_als_history['Site_of_Onset'].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb'].isnull()==True)
    &(df_als_history['Site_of_Onset___Spine'].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb_and_Bulbar'].isnull()==True)
    &(df_als_history['Site_of_Onset___Other'].isnull()==True)
    &(df_als_history['Site_of_Onset___Other_Specify'].isnull()==True)
    , 'site_onset'] = 'Bulbar' 


    # ===================================================================
    # 2) Set column site_onset for LIMB / SPINAL onset samples
    # ===================================================================
    # Update samples with Site_of_Onset = "Onset: Limb" or "Onset: Spine"
    df_als_history.loc[
        (df_als_history['Site_of_Onset']=='Onset: Limb')
        | (df_als_history['Site_of_Onset']=='Onset: Spine'), 'site_onset'] = 'Limb/Spinal'


    # Update samples with Site_of_Onset___Limb = 1 OR Site_of_Onset___Spine = 1
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Limb']==1)
    &(df_als_history['Site_of_Onset___Bulbar'         ].isnull()==True)
    &(df_als_history['Site_of_Onset'                  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Spine'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb_and_Bulbar'].isnull()==True)
    &(df_als_history['Site_of_Onset___Other'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Other_Specify'  ].isnull()==True)
    , 'site_onset'] = 'Limb/Spinal' 
    
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Spine']==1)
    &(df_als_history['Site_of_Onset___Limb'           ].isnull()==True)
    &(df_als_history['Site_of_Onset___Bulbar'         ].isnull()==True)
    &(df_als_history['Site_of_Onset'                  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb_and_Bulbar'].isnull()==True)
    &(df_als_history['Site_of_Onset___Other'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Other_Specify'  ].isnull()==True)
    , 'site_onset'] = 'Limb/Spinal' 
    

    # ===================================================================
    # 3) Set column site_onset for "Bulbar and Limb/Spine" onset samples
    # ===================================================================
    # Update samples with Site_of_Onset = "Onset: Limb and Bulbar"
    df_als_history.loc[
        (df_als_history['Site_of_Onset']=='Onset: Limb and Bulbar')
    , 'site_onset'] = 'Bulbar and Limb/Spinal'

    # Update samples with Site_of_Onset___Limb_and_Bulbar = 1
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Limb_and_Bulbar']==1)
    &(df_als_history['Site_of_Onset___Limb'           ].isnull()==True)
    &(df_als_history['Site_of_Onset___Bulbar'         ].isnull()==True)
    &(df_als_history['Site_of_Onset'                  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Spine'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Other'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Other_Specify'  ].isnull()==True)
    , 'site_onset'] = 'Onset: Limb and Bulbar' 


    # ===================================================================
    # 4) Set column site_onset for "Other" onset samples
    # ===================================================================
    # Update samples with Site_of_Onset = "Onset: Other"
    df_als_history.loc[
        (df_als_history['Site_of_Onset']=='Onset: Other')
    , 'site_onset'] = 'Other'
    
    # Update samples with Site_of_Onset___Other = 1 OR Site_of_Onset___Other_Specify = 1
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Other'          ]==1)
    &(df_als_history['Site_of_Onset___Other_Specify'  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb_and_Bulbar'].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb'           ].isnull()==True)
    &(df_als_history['Site_of_Onset___Bulbar'         ].isnull()==True)
    &(df_als_history['Site_of_Onset'                  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Spine'          ].isnull()==True)
    , 'site_onset'] = 'Other'
    
    df_als_history.loc[
    (df_als_history['Site_of_Onset___Other_Specify'  ]==1)
    &(df_als_history['Site_of_Onset___Other'          ].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb_and_Bulbar'].isnull()==True)
    &(df_als_history['Site_of_Onset___Limb'           ].isnull()==True)
    &(df_als_history['Site_of_Onset___Bulbar'         ].isnull()==True)
    &(df_als_history['Site_of_Onset'                  ].isnull()==True)
    &(df_als_history['Site_of_Onset___Spine'          ].isnull()==True)
    , 'site_onset'] = 'Other'


    # ===================================================================
    # Drop irrelevant columns   

    irrelevant_cols = [
        'Site_of_Onset___Bulbar', 
        'Site_of_Onset___Limb', 
        'Site_of_Onset___Limb_and_Bulbar', 
        'Site_of_Onset___Other', 
        'Site_of_Onset___Other_Specify', 
        'Site_of_Onset___Spine', 
        'Disease_Duration', 
        'Symptom',
        'Symptom_Other_Specify', 
        'Location', 
        'Location_Other_Specify', 
        'Site_of_Onset',
        'Subject_ALS_History_Delta',
    ]

    df_als_history.drop(
        columns=irrelevant_cols, 
        inplace=True,
    )

    # rename columns
    df_als_history.rename(
        columns={
            'Onset_Delta': 'Symptoms_Onset_Delta', 
            'site_onset': 'Site_Onset'
        }, 
        inplace=True
    )


    return df_als_history

   # ===================================================================
   # ===================================================================
   # ===================================================================
   # ===================================================================
   # ===================================================================
   # ===================================================================


# ===================================================================
# ===================================================================
# ===================================================================
# ===================================================================
# ===================================================================
# ===================================================================

'''
Read each longitudinal CSV file and calculate the quantity of 
measurements for each patient for the following biomarkers:
 - ALSFRS
 - FVC
 - SVC
'''

def get_measurements(df, data_dir):
    # list of CSV files
    csv_names = [
        'PROACT_ALSFRS', 
        'PROACT_FVC', 
        'PROACT_SVC', 
    #     'PROACT_VITALSIGNS', 
    #     'PROACT_LABS',
    #     'PROACT_HANDGRIPSTRENGTH', 
    #     'PROACT_MUSCLESTRENGTH',
    # *** CHECK OTHER CSV FILES
    ] 

    df_measurements = df.copy()

    col_to_count = 'subject_id'

    cols_measurement = []

    for csv_name in csv_names:

        renamed_col = f"Qty_Measurements_{csv_name.replace('PROACT_', '')}"
        cols_measurement.append(renamed_col)
        
        #set the name of CSV file
        data_file = f'{data_dir}/{csv_name}.csv'

        #read data and show some info
        df_grouped = pd.read_csv(data_file, delimiter=',')
        #df_grouped.head()

        df_grouped = pd.DataFrame(df_grouped.groupby(by='subject_id')[col_to_count].count() )
        df_grouped.rename(columns={col_to_count: renamed_col}, inplace=True)
        df_grouped.reset_index(inplace=True)


        df_join = utils.join_datasets_by_key(df_main=df_measurements, df_to_join=df_grouped, key_name='subject_id', how='left')

        # fill NaN values with 0
        df_join.fillna(0, inplace=True)

        df_measurements = df_join
        
    df_measurements['Qty_Measurements'] = df_measurements[cols_measurement].sum(axis=1)   
        
    return df_measurements


'''
Remove those patients with no measurements registered
'''    
def remove_patients_with_no_measurements(df_measurements):
    to_delete = df_measurements.loc[(df_measurements.Qty_Measurements == 0)].copy()
    
    df_measurements = utils.remove_rows(
        df=df_measurements,
        to_delete=to_delete,
    )

    return df_measurements


'''
Remove those patients with no ALSFRS measurements registered
Note: ALSFRS data is essential for this study
'''
def remove_patients_with_no_alsfrs(df_measurements):
    to_delete = df_to_save.loc[(df_to_save.Qty_Measurements_ALSFRS == 0)].copy()
    df_to_save = utils.remove_rows(
        df=df_to_save,
        to_delete=to_delete,
    )



'''
Group deceased patients into Short and Non-Short survival groups:
  - Short     <= 24 months     
  - Non-Short >= 25 months     
'''
def group_patients_into_short_and_non_short(df):

    df_grouped = df.copy() 

    df_grouped['Survival_Group'] = np.NaN
    
    # mark all patients deceased within 24 months from symptom onset as Short
    to_update = df_grouped.loc[
        (df_grouped.Event_Dead==True)
    &(df_grouped.Event_Dead_Time_from_Onset<=24)
    ].copy()
    df_grouped.loc[to_update.index, 'Survival_Group'] = 'Short'
    
    # mark all patients deceased beyond 24 months from symptom onset as Non-Short
    to_update = df_grouped.loc[
        (df_grouped.Event_Dead==True)
    &(df_grouped.Event_Dead_Time_from_Onset>=25)
    ].copy()
    df_grouped.loc[to_update.index, 'Survival_Group'] = 'Non-Short'    
    
    # mark all patients not deceased beyond 24 months from symptom onset as Non-Short
    to_update = df_grouped.loc[
        (df_grouped.Event_Dead==False)
    &(df_grouped.Event_Dead_Time_from_Onset>=25)
    ].copy()
    df_grouped.loc[to_update.index, 'Survival_Group'] = 'Non-Short'

    # mark all patients not deceased beyond 24 months from symptom onset as Non-Short
    to_update = df_grouped.loc[
        (df_grouped.Event_Dead==False)
    &(df_grouped.Event_Dead_Time_from_Onset<=24)
    ].copy()
    df_grouped.loc[to_update.index, 'Survival_Group'] = 'Censored'

    return df_grouped



# create dummies vars
def create_dummies(df, column, drop_first=False, dummy_na=False, prefix=None, prefix_sep='', dtype=float):
    dummies = pd.get_dummies(df[column],
                             drop_first=drop_first,
                             dummy_na=dummy_na,
                             prefix=prefix,
                             prefix_sep=prefix_sep,
                             dtype=dtype)
    return df.join(dummies)



def perform_data_codification(df_patients):

    df_coded = df_patients.copy()

    # ===================================================================
    # codify SEX
    df_coded.Sex = df_coded.Sex.map( {'Male': 1, 'Female': 0}) 
    df_coded.rename(columns={'Sex': 'Sex_Male'}, inplace=True) 

    # ===================================================================
    # codify Site_Onset
    df_coded.Site_Onset = df_coded.Site_Onset.map( {'Limb/Spinal': 1, 'Bulbar': 0}) 

    # ===================================================================
    # codify Age_at_Onset
    df_coded.Age_at_Onset = df_coded.Age_at_Onset.map( {
        '0-39' : 0, 
        '40-49': 1, 
        '50-59': 2, 
        '60-69': 3, 
        '70+'  : 4,
    }) 

    # ===================================================================
    # codify Riluzole
    df_coded.Riluzole = df_coded.Riluzole.map( {True: 1, False: 0}) 

    # ===================================================================
    # codify Diagnosis_Delay
    df_coded.Diagnosis_Delay = df_coded.Diagnosis_Delay.astype(int)

    to_update = df_coded.loc[(df_coded.Diagnosis_Delay <= 8)]
    df_coded.loc[to_update.index, 'Diagnosis_Delay'] = 0

    to_update = df_coded.loc[(
        (df_coded.Diagnosis_Delay > 8)
        &(df_coded.Diagnosis_Delay <= 18)
    )]
    df_coded.loc[to_update.index, 'Diagnosis_Delay'] = 1

    to_update = df_coded.loc[(df_coded.Diagnosis_Delay > 18)]
    df_coded.loc[to_update.index, 'Diagnosis_Delay'] = 2

    # ===================================================================
    # codify FVC
    df_coded.FVC_at_Diagnosis = df_coded.FVC_at_Diagnosis.astype(int)
    # Abnormal
    to_update = df_coded.loc[(df_coded.FVC_at_Diagnosis < 80)]
    df_coded.loc[to_update.index, 'FVC_at_Diagnosis'] = 1
    # Normal
    to_update = df_coded.loc[(df_coded.FVC_at_Diagnosis  >= 80)]
    df_coded.loc[to_update.index, 'FVC_at_Diagnosis'] = 0 

    # ===================================================================
    # codify BMI_at_Diagnosis
    # underweight    : < 18.5
    to_update = df_coded.loc[(df_coded.BMI_at_Diagnosis < 18.5 )]
    df_coded.loc[to_update.index, 'BMI_at_Diagnosis'] = 0

    # normal weight  : >= 18.5 & < 25.0 
    to_update = df_coded.loc[(df_coded.BMI_at_Diagnosis >= 18.5) & (df_coded.BMI_at_Diagnosis < 25.0 )]
    df_coded.loc[to_update.index, 'BMI_at_Diagnosis'] = 1

    # overweight     : >= 25.0 & < 30.0 
    to_update = df_coded.loc[(df_coded.BMI_at_Diagnosis >= 25.0) & (df_coded.BMI_at_Diagnosis < 30.0 )]
    df_coded.loc[to_update.index, 'BMI_at_Diagnosis'] = 2

    # grade 1 obesity: >= 30.0 
    to_update = df_coded.loc[(df_coded.BMI_at_Diagnosis >= 30.0)]
    df_coded.loc[to_update.index, 'BMI_at_Diagnosis'] = 3

    # convert to int
    df_coded.BMI_at_Diagnosis = df_coded.BMI_at_Diagnosis.astype(int)


    # ===================================================================
    # codify Survival_Group
    df_coded.Survival_Group = df_coded.Survival_Group.map( {'Short': 1, 'Non-Short': 0}) 


    # ===================================================================
    # codify ALSFRS Questions slopes
    cols_slope = [
        'Q1_Speech_slope_at_Diagnosis',
        'Q2_Salivation_slope_at_Diagnosis',
        'Q3_Swallowing_slope_at_Diagnosis',
        'Q4_Handwriting_slope_at_Diagnosis',
        'Q5_Cutting_slope_at_Diagnosis',
        'Q6_Dressing_and_Hygiene_slope_at_Diagnosis',
        'Q7_Turning_in_Bed_slope_at_Diagnosis',
        'Q8_Walking_slope_at_Diagnosis',
        'Q9_Climbing_Stairs_slope_at_Diagnosis',
        'Q10_Respiratory_slope_at_Diagnosis',
    ]

    for col_slope in cols_slope:

        slow = df_coded.loc[
            (df_coded[col_slope]<0.05)
        ].copy()

        average = df_coded.loc[
            (df_coded[col_slope]>=0.05)
            &(df_coded[col_slope]<0.14)
        ].copy()

        rapid = df_coded.loc[
            (df_coded[col_slope]>=0.14)
        ].copy()

        df_coded.loc[slow.index, f'{col_slope}'] = 0 #Slow

        df_coded.loc[average.index, f'{col_slope}'] = 1 #Average

        df_coded.loc[rapid.index, f'{col_slope}'] = 2 #Rapid

        #convert to int
        df_coded[col_slope] = df_coded[col_slope].astype(int)



    # ===================================================================
    # convert columns to integer for these columns below
    cols_integer = [
        'Qty_Regions_Involved_at_Diagnosis',	
        'Region_Involved_Bulbar_at_Diagnosis',	
        'Region_Involved_Upper_Limb_at_Diagnosis',	
        'Region_Involved_Lower_Limb_at_Diagnosis',	
        'Region_Involved_Respiratory_at_Diagnosis',	
        'Patient_with_Gastrostomy_at_Diagnosis',	
    ]

    for col in cols_integer:
        df_coded[col] = df_coded[col].astype(int)



    #
    return df_coded



#
def perform_data_scaling(df_coded):

    df_scaled = df_coded.copy()

    scaler = MinMaxScaler()
    for col in df_scaled.columns.values:
        df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])

    df_scaled = df_scaled.round(2)   

    return df_scaled 
