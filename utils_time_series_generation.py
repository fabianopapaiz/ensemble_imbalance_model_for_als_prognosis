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

def generate_biomarkers_at_diagnosis_alsfrs(df_patients, dir_time_series_alsfrs):
    
    # 1) read data for each ALSFRS QUESTION (e.g., SLOPES)
    dfs_ts_ALSFRS_q_slopes = []

    alsfrs_question_columns = [
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

    for q_col in alsfrs_question_columns:
        # read SLOPES for each question
        data_file = f'{dir_time_series_alsfrs}/ALSFRS_TimeSeries_Slope_{q_col}.csv'
        df_aux = pd.read_csv(data_file, delimiter=',')
        dfs_ts_ALSFRS_q_slopes.append([df_aux, f'{q_col}_slope'])


    # 2) Read ALSFRS Regions-Involved Time-Series data
    dfs_ts_qty_regions_involved = []

    data_file = f'{dir_time_series_alsfrs}/ALSFRS_TimeSeries_Qty_Regions_Involved.csv'
    df_aux = pd.read_csv(data_file, delimiter=',')
    dfs_ts_qty_regions_involved.append([df_aux, 'Qty_Regions_Involved'])

    # For each REGION, read QUANTITY OF REGIONS INVOLVED
    dfs_ts_regions_involved = []

    alsfrs_regions_involved = [
        'Region_Involved_Bulbar',
        'Region_Involved_Upper_Limb',
        'Region_Involved_Lower_Limb',
        'Region_Involved_Respiratory',
    ]

    for reg_inv_col in alsfrs_regions_involved:
        # read SLOPES for each question
        data_file = f'{dir_time_series_alsfrs}/ALSFRS_TimeSeries_{reg_inv_col}.csv'
        df_aux = pd.read_csv(data_file, delimiter=',')
        dfs_ts_regions_involved.append([df_aux, f'{reg_inv_col}'])



    # 3) Read ALSFRS Patient-With-Gastrostomy Time-Series data
    dfs_ts_patient_with_gastrostomy = []
    data_file = f'{dir_time_series_alsfrs}/ALSFRS_TimeSeries_Patient_with_Gastrostomy.csv'
    df_aux = pd.read_csv(data_file, delimiter=',')
    dfs_ts_patient_with_gastrostomy.append([df_aux, 'Patient_with_Gastrostomy'])                                        



    # 4) Get time-series values at diagnosis
    dfs_ts = []
    # add ALSFRS QUESTIONS slopes data_frames
    for df in dfs_ts_ALSFRS_q_slopes:
        dfs_ts.append(df)

    # add ALSFRS QUANTITY OF REGIONS INVOLVED data_frames
    for df in dfs_ts_qty_regions_involved:
        dfs_ts.append(df)
        
    # add ALSFRS QUANTITY FOR EACH REGION INVOLVED data_frames
    for df in dfs_ts_regions_involved:
        dfs_ts.append(df)
        
    # add ALSFRS PATIENT WITH GASTROSTOMY data_frames
    for df in dfs_ts_patient_with_gastrostomy:
        dfs_ts.append(df)

    # create a copy of patients dataFrame
    df_biomarkers_at_time_t = df_patients.copy()

    #get diagnosis delay in months for each patient
    times_at_diagnosis = df_biomarkers_at_time_t['Diagnosis_Delay']


    df_return = df_biomarkers_at_time_t.copy()

    for df_ts, biomarker_name in dfs_ts:
        # get biomarkers values at diagnosis
        df_return = get_biomarker_value_at_time_t(
            df_patients=df_return, 
            df_ts=df_ts, 
            time_t=times_at_diagnosis, 
            biomarker_name=biomarker_name, 
            time_t_description='Diagnosis',
            use_these_times_t=times_at_diagnosis,
        )

    return df_return    



def generate_biomarkers_at_diagnosis(df_patients, dir_time_series_data):
        
    # read FVC TimeSeries
    data_file = f'{dir_time_series_data}/FVC_TimeSeries.csv'
    df_ts_fvc = pd.read_csv(data_file, delimiter=',')

    # read SVC TimeSeries
    data_file = f'{dir_time_series_data}/SVC_TimeSeries.csv'
    df_ts_svc = pd.read_csv(data_file, delimiter=',')

    # read BMI TimeSeries
    data_file = f'{dir_time_series_data}/BMI_TimeSeries.csv'
    df_ts_bmi = pd.read_csv(data_file, delimiter=',')


    dfs_ts = [
        [df_ts_fvc, 'FVC'],
        [df_ts_bmi, 'BMI'],
        [df_ts_svc, 'SVC'],
    ]


    # create a copy of patients dataFrame
    df_biomarkers_at_time_t = df_patients.copy()

    #get diagnosis delay in months for each patient
    times_at_diagnosis = df_biomarkers_at_time_t['Diagnosis_Delay']


    df_return = df_biomarkers_at_time_t.copy()

    for df_ts, biomarker_name in dfs_ts:
        # get biomarkers values at diagnosis
        df_return = get_biomarker_value_at_time_t(
            df_patients=df_return, 
            df_ts=df_ts, 
            time_t=times_at_diagnosis, 
            biomarker_name=biomarker_name, 
            time_t_description='Diagnosis',
            use_these_times_t=times_at_diagnosis,
        )

    #
    return df_return    



def get_biomarker_value_at_time_t(df_patients, df_ts, time_t, biomarker_name, 
            time_t_description=None, use_this_column_as_time_t=None, use_these_times_t=None):
    
    # copy dataFrame
    df_joined = df_patients.copy()

    # Param "time_t_description" must be informed when using "use_this_column_as_time_t"
    if (use_this_column_as_time_t is not None) & (time_t_description is None):
        raise Exception('Param "time_t_description" must be informed when using "use_this_column_as_time_t"')
    
    # if not informed 'time_t_description' use 'time_t' as description
    if (time_t_description is None) & (use_this_column_as_time_t is None):
        time_t_description = f'{time_t}'

    # create column to store the biomarker value at time t
    col_created = f'{biomarker_name}_at_{time_t_description}'

    # create the column with NaN
    df_joined[col_created] = np.NaN

    # counters for success and errors
    count_err = 0
    count_upd = 0


    for index, row in df_joined.iterrows():
        # get the subject ID
        subject_id = row['subject_id']

        # locate the patient in Time-Series dataFrame (df_ts)
        patient = df_ts.loc[(df_ts['subject_id'] == subject_id)].copy()

        # if found the patient in the Time-Series dataFrame
        if patient.shape[0] > 0:

            # if was to use specific column as time-t (e.g., "Diagnosis_Delay")
            if use_this_column_as_time_t is not None:
                time_t = row[use_this_column_as_time_t]

            # if was to use specific Series to time-t
            if use_these_times_t is not None:
                time_t = use_these_times_t.loc[index]

            # verify if time-t is negative
            if time_t < 0.0:
                time_t = 0.0    

            # get timet_t as string to be used as the name of time-t column
            col_time = str(time_t)

            # get the value at time t
            value_at_time_t = patient[col_time].head(1).values[0]

            try:
                # get row to be updated with value-at-time-t
                to_update = df_joined.loc[(df_joined['subject_id']==subject_id)].copy()
                # set biomarker value at time t
                df_joined.loc[to_update.index, col_created] = value_at_time_t
                count_upd += 1

                # print(f"value_at_time_t: {value_at_time_t}\n")
                # print(patient)
                # print(to_update)
                # break
            except Exception as err:
                print(f"ERROR: subject_id   : {subject_id}")
                print(f"       Time t       : {time_t}")
                print(f"       Row at time t: {value_at_time_t}")
                print(f"                      {type(value_at_time_t)}")
                print()    
                print(f"   Error Type: {type(err)}") # the exception instance
                print(f"         Msg : {err}") # __str__ allows args to be printed directly
                print(f"         Msg : {err.args}") # arguments stored in .args
                utils.print_string_with_separators('DF_JOINED')
                print(df_joined.loc[(df_joined['subject_id']==subject_id)])
                utils.print_string_with_separators('PATIENT')
                print(patient)

                count_err += 1
                pass
            
    
    # return            
    return df_joined




def generate_time_series_bmi(df_temporal, dir_dest):
    # Get values by month up to 72 months (i.e., 6 years)
    n_years = 10
    threshold = 12 * n_years # "n" years
    months = np.linspace(0, threshold, threshold+1, dtype=float) #[1.0, 2.0, 3.0,..., 72.0]


    baselines = ['Symptoms_Onset']

    for baseline in baselines:

        col_baseline = f'Delta_from_{baseline}'
        
        # copy data ordering by col_baseline
        df_copy = df_temporal.sort_values(by=['subject_id', col_baseline]).copy()
        
        # drop rows with NaN in col_baseline
        df_copy.dropna(
            subset=[
                col_baseline, 
                'BMI', 
            ], 
            inplace=True
        )

        # filter rows by threshold
        df_pivot = df_copy.copy()

        # get only the names of the Values columns 
        cols_to_pivot = df_pivot.columns[2:]

        # create pivot by column Result
        df_aux = df_pivot.pivot_table(
            index='subject_id', 
            columns=col_baseline, 
            values='BMI',
            aggfunc=np.max, # get max value in that month (can exist 2+ measurements for a same month)
        )
        
        # reset index
        df_aux.reset_index(inplace=True)

        # get the month-columns existing in the pivot-table
        cols_months = df_aux.columns[1:]

        # check if all months columns were created [1-72]
        for month in months:
            # if month not present in the columns
            if month not in cols_months:
                # Creating column for this month and set its values to NaN
                # PS: "int(month)" is used to keep columns ordered by month number
                df_aux.insert(int(month), month, np.NaN)

        # code to ensure the order of the columns
        cols_months_ordered = list(sorted(months))
        cols_months_ordered.insert(0, 'subject_id')
        df_aux = df_aux[cols_months_ordered]
                
        # set month-0 = NaN 
        df_aux[0.0] = np.NaN
 
        # read file saved to fill NaN values using interpolation
        df_fill_nan_using_interpolation = df_aux
        # get columns ignoring 'subject_id'
        cols_months = df_fill_nan_using_interpolation.columns[1:]

        # print (cols_months)
        df_aux = df_fill_nan_using_interpolation[cols_months].interpolate(
            method='linear', 
            limit_direction='both',
            limit=1000, 
            axis=1, 
            inplace=False,
        ).copy()

        # round Values using 0 decimal places
        df_aux[cols_months] = np.round(df_aux[cols_months], 0)

        # get subject_id column
        df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]

        # drop rows with NaN values (where there is no Value registered)
        df_fill_nan_using_interpolation.dropna(inplace=True)

        # save data again for each Value column with interpolation
        print(f'BMI')
        csv_file = f'{dir_dest}/TimeSeries/BMI_TimeSeries.csv'
        utils.save_to_csv(df=df_fill_nan_using_interpolation, csv_file=csv_file)

        # just for further tests
        df_aux = df_fill_nan_using_interpolation.copy()
        
        print()





def generate_time_series_svc(df_temporal, dir_dest):
    # Get values by month up to 72 months (i.e., 6 years)
    n_years = 10
    threshold = 12 * n_years # "n" years
    months = np.linspace(0, threshold, threshold+1, dtype=float) #[1.0, 2.0, 3.0,..., 72.0]


    baselines = ['Symptoms_Onset']

    for baseline in baselines:

        col_baseline = f'Delta_from_{baseline}'
        
        # copy data ordering by col_baseline
        df_copy = df_temporal.sort_values(by=['subject_id', col_baseline]).copy()
        
        # drop rows with NaN in col_baseline
        df_copy.dropna(
            subset=[
                col_baseline, 
                'SVC_Perc_of_Normal', 
            ], 
            inplace=True
        )

        # filter rows by threshold
        df_pivot = df_copy.copy()

        # get only the names of the Values columns 
        cols_to_pivot = df_pivot.columns[2:]

        # create pivot by column Result
        df_aux = df_pivot.pivot_table(
            index='subject_id', 
            columns=col_baseline, 
            values='SVC_Perc_of_Normal',
            aggfunc=np.max, # get max value in that month (can exist 2+ measurements for a same month)
        )
        
        # reset index
        df_aux.reset_index(inplace=True)

        # get the month-columns existing in the pivot-table
        cols_months = df_aux.columns[1:]

        # check if all months columns were created [1-72]
        for month in months:
            # if month not present in the columns
            if month not in cols_months:
                # Creating column for this month and set its values to NaN
                # PS: "int(month)" is used to keep columns ordered by month number
                df_aux.insert(int(month), month, np.NaN)

        # code to ensure the order of the columns
        cols_months_ordered = list(sorted(months))
        cols_months_ordered.insert(0, 'subject_id')
        df_aux = df_aux[cols_months_ordered]
                
        # set month-0 = NaN 
        df_aux[0.0] = np.NaN
 
        # read file saved to fill NaN values using interpolation
        df_fill_nan_using_interpolation = df_aux
        # get columns ignoring 'subject_id'
        cols_months = df_fill_nan_using_interpolation.columns[1:]

        # print (cols_months)
        df_aux = df_fill_nan_using_interpolation[cols_months].interpolate(
            method='linear', 
            limit_direction='both',
            limit=1000, 
            axis=1, 
            inplace=False,
        ).copy()

        # round Values using 0 decimal places
        df_aux[cols_months] = np.round(df_aux[cols_months], 0)

        # get subject_id column
        df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]

        # drop rows with NaN values (where there is no Value registered)
        df_fill_nan_using_interpolation.dropna(inplace=True)

        # save data again for each Value column with interpolation
        print(f'SVC')
        csv_file = f'{dir_dest}/TimeSeries/SVC_TimeSeries.csv'
        utils.save_to_csv(df=df_fill_nan_using_interpolation, csv_file=csv_file)

        # just for further tests
        df_aux = df_fill_nan_using_interpolation.copy()
        
        print()



def generate_time_series_fvc(df_temporal, dir_dest):
    # Get values by month up to 72 months (i.e., 6 years)
    n_years = 10
    threshold = 12 * n_years # "n" years
    months = np.linspace(0, threshold, threshold+1, dtype=float) #[1.0, 2.0, 3.0,..., 72.0]


    baselines = ['Symptoms_Onset']

    for baseline in baselines:

        col_baseline = f'Delta_from_{baseline}'
        
        # copy data ordering by col_baseline
        df_copy = df_temporal.sort_values(by=['subject_id', col_baseline]).copy()
        
        # drop rows with NaN in col_baseline
        df_copy.dropna(
            subset=[
                col_baseline, 
                'FVC_Perc_of_Normal', 
            ], 
            inplace=True
        )

        # filter rows by threshold
        df_pivot = df_copy.copy()

        # get only the names of the Values columns 
        cols_to_pivot = df_pivot.columns[2:]

        # create pivot by column Result
        df_aux = df_pivot.pivot_table(
            index='subject_id', 
            columns=col_baseline, 
            values='FVC_Perc_of_Normal',
            aggfunc=np.max, # get max value in that month (can exist 2+ measurements for a same month)
        )
        
        # reset index
        df_aux.reset_index(inplace=True)

        # get the month-columns existing in the pivot-table
        cols_months = df_aux.columns[1:]

        # check if all months columns were created [1-72]
        for month in months:
            # if month not present in the columns
            if month not in cols_months:
                # Creating column for this month and set its values to NaN
                # PS: "int(month)" is used to keep columns ordered by month number
                df_aux.insert(int(month), month, np.NaN)

        # code to ensure the order of the columns
        cols_months_ordered = list(sorted(months))
        cols_months_ordered.insert(0, 'subject_id')
        df_aux = df_aux[cols_months_ordered]
                
        # set month-0 = NaN 
        df_aux[0.0] = np.NaN
 
        # read file saved to fill NaN values using interpolation
        df_fill_nan_using_interpolation = df_aux
        # get columns ignoring 'subject_id'
        cols_months = df_fill_nan_using_interpolation.columns[1:]

        # print (cols_months)
        df_aux = df_fill_nan_using_interpolation[cols_months].interpolate(
            method='linear', 
            limit_direction='both',
            limit=1000, 
            axis=1, 
            inplace=False,
        ).copy()

        # round Values using 0 decimal places
        df_aux[cols_months] = np.round(df_aux[cols_months], 0)

        # get subject_id column
        df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]

        # drop rows with NaN values (where there is no Value registered)
        df_fill_nan_using_interpolation.dropna(inplace=True)

        # save data again for each Value column with interpolation
        print(f'FVC')
        csv_file = f'{dir_dest}/TimeSeries/FVC_TimeSeries.csv'
        utils.save_to_csv(df=df_fill_nan_using_interpolation, csv_file=csv_file)

        # just for further tests
        df_aux = df_fill_nan_using_interpolation.copy()
        
        print()




def generate_time_series_alsfrs(df_temporal, dir_dest):

    # Get values by month up to 72 months (i.e., 6 years)
    n_years = 10
    threshold = 12 * n_years # "n" years
    months = np.linspace(0, threshold, threshold+1, dtype=float) #[1.0, 2.0, 3.0,..., 72.0]


    baselines = [
        'Symptoms_Onset'
    ]

    columns_questions_ALSFRS = [
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

    columns_not_to_interpolate = [
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
        #
        'Region_Involved_Bulbar',
        'Region_Involved_Upper_Limb',
        'Region_Involved_Lower_Limb',
        'Region_Involved_Respiratory',
        'Qty_Regions_Involved',
        # boolean columns
        'Patient_with_Gastrostomy',

    ]
    columns_to_interpolate = [
        #
        'Slope_from_Onset_Q1_Speech',
        'Slope_from_Onset_Q2_Salivation',
        'Slope_from_Onset_Q3_Swallowing',
        'Slope_from_Onset_Q4_Handwriting',
        'Slope_from_Onset_Q5_Cutting',
        'Slope_from_Onset_Q6_Dressing_and_Hygiene',
        'Slope_from_Onset_Q7_Turning_in_Bed',
        'Slope_from_Onset_Q8_Walking',
        'Slope_from_Onset_Q9_Climbing_Stairs',
        'Slope_from_Onset_Q10_Respiratory',
    ]

    columns = columns_not_to_interpolate + columns_to_interpolate

    # dir_dest = os.path.abspath('../03_preprocessed_data/')


    for baseline in baselines:
        
        for column in columns:

            #dont process the score, only the slopes
            if column in columns_questions_ALSFRS:
                continue

            col_baseline = f'Delta_from_{baseline}'

            # copy data ordering by col_baseline
            df_copy = df_temporal.sort_values(by=['subject_id', col_baseline]).copy()


            # convert boolean values to 0/1 for Boolean cloumns
            if (column == 'Patient_with_Gastrostomy') | (column.startswith('Region_Involved_')):
                df_copy[column].replace({True: 1, False: 0}, inplace=True)
                
            # drop rows with NaN in "col_baseline" and "column"
            df_copy.dropna(
                subset=[
                    col_baseline, 
                    column,
                ], 
                inplace=True
            )

            # filter rows by threshold
            df_pivot = df_copy.copy()

            # get only the names of the Values columns 
            cols_to_pivot = df_pivot.columns[2:]

            # create pivot by column Result
            df_aux = df_pivot.pivot_table(
                index='subject_id', 
                columns=col_baseline, 
                values=column,
                aggfunc=np.max, # get max value in that month (can exist 2+ measurements for a same month)
            )

            # reset index
            df_aux.reset_index(inplace=True)

            # get the month-columns existing in the pivot-table
            cols_months = df_aux.columns[1:]

            # check if all months columns were created [1-72]
            for month in months:
                # if month not present in the columns
                if month not in cols_months:
                    # Creating column for this month and set its values to NaN
                    # PS: "int(month)" is used to keep columns ordered by month number
                    df_aux.insert(int(month), month, np.NaN)

            # code to ensure the order of the columns
            cols_months_ordered = list(sorted(months))
            cols_months_ordered.insert(0, 'subject_id')
            df_aux = df_aux[cols_months_ordered]
            
            
            round_decimal_places = 2
            
            # if column is a slope Total-Score from onset, set month-0 = 0.0 (none decline or increase)
            if (column == 'Slope_from_Onset_Total_Score') | ('Slope_from_Onset_Q' in column):
                df_aux[0.0] = 0.0
            # if column is a ALSFRS question, set month-0 = 4 (Max score for each ALSFRS question)
            elif (column in columns_questions_ALSFRS):
                df_aux[0.0] = 4.0
                round_decimal_places = 0
            # if column is to do not interpolate, set round_decimal_places = 0
            elif (column in columns_not_to_interpolate):
                round_decimal_places = 0

            
            col_name = column.replace('_from_Onset', '')

            # read file saved to fill NaN values using interpolation
            df_fill_nan_using_interpolation = df_aux

            # get columns ignoring 'subject_id'
            cols_months = df_fill_nan_using_interpolation.columns[1:]


            # perform Missing Imputation using interpolation
            df_aux = df_fill_nan_using_interpolation[cols_months].interpolate(
                method='linear', 
                limit_direction='both',
                limit=1000, 
                axis=1, 
                inplace=False,
            ).copy()
            
            # round Values using "round_decimal_places" variable
            df_aux[cols_months] = np.round(df_aux[cols_months], round_decimal_places)
            
            # get subject_id column
            df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]

            # drop rows with NaN values (where there is no Value registered)
            df_fill_nan_using_interpolation.dropna(inplace=True)

            # save data again for each Value column with interpolation
            print(f'{col_name}')
            csv_file = f'{dir_dest}/TimeSeries/ALSFRS/ALSFRS_TimeSeries_{col_name}.csv'
            utils.save_to_csv(df=df_fill_nan_using_interpolation, csv_file=csv_file)

            # just for further tests
            df_aux = df_fill_nan_using_interpolation.copy()

            print()

