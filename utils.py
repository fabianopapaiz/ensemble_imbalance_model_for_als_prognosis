import math
import os
import string

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from datetime import datetime
from datetime import timedelta
from dateutil import relativedelta




CLASS_COLUMN = 'Survival_Group'

CODE_VALUES_1 = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
}

# read a CSV file using Pandas
def read_csv(csv_file):
    return pd.read_csv(csv_file, delimiter=',')


def get_relevant_features_names():
    return [
        'Diagnosis_Delay',
        'Age_Range_at_Onset',
        'Riluzole',
        'Sex_Male',
        'Site_Onset',
        'FVC',
        'BMI',
        'Q1_Speech_slope',
        'Q2_Salivation_slope',
        'Q3_Swallowing_slope',
        'Q4_Handwriting_slope',
        'Q5_Cutting_slope',
        'Q6_Dressing_and_Hygiene_slope',
        'Q7_Turning_in_Bed_slope',
        'Q8_Walking_slope',
        'Q9_Climbing_Stairs_slope',
        'Q10_Respiratory_slope',
        'Qty_Regions_Involved',
        'Region_Involved_Bulbar',
        'Region_Involved_Upper_Limb',
        'Region_Involved_Lower_Limb',
        'Region_Involved_Respiratory',
        'Patient_with_Gastrostomy',
    ]


def get_inputs_and_output_variables(df):
    
    input_cols = list(df.columns)

    input_cols.remove(CLASS_COLUMN)

    X    = df[input_cols].copy() 
    y    = pd.DataFrame(df[CLASS_COLUMN].values, columns=[CLASS_COLUMN])

    return X, y


# get "n" first colors for a given palette
def get_n_colors(n, palette='colorblind'):
    return sns.color_palette(palette, n)


def get_train_and_validation_data(dir_data=None, scaled=False, coded=False, use_diagnosis_delay=True):

    if dir_data is None:
        dir_data = os.path.abspath('training_validation_data/')

    # file suffix
    suffix = ''
    if scaled:
        suffix = '_scaled'
    elif coded:    
        suffix = '_coded'

    # Training set
    df_train = read_csv(f'{dir_data}/training_data{suffix}.csv')
    X_train, y_train = get_inputs_and_output_variables(df_train)

    # Validation set
    df_validation = read_csv(f'{dir_data}/validation_data{suffix}.csv')
    X_valid, y_valid = get_inputs_and_output_variables(df_validation)


    # drop Diagnosis_Delay column, if needed
    if not use_diagnosis_delay:
        X_train.drop(columns=['Diagnosis_Delay'], inplace=True)
        X_valid.drop(columns=['Diagnosis_Delay'], inplace=True)


    # X_all = pd.concat([X_train, X_valid])
    # y_all = pd.concat([y_train, y_valid])

    return X_train, y_train, X_valid, y_valid #, X_all, y_all


def get_decoded_data(df, add_coded_value=False):
    #, , print_code_value=False, clean_feat_values=False):
    
    df_decoded = df.copy()
    
    # verify if the values are scaled by checking the Diagnosis_Delay column 
    scaled = (sorted(df.Diagnosis_Delay.unique()) == [0.0, 0.5, 1.0])


    # check if need to un-dummy the Sex_Male column
    undo_dummy_sex_male = ('Sex_Male' in df_decoded.columns)
    if undo_dummy_sex_male:

        series_sex = undo_dummy_variable(
            series=df_decoded.Sex_Male,
            map_values={1: 'Male', 0: 'Female'},
            new_col_name='Sex'
        )

        df_decoded.Sex_Male = series_sex
        df_decoded.rename(columns={'Sex_Male': 'Sex'}, inplace=True)

    for col in df_decoded.columns:
        series = df_decoded[col]
        df_decoded[col] = get_coded_to_string(
            series=series, 
            scaled=scaled, 
            add_coded_value=add_coded_value
        )

    #     if col.endswith('_Coded'):
    #         df_decoded.rename(columns={col: col.replace('_Coded', '')}, inplace=True)


    # if clean_feat_values:
    #     for c in df_decoded.columns:     
    #         df_decoded[c] = df_decoded[c].apply(lambda x: clean_feature_values(x))

    #
    return df_decoded    




def get_decoded_data_frame(df, scaled=False, undo_dummy_sex_male=False, print_code_value=False, clean_feat_values=False):
    raise Exception('Use the get_decoded_data() method instead this method.')

    # df_decoded = df.copy()

    # for col in df_decoded.columns:
    #     series = df_decoded[col]
    #     df_decoded[col] = get_coded_to_string(series=series, scaled=scaled, add_coded_value=print_code_value)

    #     if col.endswith('_Coded'):
    #         df_decoded.rename(columns={col: col.replace('_Coded', '')}, inplace=True)


    # if undo_dummy_sex_male:
    #     series_sex = undo_dummy_variable(
    #         series=df_decoded.Sex_Male,
    #         map_values={'Yes': 'Male', 'No': 'Female'},
    #         new_col_name='Sex'
    #     )
    #     df_decoded.Sex_Male = series_sex
    #     df_decoded.rename(columns={'Sex_Male': 'Sex'}, inplace=True)


    # if clean_feat_values:
    #     for c in df_decoded.columns:     
    #         df_decoded[c] = df_decoded[c].apply(lambda x: clean_feature_values(x))


    # return df_decoded    


def get_coded_to_string(series, scaled=False, add_coded_value=False):
    col = series.name

    if col in ['Diagnosis_Delay_Coded', 'Diagnosis_Delay']:
        series = series.apply(lambda x: get_coded_diagnosis_delay_to_string(x, scaled=scaled, 
                                                                            add_coded_value=add_coded_value))

    elif col in ['Age_Range_at_Onset_Coded', 'Age_Range_at_Onset', 'Age_at_Onset']:
        series = series.apply(lambda x: get_coded_age_range_to_string(x, scaled=scaled, 
                                                                      add_coded_value=add_coded_value))

    elif col in ['Site_Onset_Coded', 'Site_Onset']:
        series = series.apply(lambda x: get_coded_site_onset_to_string(x, scaled=scaled, 
                                                                       add_coded_value=add_coded_value))

    elif col.startswith('FVC'):
        series = series.apply(lambda x: get_coded_fvc_to_string(x, 
                                                                add_coded_value=add_coded_value))

    elif col.startswith('BMI'):
        series = series.apply(lambda x: get_coded_bmi_to_string(x, scaled=scaled, 
                                                                add_coded_value=add_coded_value))

    elif col.startswith('Q') and '_slope' in col:
        series = series.apply(lambda x: get_coded_alsfrs_slope_to_string(x, scaled=scaled, 
                                                                         add_coded_value=add_coded_value))

    elif col == 'Survival_Group':
        series = series.apply(lambda x: get_coded_group_survival_to_string(x, 
                                                                           add_coded_value=add_coded_value))

    elif col in ['Qty_Regions_Involved_at_Diagnosis', 'Qty_Regions_Involved']:
        series = series.apply(lambda x: get_coded_qty_regions_involved_to_string(x, scaled=scaled)).astype(str)
    
    # # if was a integer column
    # if ('Qty_Regions_Involved' in col):
    #     series = series.astype(int)
        
    # if was a boolean column
    if (col in ['Riluzole', 'Sex_Male', 'Group_Short_Survival', 'Short_Survival']) \
    or ('Patient_with_Gastrostomy' in col) \
    or ('Region_Involved' in col)    :
        series = series.map({1.0: 'Yes', 0.0: 'No'})
        # series = series.astype(bool)

    return series    



def get_coded_qty_regions_involved_to_string(code, scaled=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = 1
        elif code == 0.33:
            text = 2
        elif code == 0.67:
            text = 3
        elif code == 1.0:
            text = 4
    else:
        text = int(code)

    return text


def get_coded_group_survival_to_string(code, add_coded_value=False):
    code = float(code)

    if code == 0.0:
        text = 'Non-Short'
    elif code == 1.0:
        text = 'Short'
    else:
        text = None    

    return text


def get_coded_fvc_to_string(code, add_coded_value=False):
    code = float(code)
    
    if code == 0.0:
        text = 'Normal'
    elif code == 1.0:
        text = 'Abnormal'
    else:
        text = None    

    return text



def set_feature_values_order(x):

    x = x.replace('Short', '(0) Short')
    x = x.replace('Average', '(1) Average')
    x = x.replace('Long', '(2) Long')

    x = x.replace('Slow', '(0) Slow')
    x = x.replace('Rapid', '(2) Rapid')

    x = x.replace('Underweight', '(0) Underweight')
    x = x.replace('Normal', '(1) Normal')
    x = x.replace('Overweight', '(2) Overweight')
    x = x.replace('Obesity', '(3) Obesity')

    return x


def get_coded_diagnosis_delay_to_string(code, scaled=False, add_coded_value=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = '(0) Short' if add_coded_value else 'Short'
        elif code == 0.5:
            text = '(1) Average' if add_coded_value else 'Average'
        elif code == 1.0:
            text = '(2) Long' if add_coded_value else 'Long'

        #code
        # code = CODE_VALUES_1[code]    
    else:
        if code == 0.0:
            text = '(0) Short' if add_coded_value else 'Short'
        elif code == 1.0:
            text = '(1) Average' if add_coded_value else 'Average'
        elif code == 2.0:
            text = '(2) Long' if add_coded_value else 'Long'

    return text


def get_coded_alsfrs_slope_to_string(code, scaled=False, add_coded_value=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            # text = 'Slow'
            text = '(0) Slow' if add_coded_value else 'Slow'
        elif code == 0.5:
            # text = 'Average'
            text = '(1) Average' if add_coded_value else 'Average'
        elif code == 1.0:
            # text = 'Rapid'
            text = '(2) Rapid' if add_coded_value else 'Rapid'
        #code
        code = CODE_VALUES_1[code]    
    else:
        if code == 0.0:
            # text = 'Slow'
            text = '(0) Slow' if add_coded_value else 'Slow'
        elif code == 1.0:
            # text = 'Average'
            text = '(1) Average' if add_coded_value else 'Average'
        elif code == 2.0:
            # text = 'Rapid'
            text = '(2) Rapid' if add_coded_value else 'Rapid'

    return text 


def get_coded_age_range_to_string(code, scaled=False, add_coded_value=False):
    code = float(code)

    text = None
    # [0.0, 0.25, 0.5, 0.75, 1.0]
    if scaled:
        if code == 0.0:
            text = '0-39'  #if not add_coded_value else '(0) 0-39'
        elif code == 0.25:
            text = '40-49' #if not add_coded_value else '(1) 40-49'
        elif code == 0.5:
            text = '50-59' #if not add_coded_value else '(2) 50-59'
        elif code == 0.75:
            text = '60-69' #if not add_coded_value else '(3) 60-69'
        elif code == 1.0:
            text = '70+'   #if not add_coded_value else '(4) 70+'
    else:
        if code == 0.0:
            text = '0-39'  #if not add_coded_value else '(0) 0-39'
        elif code == 1.0:
            text = '40-49' #if not add_coded_value else '(1) 40-49'
        elif code == 2.0:
            text = '50-59' #if not add_coded_value else '(2) 50-59'
        elif code == 3.0:
            text = '60-69' #if not add_coded_value else '(3) 60-69'
        elif code == 4.0:
            text = '70+'   #if not add_coded_value else '(4) 70+'

    return text


def get_coded_site_onset_to_string(code, scaled=False, add_coded_value=False):
    code = float(code)

    text = None    

    ## NOTE ONLY HAVE 'BULBAR' AND 'LIMB/SPINAL' IN DATASET (n=1967)

    if code == 0.0:
        text = 'Bulbar'
    elif code == 1.0:
        text = 'Limb/Spinal'
    elif code == 2.0:
        text = 'Other'

    return text


def get_coded_bmi_to_string(code, scaled=False, add_coded_value=False):
    code = float(code)

    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            # text = '(0) Underweight'
            text = '(0) Underweight' if add_coded_value else 'Underweight'
        elif code == 0.33:
            # text = '(1) Normal'
            text = '(1) Normal' if add_coded_value else 'Normal'
        elif code == 0.67:
            # text = '(2) Overweight'
            text = '(2) Overweight' if add_coded_value else 'Overweight'
        elif code == 1.0:
            # text = '(3) Obesity'
            text = '(3) Obesity' if add_coded_value else 'Obesity'
    else:
        if code == 0.0:
            # text = '(0) Underweight'
            text = '(0) Underweight' if add_coded_value else 'Underweight'
        elif code == 1.0:
            # text = '(1) Normal'
            text = '(1) Normal' if add_coded_value else 'Normal'
        elif code == 2.0:
            # text = '(2) Overweight'
            text = '(2) Overweight' if add_coded_value else 'Overweight'
        elif code == 3.0:
            # text = '(3) Obesity'
            text = '(3) Obesity' if add_coded_value else 'Obesity'
  
    return text



def undo_dummy_variable(series, map_values, new_col_name):
    
    series_aux = series.copy()
    
    values = series_aux.unique()

    series_return = pd.Series(data=series_aux.values, name=new_col_name)

    series_return = series_return.map(map_values)

    return series_return


def clean_feature_values(value):
    try:
        new_value = value.split(')')[1].strip()

    except Exception as ex: 
        new_value = value
    
    return new_value


# Plot the distributions of informed variables (columns)
def plot_variables_distributions(df, columns=None, zero_and_one_as_categorical=True, bins=30, 
    figsize=[15,5], print_more_info=True, plot_one_graph_per_row=False, 
    plot_pareto_graph=True, fill_nan=False, plot_percentage_lines=True):
    
    #if dont set columns, get all coumns from dataset
    if columns is None:
        columns = list(df.columns.values)
        if 'subject_id' in columns:
            columns.remove('subject_id')

    for col in columns:
        plot_variable_distribution(df=df, column=col, zero_and_one_as_categorical=zero_and_one_as_categorical, bins=bins, figsize=figsize,
                                   print_more_info=print_more_info, plot_one_graph_per_row=plot_one_graph_per_row,
                                   fill_nan=fill_nan, plot_pareto_graph=plot_pareto_graph,
                                    plot_percentage_lines=plot_percentage_lines,
                                   )


# Plot the distribution of one given column
def plot_variable_distribution(series=None, df=None, column=None, zero_and_one_as_categorical=True, 
        bins=30, figsize=[15,5], print_more_info=True, 
        plot_one_graph_per_row=False, plot_pareto_graph=True, fill_nan=False,
        plot_percentage_lines=True):
    
    if column != 'subject_id':
        plot_histogram_and_boxplot(df=df, column=column, series=series, bins=bins,
                                figsize=figsize, 
                                print_more_info=print_more_info, plot_one_graph_per_row=plot_one_graph_per_row,
                                zero_and_one_as_categorical=zero_and_one_as_categorical,
                                fill_nan=fill_nan,
                                plot_pareto_graph=plot_pareto_graph,
                                plot_percentage_lines=plot_percentage_lines,
                                )


# Plot the distributions of informed [dataFrame + column] or [Series]
def plot_histogram_and_boxplot(df=None, column=None, series=None, zero_and_one_as_categorical=True, 
        bins=30, figsize=[15,5], orientation='horizontal',  
        print_more_info=True, plot_one_graph_per_row=False, plot_pareto_graph=True, fill_nan=False,
        plot_percentage_lines=True):

    series = get_series_from_parameters(df=df, column=column, series=series)

    total_of_rows = series.count()

    if total_of_rows == 0:
        print('NOTE: There is no data to be plotted! (ZERO rows)')
        return

    series_dtype = series.dtype

    if (plot_one_graph_per_row) and (figsize==[15,5]):
        figsize=[15, 13]


    # lead with boolean values on data_column
    if series.dtype == bool:
        series = series.map({True: 'YES', False: 'NO'})

    # check if zero-and-one columns must be treated as categorical (Yes/No)
    if zero_and_one_as_categorical:
        #get unique values
        unique_values = series.unique()
        #check if have only 2 elements
        if get_quantity_of_rows(unique_values) == 2:
            #compares with zer0-one lists with different orders
            comparison_1 = unique_values == np.array([1, 0])
            comparison_2 = unique_values == np.array([0, 1])
            #if some comparison was True, then set YES/NO values
            if comparison_1.all() or comparison_2.all():
                series = series.map({1: 'YES', 0: 'NO'})

    # print more info about distribution
    if print_more_info:
        print_variable_distribution(df=df, column=column, series=series, fill_nan=fill_nan)
    else:
        s_name = series.name  # .upper()

        print(f'==============================================================================')
        print(f'Column {s_name}  ({total_of_rows} rows)   (DataType: {series_dtype})')
        print(f'==============================================================================')


    #if variable is CONTINUOUS or INTEGER
    if series.dtype in [float, int]:
        # --------------------------------------------
        # HISTOGRAM ----------------------------------
        # --------------------------------------------
        plt.figure(figsize=figsize)
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        plt.subplot(2, 1, 1) if plot_one_graph_per_row else plt.subplot(1, 2, 1)
        #
        plt.title(f'{series.name.upper()}')
        ax = series.hist(bins=bins)

        # --------------------------------------------
        # BOXPLOT GRAPH
        # --------------------------------------------
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        plt.subplot(2, 1, 2) if plot_one_graph_per_row else plt.subplot(1, 2, 2)
        #
        ax = sns.boxplot(data=series.values, orient=orientation)

    #if variable is CATEGORICAL
    else:
        if fill_nan:
            series.fillna('**Not Informed', inplace=True)    

        #create a dataframe to plot the data
        df_distrib = pd.DataFrame(series.value_counts())
        df_distrib = df_distrib.reset_index()
        sn = series.name
        df_distrib[sn] = df_distrib[sn].astype(float)
        df_distrib.rename(columns={series.name: 'count', 'index': sn}, inplace=True)
        sum = df_distrib['count'].sum()

        df_distrib = df_distrib.sort_values(by=sn)

        df_distrib['x'] = '' + df_distrib[sn].astype(str)
        df_distrib['percentage'] = np.round(df_distrib['count'] / sum * 100, 2)
        df_distrib['percentage'] = df_distrib['percentage'].map('{:,.2f}%'.format)
        df_distrib['cum_sum'] = df_distrib['count'].cumsum()
        df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

        # --------------------------------------------
        # BAR GRAPH (WITH QUANTITY AND PERCENTAGE)
        # --------------------------------------------
        plt.figure(figsize=figsize)
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        ax = plt.subplot(2, 1, 1) if plot_one_graph_per_row else plt.subplot(1, 2, 1)
        #
        x = df_distrib['x']
        y = df_distrib['count']
        y_cum = df_distrib['cum_percentage']

        # plot bar graph
        plt.title(f"{series.name.upper()}\n")
        sns.barplot(x=x, y=y, ax=ax)
        ax.set_xlabel('')
        plt.xticks(rotation=15)

        show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

        #plot cumulative line
        if plot_percentage_lines:
            ax2 = ax.twinx()
            ax2.plot(x, y_cum, color='blue', marker='D', ms=4, alpha=0.5)
            ax2.yaxis.set_major_formatter(PercentFormatter())
            ax2.spines['right'].set_color('#a70000')
            ax2.tick_params(axis='y', colors='blue', grid_color='blue', grid_alpha=0.5, grid_linestyle='--')


        if plot_pareto_graph:
            # --------------------------------------------
            # PARETO GRAPH
            # --------------------------------------------
            # calculate the cumulative percentage
            df_distrib = df_distrib.sort_values(by='count', ascending=False)
            df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

            # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
            plt.subplot(2, 1, 2) if plot_one_graph_per_row else plt.subplot(1, 2, 2)
            #
            plt.title(f'Cumulative Percentage (Pareto Graph)\n')
            plt.xticks(rotation=15)


            # Plot Pareto graph
            x = df_distrib['x']
            y = df_distrib['count']
            ax = sns.barplot(x=x, y=y)
            # ax.set_xlabel(sn)
            ax.set_xlabel('')

            #
            show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

            # Plot lines with cumulative percentage
            if plot_percentage_lines:
                ax2 = ax.twinx()
                y_cum_perc = df_distrib['cum_percentage']
                ax2.plot(x, y_cum_perc, color='#a70000', marker='D', ms=5, alpha=0.5)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax2.spines['right'].set_color('#a70000')
                ax2.tick_params(axis='y', colors='red', grid_color='red', grid_alpha=0.5, grid_linestyle='--')



    plt.show()


# show quantity and percentage at the top of each bar on graph
def show_quantity_and_percentage_on_bars(ax, total_of_rows, fontsize=11, format='.2f',
                                         show_only_percentage=False, show_only_quantity=False):
    for p in ax.patches:
        value = p.get_height()
        perc = np.round((value / total_of_rows * 100), 2)

        if show_only_percentage:
            text = f'{perc:.0f}%'
        elif show_only_quantity:
            text = f'{value:.0f}%'
        else:
            text = f'{value:.0f}\n{perc:.0f}%'

        ax.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=fontsize, color='black', xytext=(0, 13),
                    textcoords='offset points')



# check errors and return series from parameters
# ** AUXILIARY METHOD USED ONLY INTERNALLY **
def get_series_from_parameters(df=None, column=None, series=None):
    if df is not None:
        if column is None:
            raise NameError('Parameter COLUMN must be informed')

        df = df.copy()
        series = df[column]
    #
    elif series is not None:
        series = series.copy()
    else:
        raise NameError('Must inform [DF + COLUMN] parameters or [SERIES] parameter')

    return series


# Print the distribution of one given column
def print_variable_distribution(df=None, column=None, series=None, fill_nan=False):

    series = get_series_from_parameters(df=df, column=column, series=series)

    s_name = series.name #.upper()

    # get total of samples
    qty_samples = series.shape[0]
    # get total of samples without NaN values
    qty_samples_non_nan = series.count()
    # get total of samples without NaN values
    qty_samples_nan = qty_samples - qty_samples_non_nan
    perc_samples_nan = np.round( (qty_samples_nan / qty_samples * 100), 2)



    print(f'==============================================================================')
    print(f'Column: {s_name}  (DataType: {series.dtype})')
    print(f'==============================================================================')

    N = series.count()
    print(f'N         = {N} [Non-NaN: {qty_samples_non_nan}]')
    print(f'Missingness: {qty_samples_nan} ({perc_samples_nan}%)')

    #if variable is CONTINUOUS or INTEGER
    if series.dtype in [float, int]:
        if series.mode().count() > 10:
            mode = ' *** TOO MANY VALUES FOR Mode (length = %s)' % series.mode().count()
        else:
            mode = str(series.mode().to_list()).replace('[', '').replace(']', '').replace(',', ';')

        #
        min = series.min()
        max = series.max()
        mean = series.mean()
        std_dev = series.std()
        variance = series.var()
        coef_variation = std_dev / mean
        skewness = get_skewness(series)
        kurtosis = get_kurtosis(series)
        sem = std_dev / np.sqrt(N)
        # print more information about distribution
        more_info = 'Min       = %.2f  ' \
                    '\nMax       = %.2f  ' \
                    '\nSE / Std  = %.2f  (Standard Error or Std.Dev.)' \
                    '\nSEM       = %.2f  (Standard Error of the Mean) [Formula: Std.Dev/sqrt(N) => %.2f/sqrt(%s)] ' \
                    '\nMean      = %.2f +/- %.2f (Std.Dev.) [Precision of the Mean = %.2f +/- %.2f (SEM)] ' \
                    '\nMedian    = %.2f  ' \
                    '\nMode      = %s'  \
                    '\nVariance  = %.2f (Coefficient of Variation = %.2f)' \
                    '\nSkewness  = %s '  \
                    '\nKurtosis  = %s ' % \
                    (min,
                     max,
                     std_dev,
                     sem,
                     std_dev,
                     N,
                     mean,
                     std_dev,
                     mean,
                     sem,
                     series.median(),
                     mode,
                     variance,
                     coef_variation,
                     skewness,
                     kurtosis
                     )
        q25 = series.quantile(q=0.25)
        q50 = series.quantile(q=0.5)
        q75 = series.quantile(q=0.75)
        more_info += '\nQuartiles = [Q1: 25%% < %.2f]   [Q2: 50%% < %.2f]   [Q3: 75%% < %.2f]' % \
                    (q25, q50, q75)

        Q1 = q25
        Q3 = q75
        IQR = Q3 - Q1

        outliers_threshold_inf = (Q1 - 1.5 * IQR)
        if outliers_threshold_inf < min:
            outliers_threshold_inf = min

        outliers_threshold_sup = (Q3 + 1.5 * IQR)
        if outliers_threshold_sup > max:
            outliers_threshold_sup = max

        more_info += f'\n -IQR     = {IQR:.3f} (Interquartile Range: IQR = Q3-Q1)'
        more_info += f'\n -Outliers Threshold (IQR * +/-1.5):  [Lower = {outliers_threshold_inf:.3f}]   [Upper = {outliers_threshold_sup:.3f}]'

        print(more_info)

    else:
        if fill_nan:
            series.fillna('**Not Informed', inplace=True)    

        # print(f'N         = {N} {column} (rows with value, excluding NaN´s)')
        print('\nSummary Table:')
        # df_distrib = pd.DataFrame(df[column].value_counts())
        df_distrib = pd.DataFrame(series.value_counts())
        sum = df_distrib[series.name].sum()
        df_distrib['percentage'] = np.round(df_distrib[series.name] / sum * 100, 2)
        df_distrib['percentage'] = df_distrib['percentage'].map('{:,.2f}%'.format)
        df_distrib.rename(columns={series.name: 'count'}, inplace=True)
        # df_distrib['count'] = df_distrib['count'].map('{:,.0f}'.format)

        print(df_distrib)


 # return a string with Skewness info of given series
def get_kurtosis(series, short_description=False):
    kurtosis = series.kurtosis()
    ret = '%.2f' % kurtosis
    #
    if short_description:
        if kurtosis > 0:
            ret += ' (point head)'
        elif kurtosis < 0:
            ret += ' (rounded)'
        else:  # kurtosis = 0:
            ret += ' (normal)'
    #
    else:
        if kurtosis > 0:
            ret += ' (leptokurtic, point head appearance)'
        elif kurtosis < 0:
            ret += ' (platykurtic, rounded appearance)'
        else: # kurtosis = 0:
            ret += ' (mesokurtic, normal appearance)'
        #
        ret += '(PS: a high kurtosis indicates too many outliers)'
    #
    return ret



# return a string with Skewness info of given series
def get_skewness(series, short_description=False):
    skewness = series.skew()
    ret = '%.2f' % skewness
    #
    if short_description:
        if skewness < 0:
            ret += ' (Left)'
        elif skewness > 0:
            ret += ' (Right)'
        else: # skewness = 0:
            ret += ' (Symmetric)'
    #
    else:
        if skewness < 0:
            ret += ' (Left-Skewed)'
        elif skewness > 0:
            ret += ' (Right-Skewed)'
        else: # skewness = 0:
            ret += ' (Symmetric, Unskewed)'
        #
        if (skewness < -1.0) or (skewness > 1.0):
            ret += ' (Highly Skewed)'
        elif (skewness < -0.5) or (skewness > 0.5):
            ret += ' (Moderately Skewed)'
        else:
            ret += ' (Approximately Symmetric)'
    #
    return ret


# get total of samples in dataset
def get_quantity_of_rows(df):
    return df.shape[0]


#Show columns statistics, such as:
def show_columns_stats(df, columns=None):
    qty_rows = get_quantity_of_rows(df)

    #if dont set columns, get all coumns from dataset
    if columns is None:
        columns = df.columns

    #get the size of longest column name
    offset = 35
    try:
        offset = columns.str.len().max()
    except:
        offset = len(max(columns, key=len))

    #show stats for each column
    for col in columns:
        q = df.count()[col]
        perc = np.round((q / qty_rows * 100), 2)
        missing_perc = np.round((100 - perc), 2)
        missing_qty = get_quantity_of_rows(df[df[col].isnull() == True])
        has_nan = df[col].hasnans
        uniques = df[col].unique().size
        #left all columns names with same size
        col_print = (col + ('.'*offset))[0:offset]
        #print stats
        print(f'{col_print:>} = {q:>5} rows ({perc:>5}%) {missing_qty:>5} with NaN ({missing_perc:>5}%) Uniques= {uniques:>5} ')


#utilitary method to calculate the Age based on Date_of_Birth Delta
def calculate_age_from_birth_delta(days):
    today = datetime.now()
    birth = timedelta(days=days)
    diff = today - birth
    years, months, days = date_diff(today, diff)
    return np.abs(years)


#utilitary method to calculate the Age_at_Diagnosis based on Age_at_Trial and Diagnosis_Delta
def calculate_age_at_diagnosis_delta(age_at_trial, diagnosis_delta):
    age_to_subtract = calculate_years_from_days( np.abs(diagnosis_delta) )
    age_to_subtract = int(age_to_subtract)

    age_at_diagnosis = age_at_trial - age_to_subtract

    if age_to_subtract < 1:
        age_at_diagnosis = np.round(age_at_diagnosis, 0)

    return np.trunc(age_at_diagnosis)



#utilitary method to calculate the Age_at_Onset based on Age_at_Trial and Symptoms_Onset_Delta
def calculate_age_from_onset_delta(age_at_trial, symptoms_onset_delta):
    age_to_subtract = calculate_years_from_days( np.abs(symptoms_onset_delta) )
    age_to_subtract = int(age_to_subtract)

    age_at_onset = age_at_trial - age_to_subtract

    if age_to_subtract < 1:
        age_at_onset = np.round(age_at_onset, 0)

    return np.trunc(age_at_onset)



#utilitary method to calculate the number of YEARS from a given days
def calculate_years_from_days(days):
    years_total = np.NaN
    if (days != None) and (not math.isnan(days)):
        years, months, days = date_diff_from_days (days)
        years_total = years

        if months in [3, 4]:
            years_total += 0.25
        elif months in [5, 6, 7, 8]:
            years_total = years_total + 0.5
        elif months in [9, 10]:
            years_total += 0.75
        elif months >= 11:
            years_total = years_total + 1.0

    return np.abs(years_total)


#return the quantity of YEARS, MONTHS and DAYS from a given quantity of days
def date_diff_from_days(days):
    begin_date = datetime.fromisoformat('1900-01-01')
    end_date = begin_date + timedelta(days=abs(days))
    years, months, days = date_diff(begin_date=begin_date, end_date=end_date)
    return np.abs(years), np.abs(months), np.abs(days)


#return the difference between 2 dates
def date_diff(begin_date, end_date):
    delta = relativedelta.relativedelta(end_date, begin_date)
    return np.abs(delta.years), np.abs(delta.months), np.abs(delta.days)



# plot HeatMap of NaN values per column
def plot_nan_values_heatmap(data, cmap=['black','white'], cbar=False, title="", figsize=[15, 10]):
    plot_heatmap(data=data.isnull(), title=title, cmap=cmap, cbar=cbar, figsize=figsize)



# plot HeatMap 
def plot_heatmap(data, cmap=['#0050bb', '#ff7b7b'], cbar=True, title="", figsize=[15, 10]):
    fig = plt.figure(figsize=figsize)
    sns.heatmap(data, cmap=cmap, cbar=cbar)
    plt.title(title)
    plt.show()
    plt.close()



# Remove (Drop) from dataset the rows-index informed (to_delete)
def remove_rows(df, to_delete, info='', verbose=True):
    rows_previous = get_quantity_of_rows(df)
    rows_to_delete = get_quantity_of_rows(to_delete)

    df = df.drop(to_delete.index)
    rows_after = get_quantity_of_rows(df)

    if verbose:
        print(f'  - {info} Previous={rows_previous}, To delete={rows_to_delete}, After={rows_after}')

    # print(f'  - Samples Before  : {rows_previous}')
    # print(f'  - Samples Removed : {rows_to_delete}')
    # print(f'  - Samples After   : {rows_after}')
    #
    return df


#Join two dataFrames by a key column
def join_datasets_by_key(df_main, df_to_join, key_name, how='left', lsuffix='', rsuffix='', sort=False, raise_error=False):
    rows_df_main = get_quantity_of_rows(df_main)
    rows_df_to_join = get_quantity_of_rows(df_to_join)
    #
    # df_return = df_main.set_index(key_name).join(df_to_join.set_index(key_name), how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)
    df_return = df_main.join(df_to_join.set_index(key_name), on=key_name, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)
    rows_return = get_quantity_of_rows(df_return)
    #
    if (how=='left') & (rows_return > rows_df_main):
        msg = 'DF_TO_JOIN has duplicated values in KEY column. Remove duplicate keys before joining.'
        if raise_error:
            raise NameError(msg)
        else:
            print(f'ERROR: {msg}')
    #
    return df_return



# save DataFrame to CSV
def save_to_csv(df, csv_file, with_index=False):

    #create the folder if does not exist
    folder = os.path.dirname(csv_file)
    os.makedirs(folder, exist_ok=True)

    #save CSV file
    df.to_csv(csv_file, sep=',', index=with_index)
    print(f'{get_quantity_of_rows(df)} samples were saved')


#return the rows with duplicated values in given column
def get_duplicated_rows(df, column):
    return df.loc[df[column].duplicated(keep=False) == True]


#utilitary method to calculate the number of MONTHS from a given days
def calculate_months_from_days(days):
    months_total = np.NaN
    if (days != None) and (not math.isnan(days)):
        years, months, days = date_diff_from_days (days)
        months_total = (years * 12) + months
    return np.abs(months_total)


# calculate the Body Mass Index (BMI) using the formula:  BMI = weight / (height * height)
def calculate_BMI(weigth_in_kilograms, height_in_meters):
    bmi = weigth_in_kilograms / (height_in_meters * height_in_meters)
    return np.round(bmi, 0)



def print_columns_array(df):
    print('[')
    for c in df.columns:
        print(f"'{c}',")
    print(']')




def get_string_with_separators(string):
    sep = '-'*len(string)
    return f'{sep}\n{string}\n{sep}'


def print_string_with_separators(string):
    print(get_string_with_separators(string=string))



# plot graphs to compare 2 groups
def plot_grouped_variable_distribution(series_1, series_2, bins=30, figsize=[15,5], 
                                       print_more_info=True, plot_one_graph_per_row=False, 
                                       show_pareto_graph=False, check_normality=True, 
                                       show_qq_plot=False, title_group_1='Group-1', title_group_2='Group-2'):

    total_of_rows_1 = series_1.count()
    total_of_rows_2 = series_2.count()

    if (plot_one_graph_per_row) and (figsize==[15,5]):
        figsize=[15, 13]

    series_dtype = series_1.dtype

    sn = series_1.name

    print(f'==============================================================================')
    print(f'Column {sn}     (DataType: {series_dtype})')
    if not print_more_info:
        print(f'  -{title_group_1} = {total_of_rows_1:>7} rows (GROUP-1)')
        print(f'  -{title_group_2} = {total_of_rows_2:>7} rows (GROUP-2)')
    print(f'==============================================================================')

    # lead with boolean values on data_column
    if series_dtype == bool:
        series_1 = series_1.map({True: 'YES', False: 'NO'})
        series_2 = series_2.map({True: 'YES', False: 'NO'})
        series_dtype = series_1.dtype


    if (series_dtype in [float, int]) and (series_1.unique().size <= 2):
        series_1 = series_1.astype(str)
        series_2 = series_2.astype(str)
        series_dtype = series_1.dtype



    #if variable is CONTINUOUS or INTEGER (DISCRETE)
    if series_dtype in [float, int]:
        # boxplot
        df, col_groups_name = create_data_frame_from_two_groups(series_1=series_1, series_2=series_2, title_group_1=title_group_1, title_group_2=title_group_2)
        # df.rename(columns={'Group-1': title_group_1, 'Group-2': title_group_2,}, inplace=True)
        cor1 = 'cornflowerblue'
        cor2 = 'darkorange'

        my_palette = my_pal = {title_group_1: cor1, title_group_2: cor2}
        plt.figure(figsize=figsize)

        ax = plt.subplot(2, 2, 1)
        sns.boxplot(ax=ax, data=df, x=col_groups_name, y=sn, palette=my_palette)
        plt.title('Boxplot Graph')

        ax = plt.subplot(2, 2, 2)
        sns.violinplot(ax=ax, x=col_groups_name, y=sn, data=df, palette=my_palette)
        plt.title('Violin Graph')

        # histogram group-1
        q = series_1.unique().size
        if q<=2:
            b = 2
        else:
            b = bins if q > bins else q - 2

        plt.subplot(2, 2, 3)
        series_1.hist(bins=b, color=cor1)
        plt.title(f'{title_group_1}\n')

        # histogram group-2
        q = series_2.unique().size
        if q<=2:
            b = 2
        else:
            b = bins if q > bins else q - 2

        plt.subplot(2, 2, 4)
        series_2.hist(bins=b, color=cor2)
        plt.title(f'{title_group_2}\n')
        plt.show()

    #
    #else, if variable is CATEGORICAL
    else:

        if show_pareto_graph:
            if figsize==[15, 5]:
                figsize=[15, 10]

        i = 1
        i_subplot = 1

        ss = [series_1, series_2]
        dfs = []
        plt.figure(figsize=figsize)

        for series in ss:
            if i == 1:
                group_name = title_group_1
            else:
                group_name = title_group_2

            #create a dataframe to plot the data
            df_distrib = pd.DataFrame(series.value_counts())
            df_distrib = df_distrib.reset_index()
            df_distrib[sn] = df_distrib[sn].astype(float)
            df_distrib.rename(columns={series.name: 'count', 'index': sn}, inplace=True)
            sum = df_distrib['count'].sum()

            df_distrib = df_distrib.sort_values(by=sn)

            df_distrib['x'] = '.' + df_distrib[sn].astype(str)
            df_distrib['percentage'] = np.round(df_distrib['count'] / sum * 100, 2)
            df_distrib['percentage'] = df_distrib['percentage'].map('{:,.2f}%'.format)
            df_distrib['cum_sum'] = df_distrib['count'].cumsum()
            df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

            # --------------------------------------------
            # BAR GRAPH (WITH QUANTITY AND PERCENTAGE)
            # --------------------------------------------
            ax = plt.subplot(2, 2, i_subplot) if show_pareto_graph else plt.subplot(1, 2, i_subplot)

            # ax = plt.subplot(1, 2, i_subplot)
            #
            x = df_distrib['x']
            y = df_distrib['count']
            y_cum = df_distrib['cum_percentage']

            # plot bar graph
            # plt.title(f'{group_name} (Column {series.name})\n')
            plt.title(f'{group_name}\n(n={series.shape[0]})\n')
            sns.barplot(ax=ax, x=x, y=y)
            ax.set_xlabel(sn)
            show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

            #plot cumulative line
            ax2 = ax.twinx()
            ax2.plot(x, y_cum, color='blue', marker='D', ms=4, alpha=0.5)
            ax2.yaxis.set_major_formatter(PercentFormatter())
            ax2.spines['right'].set_color('#a70000')
            ax2.tick_params(axis='y', colors='blue', grid_color='blue', grid_alpha=0.5, grid_linestyle='--')


            if show_pareto_graph:
                # --------------------------------------------
                # PARETO GRAPH
                # --------------------------------------------
                # calculate the cumulative percentage
                df_distrib = df_distrib.sort_values(by='count', ascending=False)
                df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

                # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
                i_subplot += 1
                ax = plt.subplot(2, 2, i_subplot) if show_pareto_graph else plt.subplot(1, 2, i_subplot)
                #
                plt.title(f'{group_name} Cumulative Percentage (Pareto Graph)\n')

                # Plot Pareto graph
                x = df_distrib['x']
                y = df_distrib['count']
                ax = sns.barplot(x, y)
                ax.set_xlabel(sn)
                show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

                # Plot lines with cumulative percentage
                ax2 = ax.twinx()
                y_cum_perc = df_distrib['cum_percentage']
                ax2.plot(x, y_cum_perc, color='#a70000', marker='D', ms=5, alpha=0.5)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax2.spines['right'].set_color('#a70000')
                ax2.tick_params(axis='y', colors='red', grid_color='red', grid_alpha=0.5, grid_linestyle='--')

            dfs.append(df_distrib)
            i += 1
            i_subplot += 1


        #
        plt.show()

        # create and print a DataFrame with values of 2 groups
        df_1 = dfs[0].rename(columns={'count': f'{title_group_1}_Count', 'percentage': f'{title_group_1}_Perc.'})
        df_1 = df_1.drop(['x', 'cum_sum', 'cum_percentage'], axis=1)

        df_2 = dfs[1].rename(columns={'count': f'{title_group_2}_Count', 'percentage': f'{title_group_2}_Perc.'})
        df_2 = df_2.drop(['x', 'cum_sum', 'cum_percentage'], axis=1)

        df = join_datasets_by_key(df_main=df_1, df_to_join=df_2, key_name=sn, how='outer')
        df = df.rename(columns={sn: 'Values'})
        df = df.sort_values(by='Values', ascending=True)

        print(df)
        print()

    if print_more_info:
        print_grouped_variable_distribution(series_1=series_1,
                                            series_2=series_2,
                                            check_normality=check_normality,
                                            show_qq_plot=show_qq_plot,
                                            title_group_1=title_group_1,
                                            title_group_2=title_group_2,
                                            )    
        

# Print the distribution of one given column
def print_grouped_variable_distribution(series_1, series_2, check_normality=True, show_qq_plot=True, title_group_1='Group-1', title_group_2='Group-2'):

    s_name = series_1.name
    s_dtype =series_1.dtype

    # if variable is CONTINUOUS or INTEGER
    if s_dtype in [float, int]:
        # group-1 stats
        N1, mean1, median1, min1, max1, std1, var1, coef_var1, sem1, q251, q501, q751, IQR1, out_thres_low1, out_thres_up1, skewness1, kurtosis1 = \
            get_stats_from_serie(series_1)

        # group-2 stats
        # group-1 stats
        N2, mean2, median2, min2, max2, std2, var2, coef_var2, sem2, q252, q502, q752, IQR2, out_thres_low2, out_thres_up2, skewness2, kurtosis2 = \
            get_stats_from_serie(series_2)

        #
        g1 = title_group_1
        g2 = title_group_2

        print(f'GROUPS STATISTIC COMPARISON')
        print(f'           {g1:>20} {g2:>20}')
        print(f'N          {N1:>20} {N2:>20}')

        s1 = f'{mean1:.2f} +/-{std1:.2f}'
        s2 = f'{mean2:.2f} +/-{std2:.2f}'
        print(f'Mean       {s1:>20} {s2:>20}')
        print(f'SEM        {sem1:>20.2f} {sem2:>20.2f}')
        print(f'Variance   {var1:>20.2f} {var2:>20.2f}')
        print(f'Median     {median1:>20.2f} {median2:>20.2f}')

        s1 = f'{min1:.2f} / {max1:.2f}'
        s2 = f'{min2:.2f} / {max2:.2f}'
        print(f'Min/Max    {s1:>20} {s2:>20}')

        print(f'Skewness   {skewness1:>20} {skewness2:>20}')
        print(f'Kurtosis   {kurtosis1:>20} {kurtosis1:>20}')

        print(f'Quartiles:')
        print(f' -Q1 (25%) {q251:>20.2f} {q252:>20.2f}')
        print(f' -Q2 (50%) {q501:>20.2f} {q502:>20.2f}')
        print(f' -Q3 (75%) {q751:>20.2f} {q752:>20.2f}')
        print(f' -IQR      {IQR1:>20.2f} {IQR2:>20.2f}')

        print(f'Outliers:')
        out_count1 = series_1.loc[(series_1 < out_thres_low1) | (series_1 > out_thres_up1)].count()
        out_count2 = series_2.loc[(series_2 < out_thres_low2) | (series_2 > out_thres_up2)].count()
        print(f' -Count    {out_count1:>20} {out_count2:>20}')
        print(f' -Threshold (IQR * +/-1.5):')
        print(f'   -Lower  {out_thres_low1:>20.2f} {out_thres_low2:>20.2f}')
        print(f'   -Upper  {out_thres_up1:>20.2f} {out_thres_up2:>20.2f}')


        print()



#retun only the samples without NaN Values
def remove_nans_from_serie(serie):
    return serie[serie.isna() == False]


# get
def get_stats_from_serie(series):
    N = series.count()
    mean = series.mean()
    min = series.min()
    max = series.max()
    median = series.median()
    std_dev = series.std()
    variance = series.var()
    coef_variation = std_dev / mean
    skewness = get_skewness(series, short_description=True)
    kurtosis = get_kurtosis(series, short_description=True)
    sem = std_dev / np.sqrt(N)
    #
    q25 = series.quantile(q=0.25)
    q50 = series.quantile(q=0.5)
    q75 = series.quantile(q=0.75)

    Q1 = q25
    Q3 = q75
    IQR = Q3 - Q1

    outliers_threshold_lower = (Q1 - 1.5 * IQR)
    if outliers_threshold_lower < min:
        outliers_threshold_lower = min

    outliers_threshold_upper = (Q3 + 1.5 * IQR)
    if outliers_threshold_upper > max:
        outliers_threshold_upper = max

    return N, mean, median, min, max, std_dev, variance, coef_variation, sem, q25, q50, q75, IQR, outliers_threshold_lower, outliers_threshold_upper, skewness, kurtosis


# create a data-frame joining the 2 groups, labelling groups (Group-1 and Group-2)
def create_data_frame_from_two_groups(series_1, series_2, title_group_1='Group-1', title_group_2='Group-2'):
    col_group_name = 'Group'
    #
    df_1 = pd.DataFrame(series_1)
    df_1[col_group_name] = title_group_1
    #
    df_2 = pd.DataFrame(series_2)
    df_2[col_group_name] = title_group_2
    #
    # df = df_1.append(df_2)
    df = pd.concat([df_1, df_2])
    #
    return df, col_group_name



def get_model_description(model_desc):

    model_desc = str(model_desc).replace('()', '')   

    NB_models = [
        'ComplementNB', 
        'GaussianNB', 
        'CategoricalNB',
        'NB',

    ]

    KNN_models = [
        'RadiusNeighbors', 
        'RadiusNeighborsClassifier', 
        'KNeighbors',
        'KNeighborsClassifier',
        'k-NN', 
    ]

    NN_models = [
        'MLP',
        'MLPClassifier',
        'NN',
    ]

    RF_models = [
        'RandomForest',
        'RandomForestClassifier',
        'RF', 
    ]

    DT_models = [
        'DecisionTree',
        'DecisionTreeClassifier',
        'DT', 
    ]

    XGB_models = [
        'XGBClassifier',
        # 'XGBoost',
    ]

    CatBoost_models = [
        'CatBoostClassifier',
        # 'CatBoost',
    ]

    SVM_models = [
        'SVC', 
    ]

    BalancedBagging_models = [
        'BalancedBaggingClassifier',
        'Bal. Bagging',
        'BalBagging',
    ]

    BalancedRF_models = [
        'BalancedRandomForestClassifier',
        'Bal. RF',
        'BalRF',
    ]

    if model_desc in NB_models:
        return 'Naïve Bayes'
    elif model_desc in KNN_models:
        return 'k-NN'
    elif model_desc in NN_models:
        return 'Neural Networks'
    elif model_desc in RF_models:
        return 'Random Forest'
    elif model_desc in DT_models:
        return 'Decision Tree'
    elif model_desc in SVM_models:
        return 'SVM'
    elif model_desc in XGB_models:
        return 'XGBoost'
    elif model_desc in CatBoost_models:
        return 'CatBoost'
    elif model_desc in BalancedBagging_models:
        return 'Balanced Bagging'
    elif model_desc in BalancedRF_models:
        return 'Balanced Random Forest'
    else:
        return model_desc.replace('Classifier', '')



def get_model_short_description(model_desc):

    model_desc = str(model_desc).replace('()', '')   
    
    NB_models = [
        'ComplementNB', 
        'GaussianNB', 
        'CategoricalNB',
        'Naïve Bayes',
    ]

    NN_models = [
        'MLPClassifier',
        'Neural Networks',
    ]

    KNN_models = [
        'RadiusNeighborsClassifier', 
        'KNeighborsClassifier',
        'k-NN', 
    ]


    RF_models = [
        'RandomForestClassifier',
        'Random Forest', 
    ]

    DT_models = [
        'DecisionTreeClassifier',
        'Decision Tree', 
    ]

    BalancedBagging_models = [
        'BalancedBaggingClassifier',
    ]

    BalancedRF_models = [
        'BalancedRandomForestClassifier',
        'Balanced Random Forest'
    ]

    if model_desc in NB_models:
        return 'NB'
    elif model_desc in KNN_models:
        return 'k-NN'
    elif model_desc in NN_models:
        return 'NN'
    elif model_desc in RF_models:
        return 'RF'
    elif model_desc in DT_models:
        return 'DT'
    elif model_desc in BalancedBagging_models:
        return 'Bal. Bagging'
    elif model_desc in BalancedRF_models:
        return 'Bal. RF'
    else:
        return model_desc.replace('Classifier', '')



def print_array_as_list(arr):
    print('[')

    for e in arr:
        print(f"    '{e}',")

    print(']')


def print_array_as_numbered_list(arr):
    print('[')

    for i, e in enumerate(arr):
        print(f" {i} '{e}',")

    print(']')




def plot_series_side_by_side(series1, series2=None, title_list=None, figsize=[15, 3]):
    
    values_list = [series1, series2]

    # if informed the 2 series
    if (series1 is not None) and (series2 is not None):
        if title_list is None: 
            title_list = [series1.name, series2.name]

    # if informed only the "series_1"
    elif (series1 is not None) and (series2 is None):
        if title_list is None:
            title_list = [series1.name, None]

    plot_countplot_side_by_side(
        values_list=values_list, 
        title_list=title_list,
        figsize=figsize, 
    )



def plot_countplot_side_by_side(values_list, title_list, figsize=[15, 3]):

    qty = len(title_list)
    i=1

    plt.figure(figsize=figsize)
    for values, title in zip(values_list, title_list):
        if values is not None:

            subplot = int(f'1{qty}{i}')
            plt.subplot(subplot)
            plt.title(title)
            g = sns.countplot(
                x=values,
                palette= sns.color_palette("colorblind")

            )

            i += 1
            total = len(values)
            # annotate the bars with their values
            for p in g.axes.patches:
                percentage = (100 * p.get_height()/total)

                g.annotate(
                    f'{p.get_height():.0f}\n{percentage:.1f}%', 
                    xy=(
                        (p.get_x()+p.get_width()/2),
                         50 #p.get_height() * 0.1
                    ),
                    # xytext=(0, 2), 
                    textcoords='offset points', 
                    ha="center", 
                    va="center",
                    size=12,
                )    


    plt.show()
    plt.close()



def save_plot(plt, folder, file_name, dpi=300, bbox_inches='tight', save_in_pdf_format=False, save_in_eps_format=False):

    plt.savefig(
        f'{folder}/{file_name}.png', 
        bbox_inches=bbox_inches, 
        dpi=dpi
    )

    if save_in_pdf_format:
        plt.savefig(
            f'{folder}/{file_name}.pdf', 
            bbox_inches=bbox_inches, 
            dpi=dpi
        )

    if save_in_eps_format:
        plt.savefig(
            f'{folder}/{file_name}.eps', 
            bbox_inches=bbox_inches, 
            dpi=dpi
        )




### Display a 'Non-Short' and 'Short' on xticks to show in SHAP Decision Plot
def customize_xticks(ax, fontweight='bold', fontsize=10):
    xticks_modified = []
    
    # print('xticks values:', ax.get_xticks())

    for val in ax.get_xticks():
        xticks_modified.append('')
        # if val == 0.0:
        #     xticks_modified.append('Non-Short')
        # else:
        #     xticks_modified.append('')

    # xticks_modified[-1] = 'Short'        
            

    xticks_modified[1] = 'Non-Short'        
    xticks_modified[-1] = 'Short'        

    # print('xticks modif.:', xticks_modified)

    ax.set_xticks(ax.get_xticks(), xticks_modified, fontweight=fontweight, fontsize=fontsize)


def plot_shap_values_by_feature_in_one_graph(data_frames, features_ordered_names, 
                                            x_min, x_max, folder_to_save,
                                            file_name='SHAP_by_feature_values', 
                                            figsize=[5,2], bar_height = 0.1, 
                                            midline_decimal_point=False, 
                                            qty_features=14):

    dfs = data_frames.copy()

    # print(len(dfs))

    # rows = int(np.ceil(qty_features/2))
    # cols = 2
    # print(rows, cols)
    rows = 3
    cols = 4


    fig, axes = plt.subplots(
        rows, 
        cols, 
        figsize=figsize, 
        layout='constrained',
        # gridspec_kw={'height_ratios': [0.7, 0.7, 0.7, 0.7, 0.7]},
        # sharex=True, 
        # sharey=True    
    )

    # fig5 = plt.figure(constrained_layout=True)
    # widths = [2, 3, 1.5]
    # heights = [1, 3, 2]
    # spec5 = fig5.add_gridspec(ncols=3, nrows=3, width_ratios=widths,
    #                         height_ratios=heights)

    axes = axes.flatten()

    # print(axes[0], axes[1])

    alphabet = list(string.ascii_lowercase)  

    ranking = ['st', 'nd', 'rd', 'th']


    i_axes = 0
    i = 0
    for feat_ordered in features_ordered_names:

        if i == qty_features:
            break

        # print(f'FEAT_ORDERED: {feat_ordered}')

        # PLOT THE GRAPHS ordered inversed by mean SHAP value
        for feat, qty_values, df in dfs:    
            
            # print(f'    FEAT: {feat}')
            
            # print(feat, df.columns)

            if (feat == feat_ordered):
                i+=1

                if feat == 'Diagnosis_Delay': 
                    df['order'] = df['Feature Values']
                    df.order = df.order.replace({'Long': 3, 'Average': 2, 'Short': 1})
                    df.sort_values(by=['order'], inplace=True)


                feature_name = f'#{i}: {feat}'
                # rank = ranking[i-1] if i <= 3 else ranking[3]
                # feature_name = r'%s$^{%s}$ %s'%(i, rank, feat   )

                # print(f'         plotting {feat}')

                ax = axes[i_axes]

                plot_shap_values_by_feature(
                        data=df, 
                        title=f'{feat}', 
                        # figsize=[6,(qty_values / 2)],
                        bar_height=bar_height,
                        x_min=x_min,
                        x_max=x_max,
                        feature_name=feature_name,
                        midline_decimal_point=False,
                        folder_to_save=None,
                        axis=ax,
                        idx_axis=i_axes+1,
                    )
                


                ax.set_title(f'\n\n{feature_name}',fontweight='bold',fontsize=10)

                # current_row = int((i_axes/2)+0.6)
                current_row = int((i_axes/4))
                print(i_axes, current_row, rows)
                # if (current_row == rows) or (i_axes == len(axes)-2):
                if (current_row == rows-1) or (i_axes >= 6):
                    ax.set_xticks([x_min, (x_min*0.7), 0, (x_max*0.7), x_max], labels=[0,1,2,3,4],
                        fontweight='bold',
                        fontsize=8
                    )
                    ax.set_xticklabels([' ', '$\longleftarrow$ Non-Short', 0, 'Short $\longrightarrow$', ' ' ])


                i_axes += 1

                break
    
    axes[10].set_axis_off()
    axes[11].set_axis_off()


    plt.tight_layout()


    if folder_to_save is not None:
        save_plot(
            plt=plt, 
            folder=folder_to_save, 
            file_name=file_name, 
            save_in_pdf_format=True,
        )


    plt.show()
    plt.close()


def plot_shap_values_by_feature(data, title, x_min, x_max, feature_name, folder_to_save,
                                figsize=[5,2], bar_height = 0.1, midline_decimal_point=False, axis=None, idx_axis=None):
    
    
    # set colors according to positive and negatives values
    colors = ['red' if x >= 0 else 'blue' for x in data[data.columns[1]].values]
    
    if axis is None:
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=data, 
            x=data.columns[1], 
            y=data.columns[0],
            palette=colors # sns.color_palette("colorblind"),
        )
    else:
        ax = sns.barplot(
            data=data, 
            ax=axis,
            x=data.columns[1], 
            y=data.columns[0],
            palette=colors # sns.color_palette("colorblind"),
        )

      
    # annotate the bars with their values
    for p in ax.patches:
        
        x_offset = p.get_width() if p.get_width() > 0 else 0 
        
        # REPLACE "DECIMAL POINT" TO "MIDLINE DECIMAL POINT"
        val = f'{p.get_width():.3f}'
        if midline_decimal_point:
            val = val.replace('.', '·')
        
        
        ax.annotate(
            val, 
            xy=(
                x_offset, 
                p.get_y()+p.get_height()/2
            ),
            xytext=(5, 0), 
            textcoords='offset points', 
            ha="left", 
            va="center",
            size= 10 if axis is None else 8,
        )    

        # adjust bar width (or height)
        diff = p.get_height() - bar_height

        # we change the bar width
        p.set_height(bar_height)

        # we recenter the bar
        p.set_y(p.get_y() + diff * 0.5)    
        
    
    mult = 1.5
    x_min = np.round(x_min * 1.05, 3)
    x_max = np.round(x_max * 1.3, 3)
    

    ax.set_xlabel('')
    
    ax.grid(False)

    ax.spines[['right', 'top']].set_visible(False)


    # plotting only 1 graph
    if axis is None:
        plt.xlim(x_min, x_max)
        plt.xticks([x_min, (x_min*0.7), 0, (x_max*0.7), x_max], 
                fontweight='bold',
                fontsize=10)
    
        plt.axvline(x=0.0, color='#666666', lw=1, ls='--')

        ax.set_xticklabels([' ', '$\longleftarrow$ Non-Short', 0, 'Short $\longrightarrow$', ' ' ])

        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
        plt.yticks(fontsize=10)

    # ploting using subplots
    else:
        ax.axvline(x=0.0, color='#666666', lw=1, ls='--')
        ax.set_xlim(x_min, x_max)
        
        ax.set_ylabel(ax.get_ylabel(),
                fontweight='bold',
                fontsize=8)

        ax.set_xticks([])

        ax.set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels(),
                fontweight='bold',
                fontsize=8)
        
        # if idx_axis%2 == 0:
        print(idx_axis)
        if idx_axis not in [1,5,9]:
            ax.set_ylabel('', fontsize=10)


        
    
    if folder_to_save is not None:
        file_name = f'SHAP_by_feature_values_{feature_name}' 
        save_plot(
            plt=plt, 
            folder=folder_to_save, 
            file_name=file_name, 
            save_in_pdf_format=True,
        )


    if axis is None:
        plt.show()
        plt.close()




# annotate the bars of a graph with their values
def annotate_barplot(ax, horizontal=True, bar_height=0.8, midline_decimal_point=False, font_size=9,
                     deslocate=0.0, color='black', weight='normal'):

    for p in ax.patches:
        
        x_offset = p.get_width() if p.get_width() > 0 else 0 

        # REPLACE "DECIMAL POINT" TO "MIDLINE DECIMAL POINT"
        val = f'{p.get_width():.3f}'
        if midline_decimal_point:
            val = val.replace('.', '·')
        
        
        ax.annotate(
            val, 
            xy=(
                x_offset, 
                (p.get_y()+p.get_height()/2) + deslocate
            ),
            xytext=(5, 0), 
            textcoords='offset points', 
            ha="left", 
            va="center",
            size=font_size,
            color=color,
            weight=weight,
        )    

        # adjust bar width (or height)
        diff = p.get_height() - bar_height

        # we change the bar width
        p.set_height(bar_height)

        # we recenter the bar
        p.set_y(p.get_y() + diff * 0.5)    


