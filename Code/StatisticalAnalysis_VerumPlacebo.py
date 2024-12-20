import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# Printing
print_results = False

# Set up path to directory with data
directory = os.getcwd()

# Energy expenditure
csv_file_path = os.path.join(directory, 'SimulatedData/SimulatedEnergyExpenditure.csv')
energy_expenditure = pd.read_csv(csv_file_path)  

# Clinical data
csv_file_path = os.path.join(directory, 'SimulatedData/SimulatedClinicalData.csv')
clinical_data = pd.read_csv(csv_file_path) 



# ---------------------- Energy Expenditure Trends for Verum/Placebo ----------------------



# Set a seed for reproducibility
random.seed(42)

# Generate the list of participants from the data
participants = energy_expenditure.iloc[:, 0].tolist()

# Randomly select 28 participants for placebo and verum
placebo = random.sample(participants, 28)
verum = [participant for participant in participants if participant not in placebo]

# Prepare lists to store trends
slopes_placebo, intercepts_placebo = [], []
slopes_verum, intercepts_verum = [], []

week_numbers = np.arange(len(energy_expenditure.columns) - 1)

# Iterate over each row to fit a polynomial and collect coefficients
for row in range(len(energy_expenditure)):
    if energy_expenditure.iloc[row, 0] in placebo:
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        if sum(valid_idx) > 1:
            coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
            slopes_placebo.append(coefficients[0])
            intercepts_placebo.append(coefficients[1])
    else:
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        if sum(valid_idx) > 1:
            coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
            slopes_verum.append(coefficients[0])
            intercepts_verum.append(coefficients[1])

# Compute the average slope and intercept
avg_slope_placebo = np.mean(slopes_placebo)
avg_intercept_placebo = np.mean(intercepts_placebo)
avg_slope_verum = np.mean(slopes_verum)
avg_intercept_verum = np.mean(intercepts_verum)

# Use the average coefficients to create the average trend polynomial
avg_p_placebo = np.poly1d([avg_slope_placebo, avg_intercept_placebo])
avg_p_verum = np.poly1d([avg_slope_verum, avg_intercept_verum])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(week_numbers, avg_p_placebo(week_numbers), color='#F8766D', linestyle='--', label='Placebo', linewidth=3)
plt.plot(week_numbers, avg_p_verum(week_numbers), color='#00BFC4', linestyle='--', label='Verum', linewidth=3)

for row in range(len(energy_expenditure)):
    if energy_expenditure.iloc[row, 0] in placebo:
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), '.', color='#F8766D', alpha=0.2, markersize=8)
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), color='#F8766D', alpha=0.1, linewidth=2)
    else:
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), '.', color='#00BFC4', alpha=0.2, markersize=8)
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), color='#00BFC4', alpha=0.1, linewidth=2)

# Finalize plot
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout(pad=5.0)
plt.xlabel('Weeks since injury', fontsize=18)
plt.ylabel('Energy expenditure [kcal/day]', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.savefig(directory + '/Figures/Trends_VerumPlacebo.png', dpi=300)
plt.close()



# ---------------------- Statistical Test for Verum/Placebo Trends ----------------------



data = pd.melt(energy_expenditure, id_vars=['Participant_ID'],
               value_vars=['Week ' + str(i) for i in range(len(energy_expenditure.iloc[0, :])-1)],
               var_name='Week', value_name='Energy_Expenditure')

# Convert 'Week' to a numeric value indicating the week number
data['Week'] = data['Week'].str.extract('(\d+)').astype(int)

# Add a 'Group' column to indicate whether each observation is from a placebo or verum Participant
data['Treatment_Group'] = data['Participant_ID'].apply(lambda x: 'Placebo' if x in placebo else 'Verum')

data = data.dropna(axis=0)

data = pd.merge(data, clinical_data[['Participant_ID', 'Age_at_Injury', 'Sex', 'Site']], 
                       on='Participant_ID', how='left')

# Correct for sex
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})

# Feed backward to correct for baseline
energy_expenditure_temp = energy_expenditure.copy()
energy_expenditure_temp = energy_expenditure_temp.iloc[:, 1:].fillna(method='bfill', axis=1)
energy_expenditure_temp['Participant_ID'] = energy_expenditure['Participant_ID']
energy_expenditure_temp['First_Measurement'] = energy_expenditure_temp.iloc[:, 1]

# Create a function that returns the energy expenditure and week of the first measurement
def first_measurement(row):
    measurement = row.dropna()
    if measurement.size > 0:
        first_value = measurement.iloc[0]
        first_col_number = row.index.get_loc(measurement.index[0])
        return first_value, first_col_number
    return None, None

# Apply this function row-wise across the energy expenditure data
energy_expenditure_temp = energy_expenditure.copy()
energy_expenditure_temp[['First_Measurement', 'First_Measurement_Week']] = energy_expenditure_temp.iloc[:, 1:].apply(
    first_measurement, axis=1, result_type='expand'
)

# Add 'Participant_ID'
energy_expenditure_temp['Participant_ID'] = energy_expenditure['Participant_ID']

# Map the first measurement and week of the first measurement to the data dataframe
data['First_Measurement'] = data['Participant_ID'].map(energy_expenditure_temp.set_index('Participant_ID')['First_Measurement'])
data['First_Measurement_Week'] = data['Participant_ID'].map(energy_expenditure_temp.set_index('Participant_ID')['First_Measurement_Week'])

# Drop duplicates to ensure only one baseline per participant
data[['Participant_ID', 'First_Measurement', 'First_Measurement_Week']].drop_duplicates()

# Correct for number of measurements
number_of_measurements = energy_expenditure.set_index('Participant_ID').notna().sum(axis=1).reset_index()
number_of_measurements.columns = ['Participant_ID', 'Measurement_Count']
data = pd.merge(data, number_of_measurements, on='Participant_ID', how='left')

warnings.simplefilter("ignore", ConvergenceWarning)

# Fit the mixed-effects model (with site and age as effects)
model = smf.mixedlm('Energy_Expenditure ~ Week * Treatment_Group + First_Measurement + Site'
                    '+ Age_at_Injury + Sex + Measurement_Count + First_Measurement_Week', 
                    data=data, 
                    groups=data['Participant_ID'], 
                    re_formula='~Week')

result = model.fit()

# Extract relevant statistics into a DataFrame
summary_df = pd.DataFrame({
    'Parameter': result.params.index,
    'Coefficient': result.params.values,
    'Standard Error': result.bse.values,
    'p-value': result.pvalues.values,
    'Confidence Interval Lower': result.conf_int().iloc[:, 0],
    'Confidence Interval Upper': result.conf_int().iloc[:, 1]
})

# Save the DataFrame to a CSV file
summary_csv_path = os.path.join(directory, 'Results/MixedModel_VerumPlacebo_Summary.csv')
summary_df.to_csv(summary_csv_path, index=False)

if print_results:
    print('------- Mixed Linear Model -------')
    print(f'Model results saved to {summary_csv_path}')

# For later analysis of UEMS
data_all_participants = data

if print_results:
    print('--------------------------')
    print('Overall interaction effect')
    print(result.summary())
    print('--------------------------')
