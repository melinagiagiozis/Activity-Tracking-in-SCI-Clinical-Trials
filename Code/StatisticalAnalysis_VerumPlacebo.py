import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.formula.api as smf
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

# Activity intensity data
csv_file_path = os.path.join(directory, 'SimulatedData/SimulatedIntensityData.csv')
intensity_data = pd.read_csv(csv_file_path)



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



# ---------------------- Activity Intensity Trends for Verum/Placebo ----------------------



# Limit to study duration (30 weeks)
data = intensity_data.iloc[:, :33]

# Separate first two columns
first_two_columns = data.iloc[:, :2]  # Select first two columns

# Convert minutes to percentage of the day
converted_data = data.iloc[:, 2:].astype(float).apply(lambda x: (x / 1440) * 100)

# Recombine the first two columns with the converted data
data = pd.concat([first_two_columns, converted_data], axis=1)

# Intensities: Sedentary activities (SED), Light Physical Activity (LPA (MPA), Vigorous Physical Activity (VPA)
intensity_levels = ['REST', 'SED', 'LPA', 'MPA', 'VPA']



# ------------------------------ Intensity levels of all patients ------------------------------



# Initialize dictionaries to store slopes and intercepts for each intensity level
slopes_all = {intensity: [] for intensity in intensity_levels}
intercepts_all = {intensity: [] for intensity in intensity_levels}

# Iterate through each row in the DataFrame by patient and intensity level
for row in range(len(data)):
    patient_id = data.iloc[row, 0]
    intensity = data.iloc[row, 1]
    
    # Process only the selected intensities
    if intensity in intensity_levels:
        valid_idx = ~np.isnan(data.iloc[row, 2:].to_list())
        if sum(valid_idx) > 1:  # Ensure there is enough data to fit
            coefficients = np.polyfit(week_numbers[valid_idx], data.iloc[row, 2:][valid_idx].to_list(), deg=1)
            slope, intercept = coefficients
            
            # Store slopes and intercepts for each intensity level
            slopes_all[intensity].append(slope)
            intercepts_all[intensity].append(intercept)

# Compute average slopes, intercepts, and standard deviations for each intensity level
avg_slopes = {intensity: np.mean(slopes_all[intensity]) for intensity in slopes_all}
avg_intercepts = {intensity: np.mean(intercepts_all[intensity]) for intensity in intercepts_all}
std_slopes = {intensity: np.std(slopes_all[intensity]) for intensity in slopes_all}
std_intercepts = {intensity: np.std(intercepts_all[intensity]) for intensity in intercepts_all}
ci_slopes = {intensity: 1.96 * sem(slopes_all[intensity]) for intensity in slopes_all}
ci_intercepts = {intensity: 1.96 * sem(intercepts_all[intensity]) for intensity in intercepts_all}

# Create and plot trends with standard deviation shading
plt.figure(figsize=(10, 6))

# Define colors using the Magma colormap
cmap = plt.get_cmap('magma')

colors = {
    'LPA': cmap(0.2),
    'MPA': cmap(0.4),
    'VPA': cmap(0.6),
    'SED': cmap(0.8),
    'REST': cmap(1)
}

for intensity in avg_slopes:
    # Generate mean trend line
    avg_p = np.poly1d([avg_slopes[intensity], avg_intercepts[intensity]])
    # Generate trend lines for standard deviation boundaries
    upper_p = np.poly1d([avg_slopes[intensity] + std_slopes[intensity], avg_intercepts[intensity] + std_intercepts[intensity]])
    lower_p = np.poly1d([avg_slopes[intensity] - std_slopes[intensity], avg_intercepts[intensity] - std_intercepts[intensity]])

    # Plot the mean trend line
    plt.plot(week_numbers, avg_p(week_numbers), color=colors[intensity], linestyle='-', label=f'{intensity}', linewidth=2)
    
    # Individual data
    for row in range(len(data)):
        if data.iloc[row, 1] == intensity:
            plt.scatter(week_numbers, data.iloc[row, 2:].to_numpy(), color=colors[intensity], s=8, label=None, zorder=0, alpha=0.4)
            plt.plot(week_numbers, data.iloc[row, 2:].to_numpy(), color=colors[intensity], linewidth=0.4, zorder=0, alpha=0.2)

# Finalize plot
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Weeks since injury', fontsize=22)
plt.ylabel('Time [% of day]', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18, loc='upper left')
plt.tight_layout(pad=2.0)
plt.ylim(-5, 105)
plt.xlim(-1, 31)
plt.grid()
plt.savefig(directory + '/Figures/Intensity_Trends.png', dpi=300)
plt.close()



# ------------------------------ Intensity levels of verum and placebo ------------------------------



# Initialize dictionaries to store slopes and intercepts for each intensity level
slopes_placebo = {intensity: [] for intensity in intensity_levels}
intercepts_placebo = {intensity: [] for intensity in intensity_levels}
slopes_verum = {intensity: [] for intensity in intensity_levels}
intercepts_verum = {intensity: [] for intensity in intensity_levels}

week_numbers = np.arange(31)

# Iterate through each row in the DataFrame by patient and intensity level
for row in range(len(data)):
    patient_id = data.iloc[row, 0]
    intensity = data.iloc[row, 1]
    
    # Process only the selected intensities
    if intensity in intensity_levels:
        # Determine if the patient is in placebo or verum group
        if patient_id in placebo or patient_id in verum:
            valid_idx = ~np.isnan(data.iloc[row, 2:].to_list())
            if sum(valid_idx) > 1:  # Ensure there is enough data to fit
                coefficients = np.polyfit(week_numbers[valid_idx], data.iloc[row, 2:][valid_idx].to_list(), deg=1)
                slope, intercept = coefficients
                
                # Store slopes and intercepts by group and intensity
                if patient_id in placebo:
                    slopes_placebo[intensity].append(slope)
                    intercepts_placebo[intensity].append(intercept)
                else:
                    slopes_verum[intensity].append(slope)
                    intercepts_verum[intensity].append(intercept)

# Compute average slopes and intercepts for each intensity level by group
avg_slopes_placebo = {intensity: np.mean(slopes_placebo[intensity]) for intensity in slopes_placebo}
avg_intercepts_placebo = {intensity: np.mean(intercepts_placebo[intensity]) for intensity in intercepts_placebo}
avg_slopes_verum = {intensity: np.mean(slopes_verum[intensity]) for intensity in slopes_verum}
avg_intercepts_verum = {intensity: np.mean(intercepts_verum[intensity]) for intensity in intercepts_verum}

# Create and plot trends
plt.figure(figsize=(10, 6))
for intensity in avg_slopes_placebo:
    # Generate trend lines for placebo and verum groups
    avg_p_placebo = np.poly1d([avg_slopes_placebo[intensity], avg_intercepts_placebo[intensity]])
    avg_p_verum = np.poly1d([avg_slopes_verum[intensity], avg_intercepts_verum[intensity]])
    
    # Plot average trends
    plt.plot(week_numbers, avg_p_placebo(week_numbers), color=colors[intensity], linestyle='--', label=f'{intensity} - Placebo', linewidth=2)
    plt.plot(week_numbers, avg_p_verum(week_numbers), color=colors[intensity], linestyle='-', label=f'{intensity} - Verum', linewidth=2)

    # Plot shaded area for standard deviation
    avg_p = np.poly1d([avg_slopes[intensity], avg_intercepts[intensity]])
    
    # Generate trend lines for standard deviation boundaries
    upper_p = np.poly1d([avg_slopes[intensity] + ci_slopes[intensity], avg_intercepts[intensity] + ci_intercepts[intensity]])
    lower_p = np.poly1d([avg_slopes[intensity] - ci_slopes[intensity], avg_intercepts[intensity] - ci_intercepts[intensity]])

    plt.fill_between(week_numbers[2:], lower_p(week_numbers[2:]), upper_p(week_numbers[2:]), color=colors[intensity], alpha=0.1, zorder=0)

# Create custom legend entries for verum and placebo
verum_line = mlines.Line2D([], [], color='k', linestyle='-', label='Verum')
placebo_line = mlines.Line2D([], [], color='k', linestyle='--', label='Placebo')
ci_line = mlines.Line2D([], [], color='k', alpha=0.3, linestyle='-', linewidth=9, label='95% CI')
plt.legend(handles=[verum_line, placebo_line, ci_line], fontsize=18, loc='upper left')

# Finalize plot
plt.grid()
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Weeks since injury', fontsize=22)
plt.ylabel('Time [% of day]', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim(-5, 105)
plt.xlim(-1, 31)
plt.tight_layout(pad=2.0)
plt.savefig(directory + '/Figures/Intensities_VerumPlacebo.png', dpi=300)
plt.close()



# ---------------------- Statistical Test for Verum/Placebo Intensity Trends ----------------------



# Reshape intensity data to long format for each intensity level
data = pd.melt(data, id_vars=['Participant_ID', 'Parameter'],
               value_vars=['Week ' + str(i) for i in range(30)],
               var_name='Week', value_name='Minutes')

# Convert 'Week' to a numeric week number
data['Week'] = data['Week'].str.extract('(\d+)').astype(int)

# Add 'Group' column to indicate placebo or verum based on Participant_ID
data['Group'] = data['Participant_ID'].apply(lambda x: 'Placebo' if x in placebo else 'Verum')

# Drop rows with missing intensity minutes
data = data.dropna(axis=0)

# Merge with clinical data to include additional variables
data = pd.merge(data, clinical_data[['Participant_ID', 'Age_at_Injury', 'Sex', 'Site']], 
                on='Participant_ID', how='left')

# Convert Sex to numeric (0 = Female, 1 = Male)
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})

# Apply function to get first measurement and week
intensity_temp = intensity_data.copy()
intensity_temp[['First_Measurement', 'First_Measurement_Week']] = intensity_temp.iloc[:, 2:].apply(
    first_measurement, axis=1, result_type='expand'
)

# Map first measurement and first measurement week to the data dataframe
data['First_Measurement'] = data.set_index(['Participant_ID', 'Parameter']).index.map(
    intensity_temp.drop_duplicates(subset=['Participant_ID', 'Parameter'])
    .set_index(['Participant_ID', 'Parameter'])['First_Measurement']
)

data['First_Measurement_Week'] = data.set_index(['Participant_ID', 'Parameter']).index.map(
    intensity_temp.drop_duplicates(subset=['Participant_ID', 'Parameter'])
    .set_index(['Participant_ID', 'Parameter'])['First_Measurement_Week']
)

# Ensure only one baseline per participant
data[['Participant_ID', 'First_Measurement', 'First_Measurement_Week']].drop_duplicates()

# Correct for number of measurements
number_of_measurements = intensity_data.set_index('Participant_ID').notna().sum(axis=1).reset_index()
number_of_measurements.columns = ['Participant_ID', 'Measurement_Count']
data = pd.merge(data, number_of_measurements, on='Participant_ID', how='left')

# Initialize a dictionary to store results for each intensity level
results_dict = {}

# Loop through each intensity level
for intensity in intensity_levels:
    # Filter the data for the current intensity level
    intensity_data = data[data['Parameter'] == intensity]
    
    # Fit a mixed-effects model including all variables
    model = smf.mixedlm(
        'Minutes ~ Week * Group + First_Measurement + Site + Age_at_Injury + Sex + Measurement_Count + First_Measurement_Week',
        data=intensity_data,
        groups=intensity_data['Participant_ID'],
        re_formula="~Week"
    )
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
    summary_csv_path = os.path.join(directory, f'Results/MixedModel_Intensity_VerumPlacebo_Summary_{intensity}.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    # if print_results:
    print(f'Results for {intensity} intensity level:')
    print(result.summary())
    print(f'Model results saved to {summary_csv_path}')
    print("\n")

    # Store the final dataset for further analysis
    data_all_participants = data

    if print_results:
        print('--------------------------')
        print('Overall interaction effect')
        print(result.summary())
        print('--------------------------')
