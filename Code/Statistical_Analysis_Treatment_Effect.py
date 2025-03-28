import os
import random
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
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
csv_file_path = os.path.join(directory, 'Data/Simulated_Energy_Expenditure_Data.csv')
energy_expenditure = pd.read_csv(csv_file_path)  

# Clinical data
csv_file_path = os.path.join(directory, 'Data/Simulated_Clinical_Data.csv')
clinical_data = pd.read_csv(csv_file_path) 

# Activity intensity data
csv_file_path = os.path.join(directory, 'Data/Simulated_Intensity_Data.csv')
intensity_data = pd.read_csv(csv_file_path)

# Treatment key
csv_file_path = os.path.join(directory, 'Data/Simulated_Treatment_Key.csv')
treatment_key = pd.read_csv(csv_file_path)



# ---------------------- Energy Expenditure Trends for Verum/Placebo ----------------------



# Assign participants to placebo and verum groups based on the Treatment column
verum = treatment_key[treatment_key['Treatment'] == 'Nogo-A Inhibitor']['Participant_ID'].tolist()
placebo = treatment_key[treatment_key['Treatment'] == 'placebo']['Participant_ID'].tolist()

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
plt.savefig(directory + '/Figures/EE_Trends_Treatment_Effect.png', dpi=300)
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

data = pd.merge(data, clinical_data[['Participant_ID', 'Age', 'Sex', 'Site', 'NLI']], 
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
                    '+ Age + Sex + NLI + Measurement_Count + First_Measurement_Week', 
                    data=data, 
                    groups=data['Participant_ID'], 
                    re_formula='~Week')

result = model.fit()

# Extract relevant statistics into a DataFrame
summary_df = pd.DataFrame({
    'Variable': result.params.index,
    'Coefficient': np.round(result.params.values, decimals=2),
    'Standard Error': np.round(result.bse.values, decimals=2),
    'p-value': np.round(result.pvalues.values, decimals=3),
    'Confidence Interval Lower Bound': np.round(result.conf_int().iloc[:, 0].values, decimals=2),
    'Confidence Interval Upper Bound': np.round(result.conf_int().iloc[:, 1].values, decimals=2)
})

variable_mapping = {
    "Intercept": "Baseline (Intercept)",
    "Treatment_Group[T.Verum]": "Treatment Group (Verum)",
    'Site[T.BCA]': "Site: Barcelona",
    'Site[T.BSL]': "Site: Basel",
    'Site[T.HDG]': "Site: Heidelberg",
    'Site[T.HLU]': "Site: Hessisch-Lichtenau",
    'Site[T.MNU]': "Site: Murnau",
    'Site[T.NTL]': "Site: Nottwil",
    'Site[T.PRG]': "Site: Prague",
    'Site[T.TIN]': "Site: Tübingen",
    'Site[T.ZRH]': "Site: Zurich",
    "NLI[T.C2]": "NLI: C2",
    "NLI[T.C3]": "NLI: C3",
    "NLI[T.C4]": "NLI: C4",
    "NLI[T.C5]": "NLI: C5",
    "NLI[T.C6]": "NLI: C6",
    "NLI[T.C7]": "NLI: C7",
    "NLI[T.INT]": "NLI: no NLI",
    "Week": "Time (Week)",
    "Week:Treatment_Group[T.Verum]": "Interaction: Week x Treatment Group",
    "Age": "Age at Injury",
    "Sex": "Sex (Male=1, Female=0)",
    "Measurement_Count": "Number of Measurements",
    "First_Measurement": "First measured EE",
    "First_Measurement_Week": "Week of first measured EE",
    "Group Var": "Patient-Specific Variability",
    "Group x Week Cov": "Covariance: Intercept and Week",
    "Week Var": "Patient-Specific Slope Variability"
}

# Apply renaming to the DataFrame
summary_df["Variable"] = summary_df["Variable"].replace(variable_mapping)

# Save the DataFrame to a CSV file
summary_csv_path = os.path.join(directory, 'Results/Mixed_Model_Treatment_Effect.csv')
summary_df.to_csv(summary_csv_path, index=False)

if print_results:
    print('------- Mixed Linear Model -------')
    print(f'Model results saved to Results/{summary_csv_path}')

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
plt.savefig(directory + '/Figures/Intensity_Treatment_Effect.png', dpi=300)
plt.close()



# ---------------------- Statistical Test for Verum/Placebo Intensity Trends ----------------------



# Initialize storage for regression results
summary_df = []

# Compute regression statistics and store results
for intensity in intensity_levels:
    slope = avg_slopes[intensity]
    intercept = avg_intercepts[intensity]
    slope_se = std_slopes[intensity] / np.sqrt(len(slopes_all[intensity]))
    ci_lower, ci_upper = slope - 1.96 * slope_se, slope + 1.96 * slope_se
    t_score = slope / slope_se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_score)))
    df = len(slopes_all[intensity]) - 1

    # Store results
    summary_df.append({
        'Intensity': intensity,
        'Slope (%/week)': np.round(slope, 2),
        'Intercept (% of day)': np.round(intercept, 2),
        '95% CI (Slope)': f'[{np.round(ci_lower, 2)}, {np.round(ci_upper, 2)}]',
        'p-value': '<0.001' if p_value < 0.001 else f'{p_value:.3f}',
        'df': df
    })

    if print_results:
        print(f'Results for {intensity} intensity level:')
        print(f"Linear Regression Model: t({df}) = {np.round(t_score, decimals=1)}, "
                    f"slope = {np.round(slope, decimals=1)}, "
                    f"95% CI [{np.round(ci_lower, decimals=1)}, {np.round(ci_upper, decimals=1)}], "
                    f"p {'<0.001' if p_value < 0.001 else np.round(p_value, 3)}")
        print(f'Model results saved to Results/{summary_csv_path}')
        print("\n")

# Save the DataFrame to a CSV file
summary_csv_path = os.path.join(directory, f'Results/Regression_Model_Intensity_Trends.csv')
summary_df = pd.DataFrame(summary_df)
summary_df.to_csv(summary_csv_path, index=False)
print('Regression results saved to Results/Regression_Model_Intensity_Trends.csv')
