import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


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



# ---------------------- Energy Expenditure Trends for Complete/Incomplete Injuries ----------------------



# Set a seed for reproducibility
random.seed(42)

# Generate the list of participants from the data
participants = energy_expenditure.iloc[:, 0].tolist()

# Randomly select 28 participants for placebo and verum
placebo = random.sample(participants, 28)
verum = [participant for participant in participants if participant not in placebo]

# Randomly select 11 participants placebo and 17 verum Participants as complete
complete = random.sample(placebo, 11) + random.sample(verum, 17)
incomplete = [participant for participant in participants if participant not in complete]
completeness_groups = [complete, incomplete]
completeness_groups_names = ['complete', 'incomplete']

# Average trends for completeness groups
slopes_complete = []
intercepts_complete = []
slopes_incomplete = []
intercepts_incomplete = []

week_numbers = np.arange(len(energy_expenditure.iloc[0, 1:]))

# Iterate over each row to fit a polynomial and collect coefficients
for row in range(len(energy_expenditure)):
    if energy_expenditure.iloc[row, 0] in completeness_groups[0]:  # Complete injury group
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        if sum(valid_idx) > 1:
            coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
            slopes_complete.append(coefficients[0])
            intercepts_complete.append(coefficients[1])
    elif energy_expenditure.iloc[row, 0] in completeness_groups[1]:  # Incomplete injury group
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        if sum(valid_idx) > 1:
            coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
            slopes_incomplete.append(coefficients[0])
            intercepts_incomplete.append(coefficients[1])

# Compute the average slope and intercept for both groups
avg_slope_complete = np.mean(slopes_complete)
avg_intercept_complete = np.mean(intercepts_complete)
avg_slope_incomplete = np.mean(slopes_incomplete)
avg_intercept_incomplete = np.mean(intercepts_incomplete)

# Use the average coefficients to create the average trend polynomial
avg_p_complete = np.poly1d([avg_slope_complete, avg_intercept_complete])
avg_p_incomplete = np.poly1d([avg_slope_incomplete, avg_intercept_incomplete])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(week_numbers, avg_p_complete(week_numbers), color='k', linestyle='--', label='Complete Injury')
plt.plot(week_numbers, avg_p_incomplete(week_numbers), color='crimson', linestyle='--', label='Incomplete Injury')

# Plot individual trends for complete and incomplete injury groups
for row in range(len(energy_expenditure)):
    if energy_expenditure.iloc[row, 0] in completeness_groups[0]:  # Complete injury group
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), '.', color='crimson', alpha=0.2)
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), color='crimson', alpha=0.1)
    elif energy_expenditure.iloc[row, 0] in completeness_groups[1]:  # Incomplete injury group
        valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), '.', color='k', alpha=0.2)
        plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), color='k', alpha=0.1)

# Customize the plot
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Weeks since injury', fontsize=12)
plt.ylabel('Energy expenditure [kcal/day]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()

# Save the plot
plt.savefig(directory + '/Figures/Trends_CompleteIncomplete.png', dpi=300)
plt.close()



# ---------------------- Energy Expenditure Trends for Verum/Placebo of Complete/Incomplete Injuries ----------------------



# For later analysis of UEMS
data_all_participants_completeness = []

for group, group_name in zip(completeness_groups, completeness_groups_names):

    plt.figure(figsize=(10, 6))  # Start a new figure for each group
    
    # Plot both Placebo and Verum in the same plot
    for treatment_group, treatment_label, color in zip([placebo, verum], ['Placebo', 'Verum'], ['#F8766D', '#00BFC4']):
        slopes_group = []
        intercepts_group = []
        
        # Iterate over each row to fit a polynomial and collect coefficients for the current group and treatment
        for row in range(len(energy_expenditure)):
            if energy_expenditure.iloc[row, 0] in group and energy_expenditure.iloc[row, 0] in treatment_group:
                valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
                if sum(valid_idx) > 1:
                    coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
                    slopes_group.append(coefficients[0])
                    intercepts_group.append(coefficients[1])

        # Compute the average slope and intercept for the current group and treatment
        avg_slope_group = np.mean(slopes_group)
        avg_intercept_group = np.mean(intercepts_group)
        
        # Use the average coefficients to create the average trend polynomial
        avg_p_group = np.poly1d([avg_slope_group, avg_intercept_group])

        # Plot average trend line for current treatment group
        plt.plot(week_numbers, avg_p_group(week_numbers), color=color, linestyle='--', 
                 label=f'{treatment_label}', linewidth=3)

        # Plot individual lines for each Participant in the current group and treatment
        for row in range(len(energy_expenditure)):
            if energy_expenditure.iloc[row, 0] in group and energy_expenditure.iloc[row, 0] in treatment_group:
                valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
                plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), 
                         '.', color=color, alpha=0.2, markersize=8)
                plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), 
                         color=color, alpha=0.1, linewidth=2)
    
    # Customize the plot
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel('Weeks since injury', fontsize=22)
    plt.ylabel('Motor-' + group_name + ' EE [kcal/day]', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout(pad=3.0)
    plt.legend(fontsize=18, loc='upper left')
    plt.grid()
        
    # Save the plot
    plt.savefig(f'{directory}/Figures/Trends_{group_name.capitalize()}.png', dpi=300)
    plt.close()

    
    # ---------------------- Statistical Tests for Verum/Placebo of Complete/Incomplete Injuries ----------------------



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

    # Apply this function row-wise across the energy expenditure data (excluding Participant_ID)
    energy_expenditure_temp = energy_expenditure.copy()
    energy_expenditure_temp[['First_Measurement', 'First_Measurement_Week']] = energy_expenditure_temp.iloc[:, 1:].apply(
        first_measurement, axis=1, result_type='expand'
    )

    # Add 'Participant_ID' back to the temp dataframe for merging
    energy_expenditure_temp['Participant_ID'] = energy_expenditure['Participant_ID']

    # Map the energy expenditure and week of the first measurement to the data dataframe
    data['First_Measurement'] = data['Participant_ID'].map(energy_expenditure_temp.set_index('Participant_ID')['First_Measurement'])
    data['First_Measurement_Week'] = data['Participant_ID'].map(energy_expenditure_temp.set_index('Participant_ID')['First_Measurement_Week'])

    # Drop duplicates to ensure only one baseline per participant
    data[['Participant_ID', 'First_Measurement', 'First_Measurement_Week']].drop_duplicates()

    # Correct for number of measurements
    number_of_measurements = energy_expenditure.set_index('Participant_ID').notna().sum(axis=1).reset_index()
    number_of_measurements.columns = ['Participant_ID', 'Measurement_Count']
    data = pd.merge(data, number_of_measurements, on='Participant_ID', how='left')

    # Fit the mixed-effects model (with site and age as effect)
    model = smf.mixedlm('Energy_Expenditure ~ Week * Treatment_Group + First_Measurement + Site'
                        '+ Age_at_Injury + Sex + Measurement_Count + First_Measurement_Week', 
                        data=data, 
                        groups=data['Participant_ID'], 
                        re_formula='~Week')
    
    # For later analysis of UEMS
    data_all_participants_completeness.append(data)

    # Fit the model
    result = model.fit()
    if print_results:
        print('--------------------------')
        print(f'Statistical Test Results for {group_name.capitalize()} Group:')
        print(result.summary())
        print('\n' + '='*80 + '\n')
        print('--------------------------')

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
    summary_csv_path = os.path.join(directory, f'Results/MixedModel_{group_name.capitalize()}_Summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    if print_results:
        print(f'------- Mixed Linear Model {group_name.capitalize()} -------')
        print(f'Model results saved to {summary_csv_path}')