import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


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



# ---------------------- Individual Energy Expenditure Trends ----------------------



# Initialize collections for coefficients and predictions
all_coefficients = []
predictions = []
individual_results = []  # For storing Participant-specific results

# Plot setup
plt.figure(figsize=(10, 6))
week_numbers = np.arange(len(energy_expenditure.columns) - 1)
plt.grid(alpha=0.3)

# Collect polynomial coefficients for each row
for row in range(len(energy_expenditure.iloc[:, 0])):
    # Identify valid data points
    valid_idx = ~np.isnan(energy_expenditure.iloc[row, 1:].to_list())
    n = sum(valid_idx)  # Number of valid data points

    if n > 1:  # Only calculate slope and other metrics if n > 1
        coefficients = np.polyfit(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_list(), deg=1)
        all_coefficients.append(coefficients)
        slope, intercept = coefficients

        # Calculate standard error of the slope
        slope_se = np.std(energy_expenditure.iloc[row, 1:][valid_idx]) / np.sqrt(n)
        ci_lower = slope - 1.96 * slope_se
        ci_upper = slope + 1.96 * slope_se
        t_score = slope / slope_se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_score)))
    else:
        # If n = 1, mark slope and related metrics as N/A
        slope, intercept = None, None
        slope_se, ci_lower, ci_upper, t_score, p_value = None, None, None, None, None

    # Add to individual results
    Participant_id = energy_expenditure.iloc[row, 0]
    individual_results.append({
        'Participant_ID': Participant_id,
        'Slope (kcal/week)': np.round(slope, decimals=1) if slope is not None else 'N/A',
        'Intercept (kcal)': np.round(intercept, decimals=1) if intercept is not None else 'N/A',
        '95% CI (Slope)': f'[{ci_lower:.1f}, {ci_upper:.1f}]' if ci_lower and ci_upper else 'N/A',
        'p-value': '<0.001' if p_value and p_value < 0.001 else f'{p_value:.3f}' if p_value else 'N/A',
        'n': n
    })

    plt.scatter(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_numpy(), 
                color='gray', s=8, label=None, zorder=0, alpha=0.2)
    plt.plot(week_numbers[valid_idx], energy_expenditure.iloc[row, 1:][valid_idx].to_numpy(), 
                color='gray', linewidth=0.4, zorder=0, alpha=0.2)

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)

# Calculate the mean of the coefficients
if all_coefficients:
    mean_coeffs = np.mean(all_coefficients, axis=0)
    coeffs_std = np.std(all_coefficients, axis=0)
    coeffs_se = coeffs_std / np.sqrt(n)

    # Create the mean polynomial
    p_mean = np.poly1d(mean_coeffs)

    # Calculate 95% confidence intervals
    slope = mean_coeffs[0]
    slope_se = coeffs_se[0]
    ci_upper = mean_coeffs + 1.96 * coeffs_se
    ci_lower = mean_coeffs - 1.96 * coeffs_se

    # Calculate z-score and p-value for the slope
    t_score = slope / slope_se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_score)))  # Two-tailed p-value

    # Calculate degrees of freedom
    n = len(all_coefficients)
    df = n - 1

    # Print the slope, p-value, confidence intervals, and degrees of freedom
    if print_results:
        print('------- Increase of EE trend -------')
        print(f'Linear Regression Model: t({df}) = {np.round(t_score, decimals=1)}, '
              f'slope = {np.round(slope, decimals=1)}, '
              f'95% CI [{np.round(ci_lower[0], decimals=1)}, {np.round(ci_upper[0], decimals=1)}], '
              f"p {'<0.001' if p_value < 0.001 else np.round(p_value, 3)}")
        
    # Plot the mean polynomial fit with IQR (starting at week 3)
    plt.plot(week_numbers[3:], p_mean(week_numbers)[3:], '--', color='k', linewidth=1, label='trend line', zorder=2)
    
# Save individual results
results_df = pd.DataFrame(individual_results)
output_file = os.path.join(directory, 'Results/Regression_Model_EE_Trends.csv')
results_df.to_csv(output_file, index=False)
print('Individual results saved to Results/Regression_Model_EE_Trends.csv')

# Finalize plot
plt.ylabel('Energy Expenditure [kcal/day]', fontsize=14)
plt.xlabel('Weeks Since Injury', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, len(week_numbers) - 1)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig(directory + '/Figures/EE_Trends.png', dpi=300)
plt.close()


# ---------------------- Baseline Energy Expenditure as Predictor ----------------------


# Prepare energy expenditure data
energy_expenditure_temp = energy_expenditure.copy()
energy_expenditure_temp = energy_expenditure_temp.iloc[:, 1:].bfill(axis=1)

# Extract baseline EE and future EE averages
baseline_ee = energy_expenditure_temp.iloc[:, 1]  # First EE measurement after ID column
future_ee_avg = energy_expenditure_temp.iloc[:, 2:].mean(axis=1)  # Mean of all subsequent EE measurements

# Prepare data for regression
X = baseline_ee.values.reshape(-1, 1)
y = future_ee_avg.values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Retrieve regression coefficients
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

# Calculate confidence intervals for the slope
n = len(X)
residuals = y - (slope * X.flatten() + intercept)
se_slope = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(np.sum((X.flatten() - np.mean(X.flatten()))**2))
ci_lower = slope - 1.96 * se_slope
ci_upper = slope + 1.96 * se_slope

# Calculate p-value for the slope
t_stat = slope / se_slope
df = n - 2
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

# Print results in a professional format
if print_results:
    print('------- Baseline Energy Expenditure as a Predictor -------')
    print(f'Linear Regression Model: t({df}) = {np.round(t_stat, 2)}, '
          f'slope = {np.round(slope, 2)}, '
          f'95% CI [{np.round(ci_lower, 2)}, {np.round(ci_upper, 2)}], '
          f"p {'<0.001' if p_value < 0.001 else np.round(p_value, 3)}")

# Store results in a structured dictionary
regression_results = {
    'Metric': [
        'Slope (kcal/day)',
        'Intercept (kcal/day)',
        'R-squared',
        '95% CI Lower (Slope)',
        '95% CI Upper (Slope)',
        't-statistic',
        'p-value',
        'Degrees of Freedom (df)'
    ],
    'Value': [
        np.round(slope, 2),
        np.round(intercept, 2),
        np.round(r_squared, 3),
        np.round(ci_lower, 2),
        np.round(ci_upper, 2),
        np.round(t_stat, 2),
        '<0.001' if p_value < 0.001 else np.round(p_value, 3),
        df
    ]
}

# Convert results to a DataFrame
results_df = pd.DataFrame(regression_results)

# Save the table to a CSV file
output_file = os.path.join(directory, 'Results/Regression_Model_Baseline_EE.csv')
results_df.to_csv(output_file, index=False)
print('Regression results saved to Results/Regression_Model_Baseline_EE.csv')



# ---------------------- Energy Expenditure and Clinical Scores ----------------------



# Define the score and test date (e.g., UEMS; can be replaced it with other scores as needed)
score = 'UEMS'
test_date = 'Week_since_injury'

# P-values and results
p_values = []
correlation_result = []

# Aggregating score and EE
all_scores = []
all_energy_expenditure = []
Participant_IDs = []

# Ensure column names are properly formatted
energy_expenditure.columns.values[0] = 'Participant_ID'
energy_expenditure.columns = energy_expenditure.columns.str.strip()

for Participant in energy_expenditure['Participant_ID']:

    # Clinical scores
    clinical_data_Participant = clinical_data[clinical_data['Participant_ID'] == Participant][[score, test_date]]

    # Complete scores for all weeks
    weeks = list(range(1, 31))
    filled_clinical_data = pd.DataFrame({test_date: weeks})
    filled_clinical_data[score] = np.nan
    for _, row in clinical_data_Participant.iterrows():
        week = row[test_date]
        score_value = row[score]
        filled_clinical_data.loc[filled_clinical_data[test_date] == week, score] = score_value

    filled_clinical_data.update(filled_clinical_data.ffill())

    # Energy expenditure
    energy_expenditure_Participant = energy_expenditure[energy_expenditure['Participant_ID'] == Participant]
    
    # Add row of weeks
    week_row = {'Participant_ID': 'Week'}
    for i in range(54):
        week_row[f'Week {i}'] = i
    energy_expenditure_Participant = pd.concat([energy_expenditure_Participant, pd.DataFrame([week_row])], ignore_index=True)
    
    new_row = {'Participant_ID': energy_expenditure_Participant.iloc[0]['Participant_ID']}
    for i in range(54):
        new_row[f'Week {i}'] = np.nan

    # Loop through each row in clinical_data_Participant to find corresponding UEMS values
    for _, row in filled_clinical_data.iterrows():
        week = int(row[test_date])
        score_value = row[score]
        # Update the new row if the week column exists
        if f'Week {week}' in new_row:
            new_row[f'Week {week}'] = score_value

    # Append the new row to the DataFrame
    energy_expenditure_Participant = pd.concat([energy_expenditure_Participant, pd.DataFrame([new_row])], ignore_index=True)

    energy_expenditure_Participant = energy_expenditure_Participant.iloc[:, 1:].dropna(axis=1).T
    energy_expenditure_Participant.columns = ['EE', 'Week', score]

    # Aggregating data
    all_scores.extend(energy_expenditure_Participant[score])
    all_energy_expenditure.extend(energy_expenditure_Participant['EE'])
    Participant_IDs.extend([Participant] * len(energy_expenditure_Participant[score]))

# Create a DataFrame with the combined data
df = pd.DataFrame({
    'Score': all_scores,
    'EE': all_energy_expenditure,
    'Participant': Participant_IDs
})

# Calculate Spearman's rank correlation
r, p_value = spearmanr(all_scores, all_energy_expenditure)

# Store the result and the p-value for later correction
correlation_result = (score, r)

# Logistic fit
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Assuming all_scores and all_energy_expenditure are non-empty
if all_scores and all_energy_expenditure:
    # Convert lists to arrays for processing
    all_scores_array = np.array(all_scores, dtype=float)
    all_energy_expenditure_array = np.array(all_energy_expenditure, dtype=float)

    # Fit the logistic model
    initial_guesses = [max(all_energy_expenditure_array), 1, np.median(all_scores_array)]
    params, cov = curve_fit(logistic, all_scores_array, all_energy_expenditure_array, p0=initial_guesses, maxfev=10000)

    # Plotting
    x_trend = np.linspace(all_scores_array.min(), all_scores_array.max(), 100)
    y_trend = logistic(x_trend, *params)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_scores_array, all_energy_expenditure_array, s=20, label='Sensor measurement', color='gray', alpha=0.6)
    plt.plot(x_trend, y_trend, 'k-', label='Trend line', linewidth=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(score, size=22)
    plt.ylabel('Energy expenditure [kcal/day]', size=22)
    plt.grid(True)

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(directory + '/Figures/' + score + '.png', dpi=300)
    plt.close()

# Applying the Benjamini-Hochberg correction
reject, corrected_p_value, _, _ = multipletests(p_value, alpha=0.05, method='bonferroni')

# Print results with corrected p-values
score_r, corrected_p = correlation_result, corrected_p_value[0]
if corrected_p < 0.001:
    corrected_p_str = '<0.001'
else:
    corrected_p_str = f'{np.round(corrected_p, decimals=3)}'

if print_results:
    print(f'{score} - Spearman correlation coefficient: {np.round(score_r[1], decimals=3)}, Corrected p-value: {corrected_p_str}')

    # Calculate degrees of freedom for Spearman correlation
    n = len(all_energy_expenditure)  # Number of valid pairs
    spearman_df = n - 2
    print(f'Spearman degrees of freedom: {spearman_df}')
