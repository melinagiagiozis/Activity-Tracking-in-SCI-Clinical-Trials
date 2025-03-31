# Feasibility and sensitivity of wearable sensors for daily activity monitoring in spinal cord injury trials
The aim of this project is to evaluate the effectiveness of continuous sensor-derived activity metrics as a method to monitor physical activity and recovery in patients with acute traumatic cervical spinal cord injury (SCI).

## Summary

🔍 Authors:

Authors: Melina Giagiozis, Irina Lerch, Anita D. Linke, Catherine R. Jutzeler, Rüdiger Rupp, Rainer Abel, Jesús Benito-Penalva, Josina Waldmann, Doris Maier, Michael Baumberger, Jiri Kriz, Andreas Badke, Norbert Weidner, László Demkó, Armin Curt on behalf of the Nogo Inhibition in Spinal Cord Injury Study Group

📝 Abstract: 

Background: The aim of clinical trials for spinal cord injury (SCI) is to improve everyday-life activity outcomes, which requires reliable methods for monitoring patient activity. This study evaluates sensor-derived activity metrics in comparison to established clinical assessment methods.

Methods: Wearable inertial sensors collected data from 69 individuals with acute traumatic cervical SCI participating in the Nogo-A Inhibition in Spinal Cord Injury (NISCI) trial ([NCT03935321](https://clinicaltrials.gov/study/NCT03935321)), a phase 2b multicenter, randomized, placebo-controlled trial. During inpatient rehabilitation, participants wore up to five inertial sensors for up to three consecutive days each week. An estimation of average daily energy expenditure (EE) was used as an indicator of physical activity and compared to the recovery of upper extremity motor scores (UEMS) and spinal cord independence measures (SCIM).

Results: Participants in the _verum_ (n = 41; 59.4%) and placebo (n = 28; 40.6%) groups showed similar initial activity levels, however, the _verum_ group exhibited a significantly greater weekly increase in average daily EE (delta EE = 11.6 kcal/d per week, 95% CI [1.5, 21.8], p = 0.025). In contrast, no significant group differences were observed in changes in UEMS (delta UEMS = 0.1 per week, 95% CI [–0.2, 0.3], p = 0.603) or SCIM (delta SCIM = 0.2, per week 95% CI [-0.7, 1.1], p = 0.644).

Conclusion: Continuous sensor-based activity monitoring offers objective and sensitive insights into changes in physical capabilities, effectively complementing periodic clinical assessments. Thus, sensors-derived outcome measures offer potential for improving the evaluation of clinical studies in individuals with SCI.


## Getting Started

First, clone this project to your local environment.

```sh
git clone https://github.com/melinagiagiozis/Activity-Tracking-in-SCI-Clinical-Trials.git
```
Create a virtual environment with python 3.9.13.

```sh
conda create --name activity_env python=3.9.13
conda activate activity_env
```

Install python dependencies.

```sh
pip install -r requirements.txt
```

## Path Setup

The paths for data, results, and figures based on the repo setup.

## Datasets Preparation

Ensure that the simulated datasets is in the `SimulatedData` folder.

## Data Analysis

To perform a general statistical analysis, run `Code/Statistical_Analysis_General.py`.
To analyze a potential treatment effect, run `Code/Statistical_Analysis_Treatment_Effect.py`.
To analyze a potential treatment effect stratified by injury completeness, run `Code/Statistical_Analysis_Injury_Completeness.py`.

## Contact

Questions or comments related to this repository or the manuscript:

📧 Melina Giagiozis (Melina.Giagiozis@balgrist.ch)

## Funding

This research was funded by the Swiss National Science Foundation (#PZ00P3_186101, Jutzeler and #IZLIZ3_200275, Curt), as well as grants from the EU program Horizon2020 (Grant agreement nr.681094 – NISCI), Wings for Life (Salzburg), the Swiss Paraplegic Foundation, and Wyss Zurich (University of Zurich and ETH Zurich).







