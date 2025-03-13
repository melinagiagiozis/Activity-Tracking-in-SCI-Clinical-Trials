# Evaluating spinal cord injury trials using wearable sensors for daily activity monitoring
The aim of this project is to evaluate the effectiveness of continuous, unobtrusive sensor-derived activity metrics as a method to monitor physical activity and recovery in patients with acute traumatic cervical spinal cord injury (SCI).

## Summary

🔍 Authors:

Authors: Melina Giagiozis, Irina Lerch, Anita D. Linke, Catherine R. Jutzeler, Rüdiger Rupp, Rainer Abel, Jesús Benito-Penalva, Josina Waldmann, Doris Maier, Björn Zörner, Jiri Kriz, Andreas Badke, Norbert Weidner, László Demkó, Armin Curt on behalf of the Nogo Inhibition in Spinal Cord Injury Study Group

📝 Abstract: 

Background: Clinical trials for spinal cord injury (SCI) require reliable methods to monitor patient activity and evaluate activity based outcomes. This study examines the role of continuous, unobtrusive sensor-derived activity metrics in comparison to traditional assessment methods.
Methods: Wearable inertial sensors were used to collected data from 69 participants with acute traumatic cervical SCI as part of the Nogo-A Inhibition in Spinal Cord Injury (NISCI) trial (NCT03935321), a multicenter phase II randomized, placebo-controlled trial. During inpatient rehabilitation, participants wore five inertial sensors for up to three consecutive days each week. An estimation of average daily energy expenditure (EE) was used as an indicator of physical activity.
Results: A significant increase in average daily EE was observed across all participants over the initial 30 weeks after injury (Linear Regression Model: t(65) = 5.8, slope = 21.3, 95% CI [14.1, 28.5], p <0.001). Participants in the verum group exhibited a more pronounced weekly increase in average daily activity levels compared to those in the  placebo group (Linear Mixed-Effects Model: delta EE = 11.7, 95% CI [1.8, 21.7], p = 0.021). While the Upper Extremity Motor Score (UEMS), the primary outcome measures of the trial, did not demonstrate a treatment effect, this metric successfully detected subtle, quantitative differences.
Conclusion: Unobtrusive, continuous sensor-based activity measurements provide complementary insights by detecting changes in physical and functional capabilities that periodic clinical assessments might overlook. Thus, wearable sensors offer potential for improving the evaluation of clinical studies in individuals with SCI.

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







