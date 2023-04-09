# pattern-recognition-algorithm
Repository for multiple pattern recognition algorithm for heart disease prediction.

This work was developed as a project for final evaluation of Pattern Recognition course for Biomedical Engineering Master Degree at University of Coimbra.

Authors: Beatriz Negromonte (negromontebs@gmail.com), Rafael Molter (rafaelmolter@gmail.com) and Yandra Vinturini (yandravinturini@gmail.com)

## Goal:
Given dataset [heart_2020_cleaned](https://github.com/blackmountainb/pattern-recognition-algorithm/blob/main/heart_2020_cleaned.csv) the goal was to detect patterns from the data that can predict a patient’s condition.

## Dataset Description:
Source: [Personal Key Indicators or Heart Disease, on Kaggle](https://www.kaggle.com/kamilpytlak/personal-key-indicators-of-heart-disease)

The original dataset comes from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents.
BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted
health survey system in the world.". 
The most recent dataset (as of February 15, 2022) includes data from 2020. The vast majority of columns are questions asked to respondents about their health status, such as "Do you have serious difficulty walking or climbing stairs?" or "Have you smoked at least 100 cigarettes in your entire life?". 

The provided dataset has already been filtered containing a total of 318958 entries with 4 dependent
variables and 15 independent variable. The first 4 columns are binary and correspond to the following
dependent variables

1. CoronaryHeartDisease, 18085 positive cases

2. MyocardialInfarction, 16861 positive cases

3. KidneyDisease, 11672 positive cases

4. SkinCancer, 29676 positive cases

The rest of the information includes:
<div>
  <img src="https://github.com/blackmountainb/pattern-recognition-algorithm/blob/main/Dataset%20information.png" alt="Dataset information" width="700"/>&nbsp;
</div>
Figure: Information of the dataset.

In this project, it was proposed the use of two derived variables that should be calculated accordingly:

1. HeartDisease, 26536 positive cases. Positive when reported CoronaryHeartDisease OR MyocardialInfarction;

2. HeartDiseaseComorbities, 19111 with Heart Disease but no Comorbities, 7425 with Heart Disease and Comorbities and 292422 with no Heart Disease. Comorbidities are considered if reported KidneyDisease OR SkinCancer.

## Objective:
Our task was to develop classifiers for Heart Disease, considering three scenarios:

• Scenario A (Coronary Heart Disease): where a single classifier should be used to predict if a
patient has Coronary Heart Disease;

• Scenario B (Heart Disease): where a classifier, is designed to distinguish if a patient has Heart
Disease (either CoronaryHeartDisease or MyocardialInfarction);

• Scenario C (Heart Disease with comorbidities): where classifiers should be designed to distinguish three classes: HeartDiseaseNoComorbidities, HeartDiseaseComorbidities, and NoHeartDisease.

## Steps taken:

Considering the task and approaches provided by the course, we imported data, performed feature selection, feature reduction and designed experiments. 

For feature reduction, we used PCA and LDA. 
The algorithms we tested and compared results were Minimum Distance Classifier (MDC, using Euclidean and Mahalanobis classifiers), Fisher LDA, K-Nearest Neighbors, Naive Bayes and Support Vector Machines.

As per information from professors on the assignment: 
Some of the supplied features may be useless, redundant or highly correlated with others. In this phase, we should consider to use feature selection and dimensionality reduction techniques, and see how they affect the performance of the pattern recognition algorithms. Analyse the distribution of the values of your features and compute the correlation between them.

Last, we created a graphical user interface (GUI) for presentation.

The GUI should improve the interaction of the user with the code by providing options for data-loading, feature selection/dimensionality
reduction, classification, post-processing, validation and visualization.

## Conclusions and overall results:

We were able to conclude the proposed work of implementing different classification models for the three different scenarios A, B and C. Overall, the results were reasonable and have potential to be improved with a more thorough pipeline and data pre-processing, as well as tuning more parameters, although that would require a greater computational power.  

Considering the scenarios A, B and C presented on Objective:

Scenario A - Good distinction between positive and negative cases, with F1 score above 0.71 for each classifier. Best option: 10 best features and 8 components for PCA, using KNN classifier.

Scenario B - Poor distinction between the two heart conditions on the sample, the algorithm classifies more than half of Myocardial Infarction cases as Coronary Heart Diseases, in all predictors. 

Scenario C - Worst scenario, the only classifier able to separate classes well was KNN.


The detailed choices, justification and information about what was performed on the project is explained on [Report](https://github.com/blackmountainb/pattern-recognition-algorithm/blob/main/Report.pdf) and more performance metrics graphics are shown on [Presentation](https://github.com/blackmountainb/pattern-recognition-algorithm/blob/main/Presentation.pdf). 
