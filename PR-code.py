# -*- coding: utf-8 -*-
"""
#Final project

##Pattern Recognition - 2021/2022

## Due: May, 24th, 2022

 ***

**Variables documentation:**

CoronaryHeartDisease/MyocardialInfarction/KidneyDisease/SkinCancer: 2.0 negative / 1.0 positive


Smoking/Stroke/DiffWalking/Asthma/PhysicalActivity: 2.0 No / 1.0 Yes


AlcoholDrinking: 1.0 No / 2.0 Yes


Sex: 2.0 Female / 1.0 Male


AgeCategory: 

80 or older: 13

75 - 79: 12

70 - 74: 11

65 - 69: 10

60 - 64: 9

55 - 59: 8

50 - 54: 7

45 - 49: 6

40 - 44: 5

35 - 39: 4

30 - 34: 3

25 - 29: 2

18 - 24: 1


Race : Race: 1 White / 2 Black / 3 Asian / 4 American Indian/Alaskan Native / 5 Hispanic / 6 Other <p>

Diabetic : 1 Yes / 2 Yes, during pregnancy / 3 No / 4 No, borderline diabetes


General health: 1 Excellent / 2 Very good / 3 Good / 4 Fair / 5 Poor

Sleep Time: average hours of sleep in a 24-hour period.

### **Import libraries and read data**
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from statistics import mean, stdev
from matplotlib import cm
from google.colab import drive
from numpy.linalg import eig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, make_scorer, precision_score, recall_score
from math import sqrt, ceil, pi, exp
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy as sp
from numpy.linalg import inv
from numpy.random import seed
from scipy.stats import kstest
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from pingouin import kruskal
from sklearn.naive_bayes import GaussianNB

# As we were doing in a collaborative environment, we needed to mount:
# drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/Projeto - RP/heart_2020_cleaned.csv")

"""### **Functions definition**

###Functions - Data visualization
"""

def quantitative_data_plots(data_to_visualize, labels):

  n_rows = data_to_visualize.shape[1]
  n_columns = 2
  fig, axis = plt.subplots(n_rows, n_columns, figsize = (7*n_columns,6*n_rows), tight_layout = True)
  
  for i, column in enumerate(data_to_visualize):
    n_bins = ceil(data_to_visualize[column].max() - data_to_visualize[column].min()) + 1
    axis[i,0].hist(data_to_visualize[column], bins = n_bins, log=True)   # logarithmic scale to visualize outliers
    axis[i,0].set_title(column, fontsize=15)
    axis[i,1].set_title(column, fontsize=15)
    axis[i,0].set_xlabel(labels[i])
    axis[i,0].set_ylabel("Number of people")
    mean = data_to_visualize[column].mean()
    std = data_to_visualize[column].std()
    CV = std/mean # coefficient of variation
    txt = 'Mean = {:.2f} \nSTD = {:.2f} \nCV = {:.2f}'.format(mean, std, CV)
    axis[i,0].text(0.6,0.7, txt, transform = axis[i,0].transAxes, fontsize=12)
    results = axis[i,1].boxplot(data_to_visualize[column])
    axis[i,1].set_ylabel(labels[i])

  plt.plot()

def pie_plots(data_to_visualize, n_rows, n_columns):

  fig, axis = plt.subplots(n_rows, n_columns, figsize = (6*n_columns,6*n_rows), tight_layout = True)

  if(n_rows>1 or n_columns>1):
    for i, column in enumerate(data_to_visualize):

      X  = np.sort(pd.unique(data_to_visualize[column]))
      hight = data_to_visualize[column].value_counts()[X]
      axis.flat[i].pie(hight,labels=X, autopct = '%1.2f%%')
      axis.flat[i].set_title(column,fontsize=15)
  
  else: # for pandas Series
    X  = np.sort(pd.unique(data_to_visualize))
    hight = data_to_visualize.value_counts()[X]
    axis.pie(hight,labels=X, autopct = '%1.2f%%')
    axis.set_title(data_to_visualize.name,fontsize=15)

def make_corr(data, txt, features):

  df = pd.DataFrame(data, columns = features)
  matrix = df.corr() # creates correlation matrix
  plt.figure(figsize = (18,7))
  sns.heatmap(matrix, annot=True, vmin=-0.7, vmax=0.7, center = 0, cmap='coolwarm')
  plt.title(txt)
  plt.show()

def pca_analysis(data, components, seed):

  pca = PCA(n_components = components, random_state=seed)
  pca.fit(data)
  var_ratio = pca.explained_variance_ratio_
  cum_var = np.cumsum(np.round(var_ratio, decimals=3)*100) #cumulative sum of variance explained with [n] features
  print('Cumulative variance ratio: \n')

  for i, value in enumerate(cum_var):
    print(i+1,'components: ', value)

  plt.figure()
  x = np.arange(components)+1
  plt.plot(x,cum_var)
  plt.ylabel('% Variance Explained')
  plt.xlabel('Number of components')
  plt.title('PCA Analysis')
  plt.ylim(cum_var[0],100.5)
  plt.xticks(x)
  plt.grid()
  plt.show()

def scatter_plot(X, Y, colors, target, legend, n_rows, n_columns, titles):
  fig, ax = plt.subplots(n_rows, n_columns, figsize = (7*n_columns, 3*n_rows), tight_layout = True)

  for i, vec in enumerate(X):
    for color, category, name in zip(colors, target, legend):
      x = vec[Y[i]==category]
      y = np.ones_like(x)*category
      ax.flat[i].scatter(x,y, edgecolors=color, label=name, facecolors='none')
    
    ax.flat[i].set_title(titles[i])
    ax.flat[i].set_xlabel('LDA component 1')
    ax.flat[i].legend()
    ax.flat[i].set_yticks(target)
  plt.show()

"""### Functions - Pre processing"""

def scenario_options(option, data, feature_choice):
  columns = ['CoronaryHeartDisease','MyocardialInfarction','KidneyDisease','SkinCancer', 'Smoking', 'Stroke', 'DiffWalking', 'Asthma', 'PhysicalActivity']
  data[columns] = data[columns].replace([2.0],0).astype(int)
  data['AlcoholDrinking'] = data['AlcoholDrinking'].replace([1.0, 2.0], [0, 1]).astype(int)
  
  # for sex: 0 for male and 1 to female
  data['Sex'] = data['Sex'].replace([1.0, 2.0], [0, 1]).astype(int)
  data[['Race', 'Diabetic', 'GenHealth', 'AgeCategory']] = data[['Race', 'Diabetic', 'GenHealth', 'AgeCategory']].astype(int)
  data['Classification'] = data['CoronaryHeartDisease']
  columns = []

  
  if option == 'A':
    data = data.drop(["MyocardialInfarction", "KidneyDisease","SkinCancer"], axis=1)

  elif option == 'B':
    data = data.drop(["KidneyDisease","SkinCancer"], axis=1)
    data = data.where((data.MyocardialInfarction == 1) | (data.CoronaryHeartDisease == 1)).dropna(how="all") #get the samples with any heart disease
    data = data.mask((data.MyocardialInfarction == 1) & (data.CoronaryHeartDisease == 1)).dropna(how="all")  # remove samples that have both heart diseases
    
  elif option == 'C':
    # Changes the classification value to 1 if has heart diseases but no comorbities
    data.loc[
        (((data['MyocardialInfarction'] == 1) | (data['CoronaryHeartDisease'] == 1)) & 
        ((data['KidneyDisease'] == 0) & (data['SkinCancer'] == 0))),
        'Classification'
    ] = 1
    # Changes the classification value to 2 if has heart diseases and comorbities
    data.loc[
        (((data['MyocardialInfarction'] == 1) | (data['CoronaryHeartDisease'] == 1)) & 
        ((data['KidneyDisease'] == 1) | (data['SkinCancer'] == 1))),
        'Classification'
    ] = 2

  feature_choice.append('Classification')
  
  if 'All' in feature_choice:
    data = data
  else:
    for i in data:
      if i not in feature_choice:
        data = data.drop(i, axis = 1)

  return data

def kruskal_fs_backup(data_scenario, target):
  # returns kw_ds = [name_feature, ddof1, H, p-unc]
  kw = []
  for column in data_scenario.iloc[:,2:-1]:
    kw.append(kruskal(data = data_scenario, dv=target, between=column))
  kw_ds = pd.concat(kw, axis=0)
  kw_ds = kw_ds.sort_values(by=['H'], ascending = False)
  #kw_ds = kw_ds.to_numpy()
  return kw_ds

def kruskal_fs(data_scenario, target, scenario):
  kw = []
  if (scenario == 'A'):
    for column in data_scenario.iloc[:,1:-1]:
      kw.append(kruskal(data = data_scenario, dv=target, between=column))
  if (scenario == 'B'):
    for column in data_scenario.iloc[:,2:-1]:
      kw.append(kruskal(data = data_scenario, dv=target, between=column))
  if scenario == 'C':
    for column in data_scenario.iloc[:,4:-1]:
      kw.append(kruskal(data = data_scenario, dv=target, between=column))
  
  kw_ds = pd.concat(kw, axis=0)
  kw_ds = kw_ds.sort_values(by=['H'], ascending = False)

  return kw_ds

def feature_selection_function(data_X, target_label, scenario):
  all_features = kruskal_fs(data_X, target_label, scenario)
  print(all_features)
  print(f"\n")
  n_features = int(input('Define the number of features:'))
  all_features = all_features.to_numpy()
  list_features = []
  for i in range(0, n_features):
    list_features.append(all_features[i][0])

  return list_features

def drop_function(data, scenario, feature_selected):
  drop_labels = []
  dep_labels = ['CoronaryHeartDisease', 'MyocardialInfarction', 'KidneyDisease', 'SkinCancer','Classification']
  
  for columns in data:
    if columns in dep_labels:
      drop_labels.append(columns)
    elif columns not in feature_selected:
      drop_labels.append(columns)

  for label in drop_labels:
    data = data.drop([label], axis = 1)

  return data

def resample_function(data_X, option, resample, feature_selected):
  
  data_X_labels = data_X['Classification']
  data_X = drop_function(data_X, option, feature_selected)
  
  if resample == 'Yes':
    if option == 'A':
      over = RandomOverSampler(sampling_strategy=0.8, random_state=0) #oversample minority class to 80% of majority class
      under = RandomUnderSampler(sampling_strategy=1, random_state=0) #undersample majority class so minority class is 100% of it (same number of samples)
      
      x_oversampled, y_oversampled = over.fit_resample(data_X, data_X_labels)
      data_X, data_X_labels = under.fit_resample(x_oversampled, y_oversampled)

    elif option == 'C':
      n_majority = sum(data_X_labels==0)
      n_minority = ceil(0.8*n_majority) #oversample minority classes to 80% of majority class

      over = RandomOverSampler(sampling_strategy={0: n_majority, 1: n_minority, 2: n_minority}, random_state=0) 
      x_oversampled, y_oversampled = over.fit_resample(data_X, data_X_labels)
      under = RandomUnderSampler(sampling_strategy='majority', random_state=0) #undersample majority class to same number of other classes

      data_X, data_X_labels = under.fit_resample(x_oversampled, y_oversampled)
  
  elif resample == 'No':
    data_X = drop_function(data_X, option)

  return data_X, data_X_labels

def scaling_function(data_X, labels):
  test = 0.2
  seed = 2
  scaler = StandardScaler()
  
  x_train, x_test, y_train, y_test = train_test_split(data_X, labels, random_state = seed, test_size = test, stratify = labels)
  x_train = scaler.fit_transform(x_train)  # fit scalers to train data and transform
  x_test = scaler.transform(x_test)  # transform test data with fitted scalers from train sets

  return x_train, x_test, y_train, y_test

def pca_transform(train_data, test_data, seed, lenght):
  components = (lenght + 1)
  while components > lenght:
    components = int(input('How many components? '))
    print(f"\n")
  pca = PCA(n_components = components, random_state=seed)
  reduc_train = pca.fit_transform(train_data)
  reduc_test = pca.transform(test_data)

  return reduc_train, reduc_test

def lda_transform(train_dataX, test_dataX, train_dataY, solver):
  
  lda = LDA(solver=solver)
  reduc_train = lda.fit_transform(train_dataX, train_dataY)
  reduc_test = lda.transform(test_dataX)

  return lda, reduc_train, reduc_test

def pca_or_lda(choice, x_train, x_test, y_train, seed, list_features):
  lenght = len(list_features)
  if choice == 'PCA':
    x_train_final, x_test_final = pca_transform(x_train, x_test, seed, lenght)
    lda = LDA().fit(x_train_final, y_train)

  elif choice == 'LDA':
    lda_solver = 'eigen'
    lda, x_train_final, x_test_final = lda_transform(x_train, x_test, y_train, lda_solver)

  return lda, x_train_final, x_test_final

"""###Functions - Metrics"""

def get_metrics(y_true, y_pred, title):

  cmatrix = confusion_matrix(y_true, y_pred)

  ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=[0,1]).plot()
  plt.title(title)
  plt.show()

  TP = cmatrix[1,1]
  TN = cmatrix[0,0]
  FP = cmatrix[0,1]
  FN = cmatrix[1,0]

  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  precision = TP/(TP + FP)
  f1 = f1_score(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)

  print(f"\nAccuracy: {100*accuracy:.2f} %\nSensitivity: {100*sensitivity:.2f} %\nSpecificity: {100*specificity:.2f} %\nPrecision: {100*precision:.2f} %\nF1 score: {f1:.2f}\n")

  #return accuracy, sensitivity, specificity, precision, f1

def get_metrics_multiclass(y_true, y_predict,labels):

  cm = confusion_matrix(y_true, y_predict)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot()
  plt.show()

  accuracy = accuracy_score(y_true, y_predict)
  f1 = f1_score(y_true, y_predict, average='weighted',labels=labels)
  precision = precision_score(y_true, y_predict, average='weighted',labels=labels)
  sensitivity = recall_score(y_true, y_predict, average='weighted',labels=labels)
  print(f"\nAccuracy: {100*accuracy:.2f} %\nSensitivity weighted: {100*sensitivity:.2f} %\nPrecision weighted: {100*precision:.2f} %\nF1 score weighted: {f1:.2f}\n")

  f1 = f1_score(y_true, y_predict, average='micro',labels=labels)
  precision = precision_score(y_true, y_predict, average='micro',labels=labels)
  sensitivity = recall_score(y_true, y_predict, average='micro',labels=labels)
  print(f"Sensitivity micro: {100*sensitivity:.2f} %\nPrecision micro: {100*precision:.2f} %\nF1 score micro: {f1:.2f}\n")

def get_cross_validation(model, X, y):

  scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

  scores = cross_validate(model, X, y, cv=5, scoring=scoring)

  print(f"Accuracy mean: {100*scores['test_accuracy'].mean():.2f} %\nAccuracy standard-deviations: {100*scores['test_accuracy'].std():.2f} %\nPrecision mean: {100*scores['test_precision'].mean():.2f} %\nPrecision standard deviation: {100*scores['test_precision'].std():.2f} %\nRecall mean:  {100*scores['test_recall'].mean():.2f} %\nRecall standard deviation:  {100*scores['test_recall'].std():.2f} %\nF1 score mean:  {100*scores['test_f1_score'].mean():.2f} %\nF1 score standard deviation:  {100*scores['test_f1_score'].std():.2f}" )

"""###Functions - Methods"""

def get_mean_std(data_X, labels):
  means = []
  std = []
  for n in range(labels.nunique()):      
    means.append(np.mean(data_X[labels==n], axis=0))
    std.append(np.std(data_X[labels==n], axis = 0))
  return means, std

def get_methods(scenario):
  if (scenario == 'A') | (scenario == 'B'):
    list_methods = ['Euclidean', 'Mahalanobis', 'FisherLDA', 'NaiveBayes','KNN', 'SVM']
  if (scenario == 'C'):
    list_methods = ['NaiveBayes','KNN', 'SVM']
  return list_methods

def euclidean(data_X, means):  

  sum_first = 0
  sum_second = 0

  all_dist = []
  labels = []
  for d in range(len(data_X)):
    dist = []
    for label in range(len(means)): 
      mean_data = list(means[label]*data_X[d]) 
      sum_first = np.sum(mean_data)
      multiple = list(means[label]*means[label])
      sum_second = np.sum(multiple)
      dist.append(sum_first - 0.5*sum_second)

    max_distance = max(dist)
    all_dist.append(max_distance)
    max_distance_index = dist.index(max_distance)
    labels.append([max_distance_index])

  return labels

def mahalanobis(dataX, dataY, means):

  c0 = np.cov(dataX[dataY==0], rowvar=False)
  c1 = np.cov(dataX[dataY==1], rowvar=False)
  C = (c0 + c1)/2
  labels = []

  if(isinstance(C, np.ndarray)):  
    C = inv(C)
  
  for x in dataX:
    inv_mean_prod = np.dot((means[0] - means[1]).T, C)
    d01 = np.dot(inv_mean_prod, (x - 0.5*(means[0]+means[1])))
    if d01>0:
      labels.append(0)
    else:
      labels.append(1)

  return labels

def pdf(x, mean, std):
  # Probability density function, estimates the probability of a given value
  e = exp(-((x-mean)**2 / (2*std**2)))
  fx = (1/(sqrt(2*pi) * std)) * e
  return fx

def probabilities_for_each_row(data_X, data_X_row, labels, means, std):
  total_rows = len(data_X)
  labels.to_numpy()
  prob = dict()
  # counting labels:
  cont = [0, 0]
  for i in labels:
    if i==float(0):
      cont[0] += 1
    elif i==float(1):
      cont[1] += 1
  for label in range(len(means)):
    prob[label] = cont[label]/total_rows
    for feature in range(len(means[label])):
      mean = means[label][feature]
      st = std[label][feature]
      prob[label] *= pdf(data_X_row[feature], mean, st)
  return prob

def naiveBayes(data_X, labels, means, std):
  # Classifies the row where it has bigger probability to belong. If the
  # probabilities are equal, checks for the one that has less classified items
  predict = []
  cont = [0, 0]
  for row in range(len(data_X)):
    r = data_X[row]
    prob = probabilities_for_each_row(data_X, r, labels, means, std)
    if prob[0]>prob[1]:
      predict.append([0])
    elif prob[1]>prob[0]:
      predict.append([1])
    else:
      for i in predict:
        if i==float(0):
          cont[0] += 1
        elif i==float(1):
          cont[1] += 1
      if cont[0]>cont[1]:
        predict.append([1])
      else:
        predict.append([0])
  return predict

def knn(k, X_train_f, X_test_f, Y_train_f):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(X_train_f, Y_train_f)
  predict = classifier.predict(X_test_f)
  return predict

def svm_classifier(kernel_function, X_train, y_train, X_test):
  if(kernel_function == 'linear'):
    param_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
    SVM_model = LinearSVC(dual=False).fit(X_train, y_train)
  if(kernel_function == 'rbf' ):
    param_grid = [{'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}]
    SVM_model = SVC(kernel='rbf').fit(X_train, y_train)

  grid = GridSearchCV(estimator=SVM_model, param_grid=param_grid, scoring="accuracy", cv=5)
  grid_result = grid.fit(X_train, y_train)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']

  SVM_model = grid_result.best_estimator_
  SVM_pred = SVM_model.predict(X_test)
  return SVM_pred

def prediction(choice, scenario, x_train, x_test, y_train, y_test):
  metrics = get_mean_std(x_test, y_test)
  means = metrics[0]
  std = metrics[1]
  if choice == 'Euclidean':
    prediction = euclidean(x_test, means)
    
  elif choice == 'Mahalanobis':
    prediction = mahalanobis(x_test, y_test, means)

  elif choice == 'FisherLDA':
    lda = LDA().fit(x_train, y_train)
    prediction = lda.predict(x_test)

  elif (choice == 'NaiveBayes') & ((scenario == 'B') | (scenario == 'A')):
    prediction = naiveBayes(x_test, y_test, means, std)

  elif (choice == 'NaiveBayes') & (scenario == 'C'):
    gnb = GaussianNB().fit(x_train, y_train)
    prediction = gnb.predict(x_test)
    
  elif choice == 'KNN':
    k = int(input('Choose "k" value:'))
    prediction = knn(k, x_train, x_test, y_train)
    
  elif choice == 'SVM':
    prediction = svm_classifier('linear', x_train, y_train, x_test)

  if (scenario == 'A') | (scenario == 'B'):
    title = 'Test results for '+choice+' method'
    #metrics = get_metrics(y_test, prediction, title)
    get_metrics(y_test, prediction, title)

  elif (scenario == 'C'):
    get_metrics_multiclass(y_test, prediction, y_test.unique())

"""### **Data Cleaning and Subsets**

Changing labels 1.0 and 2.0 to 0 (negative) and 1 (positive) (for better reading and understanding)

Qualitative data changed from float to integer.

Subset A: drop the unwanted columns for Scenario A - MyocardialInfarction, KidneyDisease and SkinCancer - leaving only CoronaryHeartDisease. This scenario classifies will distinguish if patient has the condition or not.


Subset B: drop the unwanted columns for Scenario B, the ones where MyocardialInfarctian and CoronaryHeartDisease are both 0 or both 1.
We will classify between one class or the other. 

As they are complementary, we will use MyocardialInfarction as the principal label, therefore, when we receive 1 as result, it's MyocardialInfarction and when the classification results in 0, is CoronaryHeartDisease.

Subset C: there are no columns dropped for scenario C, only the classification one that is added and has 3 classes: 0, 1 and 2, where the first corresponds to patients that are negative for heart conditions and 1 and 2 are patients that are positive for heart conditions and don't have comorbities and have comorbities, respectively. 
"""

data = pd.read_csv("/content/drive/MyDrive/Projeto - RP/heart_2020_cleaned.csv")
scenario = 'A'
feature_choice = ['All']
data_X = scenario_options(scenario, data, feature_choice)
data_X

"""Checking if there's any missing data and selecting the labels: """

data_X.count(), data_X.shape

data_X_labels = data_X['Classification']

data_X['Classification'].value_counts()

"""###Feature Selection

Kruskal Wallis:
"""

list_features = feature_selection_function(data_X, 'Classification', scenario)
print(list_features)

"""### **Quantitative data visualization:**

Histogram and boxplots:
"""

quantitative_data = ['_BMI5', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
labels = ['kg/m^2', 'Days', 'Days', 'Hours']

quantitative_data_plots(data_X[quantitative_data], labels)

"""The standard deviation of both Physical and Mental Health are high, which is determined by the coefficient of variation above 1.

### **Resampling**

Distribution of labels before balancing:
"""

data_X_labels.value_counts()
pie_plots(data_X_labels, 1, 1)

"""Distribution of labels after balancing:"""

resample_data = 'Yes'
resample = resample_function(data_X, scenario, resample_data, list_features)
data_X = resample[0]
data_X_labels = resample[1]
data_X

print(data_X_labels.value_counts())
pie_plots(data_X_labels, 1, 1)

"""### **Dimensionality reduction:**
###Scaling:

For both PCA and LDA, it is required to standardize the data. The datasets are first divided into train and test to avoid data leakage
"""

scaling = scaling_function(data_X, data_X_labels)
x_train = scaling[0]
x_test = scaling[1]
y_train = scaling[2]
y_test = scaling[3]

"""To analyse the features correlation:"""

features = data_X.columns
make_corr(x_train, 'Standardized data', features)

"""Negative correlations are shown in blue, whilst positive correlation is shown in red. The most uncorrelated features are shown in gray.

#### Principal Component Analysis or Linear Discriminant Analysis

Visualizing PCA for the chosen features:
"""

n_components = len(list_features)
pca_analysis(x_train, n_components, seed)

"""Choosing between PCA or LDA:"""

seed = 2
choice = 'PCA'
pca_lda = pca_or_lda(choice, x_train, x_test, y_train, seed, list_features)

lda = pca_lda[0]
x_train = pca_lda[1]
x_test = pca_lda[2]

"""These results will be further used on our classifier to predict our classes

### **Classifiers**
"""

methods = get_methods(scenario)
print(methods)

choice = 'KNN'
method = prediction(choice, scenario, x_train, x_test, y_train, y_test)

"""###**GUI definition**

GUI options:
Scenario | LDA/PCA | # of features | Method

Scenario options
"""

opt = ['A', 'B', 'C']
y_n = ['Yes', 'No']
scaling_options = ['PCA','LDA']
methods_options = ['LDA Fisher','Euclidean', 'Mahalanobis', 'Naive Bayes', 'KNN', 'SVM']
options_features = ['All']
dep_labels = ['CoronaryHeartDisease', 'MyocardialInfarction', 'KidneyDisease', 'SkinCancer']
for i in data:
  if i in dep_labels:
    continue
  else:
    options_features.append(i)

output = widgets.Output()

dropdown_scenarios = widgets.Dropdown(options = opt, description = 'Scenario')
#dropdown_features = widgets.SelectMultiple(options = options_features, description = 'Features', disabled = False)
dropdown_balance = widgets.Dropdown(options = y_n, description = 'Balanced?')
dropdown_pca_lda = widgets.Dropdown(options = scaling_options, description = 'PCA or LDA?')

slider_components = widgets.IntSlider(min = 1, max = 15, step = 1, description = '# of components')

dropdown_methods = widgets.Dropdown(options = methods_options)

def common_filtering(scenario, feature_selection, balance, components, pca_lda):
  output.clear_output()
  #feature_selection = list(feature_selection)
  d = scenario_options(scenario, data_X, feature_selection)
  
  with output:
    display(print(d))

def dropdown_scenarios_change(change):
  common_filtering(change.new, dropdown_features.value, dropdown_balance.value, slider_components.value, dropdown_pca_lda.value)

def dropdown_features_change(change):
  common_filtering(dropdown_scenarios.value, change.new, dropdown_balance.value, slider_components.value, dropdown_pca_lda.value)

def dropdown_balance_change(change):
  common_filtering(dropdown_scenarios.value, dropdown_features.value, change.new, slider_components.value, dropdown_pca_lda.value)

def slider_components_change(change):
  common_filtering(dropdown_scenarios.value, dropdown_features.value, dropdown_balance.value, change.new, dropdown_pca_lda.value)

def dropdown_pca_lda_change(change):
  common_filtering(dropdown_scenarios.value, dropdown_features.value, dropdown_balance.value, slider_components.value, change.new)

dropdown_scenarios.observe(dropdown_scenarios_change, names = 'value')
#dropdown_features.observe(dropdown_features_change, names = 'value')
dropdown_balance.observe(dropdown_balance_change, names = 'value')
slider_components.observe(slider_components_change, names = 'value')
dropdown_pca_lda.observe(dropdown_pca_lda_change, names = 'value')

display(dropdown_scenarios)
#display(dropdown_features)
display(dropdown_balance)
display(slider_components)
display(dropdown_pca_lda)

display(output)