#!/usr/bin/env python
# coding: utf-8

# <center>Copyright by Pierian Data Inc.
# For more information, visit us at www.pieriandata.com

# ## <center>Churn Analysis and Predictive Modeling for an Internet and Telephone Provider

# ### Part I: Data Checking

# <b> Step #1: Import all of the libraries we need in this project </b>

# In[128]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <b>Step #2: Read the CSV file as our DataFrame </b>

# In[129]:


df = pd.read_csv('Telco-Customer-Churn.csv')


# In[130]:


df


# <b> Step #3: Use .info() and .isnull().sum() check whether we have any null values in the dataset. </b>

# In[131]:


df.info()


# In[132]:


df.isnull().sum()


# <i>Based on the output above, we can conclude that our dataset is already cleaned and does not contain any null values</i>

# <b>Step #4: Use .describe() to get a quick statistical summary of the numeric columns in the dataset </b>

# In[133]:


df.describe()


# <i>Based on the output above, we can conclude that our dataset contains a significant number of categorical columns. Therefore, it is advisable to convert these categorical variables into dummy variables</i>

# ### Part II:  Exploratory Data Analysis

# <b>Step #5: Use sns.countplot() to display the balance of the class labels (Churn) </b>

# In[134]:


plt.figure(figsize = (3, 3), dpi = 200)
sns.countplot(x = 'Churn', data = df)


# <b>Step #6: Use sns.violinplot() to explore the distribution of TotalCharges between Churn categories </b>

# In[135]:


plt.figure(figsize = (3, 3), dpi = 200)
sns.violinplot(x = 'Churn', y = 'TotalCharges', data = df)


# <b>Step #7: Use sns.boxplot() to show the the distribution of TotalCharges per Contract type </b>

# In[136]:


#CODE HERE
plt.figure(figsize = (4, 4), dpi = 200)
sns.boxplot(x = 'Contract', y = 'TotalCharges', 
            hue = 'Churn', data = df)
plt.legend(bbox_to_anchor = (1.05,0.5))


# <b>Step #8: Use pd.get_dummies() to convert categorical features into dummy variables. Use sns.barplot() to show the correlation of the following features to the class label
# <br><br>
# List of Features: <br>
# ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']<br></b>

# In[137]:


categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

df2 = pd.get_dummies(df, columns = categorical_columns)
df2


# In[138]:


df2['Churn_Yes'] = df2['Churn'].map({'No': 0, 'Yes': 1})
df2.corr(numeric_only = True)['Churn_Yes'].sort_values()


# In[139]:


churn_corr = df2.corr(numeric_only = True)['Churn_Yes']
churn_corr = churn_corr.drop(['tenure', 'Churn_Yes'])
corrs = pd.Series(index = churn_corr.index, data = churn_corr.values)
corrs = corrs.sort_values()

plt.figure(figsize = (8, 8), dpi = 200)
sns.barplot(x = corrs.index, y = corrs.values)
plt.xticks(rotation = 90);


# ### Part III: Churn Analysis

# <b>Step #9: Find the 3 contract types available using .unique()</b>

# In[140]:


df['Contract'].unique()


# <b>Step #10: Use sns.histplot() to create a histogram displaying the distribution of 'tenure' column, which is the amount of months a customer was or has been on a customer</b>

# In[141]:


plt.figure(figsize = (5, 5), dpi = 200)
sns.histplot(data=df, x="tenure", binwidth=2)


# <b>Step #11: Use sns.FacetGrid() to create histograms separated by two additional features, Churn and Contract</b>

# In[142]:


g = sns.FacetGrid(df, row="Churn", col="Contract")
g.map(sns.histplot, "tenure")
g.set_axis_labels("Tenure", "Count")
g.fig.suptitle("Histograms Separated by Additional Features")
plt.tight_layout()


# <b>Step #12: Use sns.scatterplot() to display a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn </b>

# In[143]:


#CODE HERE
df.columns


# In[144]:


plt.figure(figsize = (10, 5), dpi = 200)
sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="Churn", alpha = 0.5)


# <b>Step #13: Use .groupby() and others to calculate the Churn rate (percentage that had Yes Churn) per cohort. For example, the cohort that has had a tenure of 1 month should have a Churn rate of 61.99%.</b>

# In[145]:


churned_df = df[df['Churn'] == 'Yes']
total_count = df.groupby('tenure').size()
churned_count = churned_df.groupby('tenure').size()
percentage = (churned_count / total_count) * 100
new_df = pd.DataFrame({'Tenure': total_count.index, 'Churn Rate': percentage})
new_df = new_df.set_index('Tenure')


# In[146]:


new_df


# <b>Step #14: Use sns.lineplot() to create a plot showing churn rate per months of tenure</b>

# In[147]:


plt.figure(figsize = (10, 5), dpi = 200)
sns.lineplot(data=new_df, x="Tenure", y="Churn Rate")


# <b>Step #15: Use pd.cut() to create a new column called Tenure Cohort that creates 4 separate categories:<br><br>
# 0-12 Months'<br>
# '12-24 Months'<br>
# '24-48 Months'<br>
# 'Over 48 Months'<br></b>

# In[148]:


bins = [0, 12, 24, 48, 73]
labels = ['0-12 Months', '12-24 Months', '24-48 Months', 'Over 48 Months']
df['tenure_cohort'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
df[['tenure', 'tenure_cohort']]


# In[149]:


plt.figure(figsize = (10, 5), dpi = 200)
sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="tenure_cohort", alpha = 0.5)


# <b>Step #16: Use sns.countplot() to show the churn count per cohort</b>

# In[150]:


plt.figure(figsize = (10, 5), dpi = 200)
sns.countplot(data=df, x="tenure_cohort", hue="Churn")


# <b>Step #17: Use sns.catplot() to create a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the Churn hue</b>

# In[151]:


g = sns.catplot(data=df, x='tenure_cohort', col='Contract', hue='Churn', kind='count')

g.set_axis_labels('Tenure Cohort', 'Count')

g.fig.subplots_adjust(top=0.8)
g.fig.suptitle('Counts per Tenure Cohort, Separated by Contract Type and Colored by Churn');


# # Part IV: Predictive Modeling
# 
# ## Single Decision Tree

# <b>Step #18: separate out the data into X features and Y label </b>

# In[152]:


X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis = 1), drop_first = True)


# In[153]:


y = df['Churn']


# <b>Step #19: perform a train test split, holding out 10% of the data for testing </b>

# In[154]:


from sklearn.model_selection import train_test_split


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)


# <b>Step #20: Decision Tree Perfomance.<br><br>
#    Complete the following tasks:<br><br>
#    1. Train a single decision tree model (feel free to grid search for optimal hyperparameters).<br>
#    2. Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.<br>
#    2. Calculate feature importances from the decision tree.<br>
#    4. OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your notebook if you display it with plot_tree.

# In[156]:


from sklearn.tree import DecisionTreeClassifier


# In[157]:


from sklearn.model_selection import GridSearchCV


# In[158]:


criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
max_features = ['auto', 'sqrt', 'log2']

param_grid = {'criterion': criterion,
           'splitter': splitter,
           'max_features': max_features}


# In[160]:


model = DecisionTreeClassifier()


# In[161]:


grid = GridSearchCV(model, param_grid)


# In[162]:


grid.fit(X_train, y_train)


# In[163]:


grid.best_params_


# In[164]:


dtc = DecisionTreeClassifier(criterion = 'gini', max_features = 'log2',
                             splitter = 'random')


# In[165]:


dtc.fit(X_train, y_train)


# In[166]:


predictions = dtc.predict(X_test)


# In[167]:


from sklearn.metrics import classification_report, plot_confusion_matrix


# In[168]:


print(classification_report(y_test, predictions))


# In[169]:


plot_confusion_matrix(dtc, X_test, y_test);


# In[170]:


dtc.feature_importances_


# In[174]:


imp_feats.sort_values('Importance')


# In[175]:


feat = pd.DataFrame(index=X.columns, data=dtc.feature_importances_, columns=['Importance'])
imp_feats = feat[feat['Importance'] > 0]
plt.figure(figsize = (10, 5), dpi = 200)
sns.barplot(data = imp_feats, x=imp_feats.index, y='Importance')
plt.xticks(rotation = 90);


# ## Random Forest
# 
# <b>Step #21: create a Random Forest model and create a classification report and confusion matrix from its predicted results on the test set </b>

# In[179]:


from sklearn.ensemble import RandomForestClassifier


# In[180]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)


# In[181]:


n_estimators = [10, 25, 50, 75, 100]
criterion = ['gini', 'entropy', 'log_loss']
max_features = ['sqrt', 'log2']
bootstrap = [True, False]
oob_score = [True, False]

param_grid = {'n_estimators': n_estimators,
             'criterion': criterion,
             'max_features': max_features,
             'bootstrap': bootstrap,
             'oob_score': oob_score}


# In[182]:


model = RandomForestClassifier(verbose = True)


# In[183]:


grid = GridSearchCV(model, param_grid)


# In[184]:


grid.fit(X_train, y_train)


# In[185]:


grid.best_params_


# In[186]:


rfc = RandomForestClassifier(n_estimators = 100, criterion = 'gini',
                             max_features = 'sqrt',
                             bootstrap = True,
                             oob_score = True)


# In[187]:


rfc.fit(X_train, y_train)


# In[188]:


predictions = rfc.predict(X_test)


# In[189]:


print(classification_report(y_test, predictions))


# In[190]:


plot_confusion_matrix(rfc, X_test, y_test)


# ## Boosted Trees
# 
# <b>Step #22: use AdaBoost or Gradient Boosting to create a model and report back the classification report and plot a confusion matrix for its predicted results

# In[197]:


from sklearn.ensemble import AdaBoostClassifier


# In[198]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)


# In[199]:


model = AdaBoostClassifier(n_estimators = 10)


# In[200]:


model.fit(X_train, y_train)


# In[201]:


predictions = model.predict(X_test)


# In[202]:


print(classification_report(y_test, predictions))


# In[203]:


plot_confusion_matrix(model, X_test, y_test);


# <b>Step #23: Analyze the results, which is the best model performance?</b>

# <b><i>The AdaBoostClassifier model achieved the best performance for our machine learning project. It has an accuracy of 0.83, meaning it accurately predicts the outcome in 83% of cases. <br>
# The macro average score of 0.70 indicates good performance across different classes, and the weighted average score of 0.82 suggests effectiveness in handling imbalanced data. <br>
# Overall, the AdaBoostClassifier is the top-performing model for our project!</i></b>
