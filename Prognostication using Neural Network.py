#!/usr/bin/env python
# coding: utf-8

# ## **Data Science in Agriculture. Prognostication using Neural Network**
# 

# ## Abstract
# 

# This laboratory is designed to study forecasting based on BigData methods through the example of pesticide sales. We will learn how to make predictions based on linear regression and two frameworks for building neural networks, as well as obtain statistics and compare results.
# 

# ## Introduction
# 

# Nowadays, there is a lot of open data about agriculture statistics in the world. However, few tools are presented to predict and visualize these processes. This laboratory work will show how to download data from open sources, perform preliminary data analysis, transform and clear data, perform correlation.
# 
# Next we will consider 2 different mathematical approaches to the calculation of a forecast based on linear regression.
# 
# To do this, the division of the DataSet into training and test sets will be demonstrated. It will be shown how to build models using 2 different frameworks. Then we will build a forecast and analyze the accuracy and adequacy of the obtained models.
# 
# Then two different Neuro Networks will be build and fitted. We will study how to normalize and inverse normalize data. At the end we will compare all the results.
# 

# ### Downloading data
# 

# Some libraries should be imported before you can begin.
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("data_agric.csv")
df.drop(['Unnamed: 0','level_0','index'],axis=1,inplace=True)
df


# As you can see, the DataSet consists of 1675 rows  and 7 columns. Let's study the columns.
# 

# In[3]:


col = df.columns
col


# In this lab, we will try to build a sales forecast of one pesticide type.
# As you can see from the table, we don't need to use all the columns to make this forecast because some columns contain similar data. For forecasting we need to know only the data from columns: 'pesticid', 'geo', 'TIME_PERIOD', 'OBS_VALUE'.
# 
# Let's select the necessary columns.
# 

# In[4]:


df =df[['pesticid', 'geo', 'TIME_PERIOD', 'OBS_VALUE']]
df


# In[5]:


df.info()


# As you can see, two last columns were recognized correctly (int64 and float64). First 2 columns were recognized as objects. Let's investigate them:
# 

# ### Сhanging the data and data types of columns
# 

# In[28]:


df['TIME_PERIOD']=df['TIME_PERIOD'].astype('object')


# In[29]:


col = ['pesticid', 'geo']
df.loc[:, col] = df.astype('category')
df[col].describe()


# In[7]:


df['pesticid'].cat.categories


# In[8]:


df['geo'].unique()


# In[9]:


df.loc[:, 'geo'] = df['geo'].cat.add_categories(["GB", "GR"])


# Then we should change the values using a binary mask:
# 

# In[10]:


pd.options.mode.chained_assignment = None  # swich of the warnings
mask = df['geo'] == 'UK' # Binary mask
df.loc[mask, 'geo'] = "GB" # Change the values for the mask
df


# In[11]:


mask = df['geo']=='EL'
df.loc[mask, 'geo'] = "GR"
df


# In[12]:


import pycountry


# In[13]:


list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]  # create a list of country codes
print("Country codes", list_alpha_2)

def country_flag(df):
    '''
    df: Series
    return: Full name of counry or "Invalide code"
    '''
    if (df['geo'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['geo']).name
    else:
        print(df['geo'])
        return 'Invalid Code'

df['country_name']=df.apply(country_flag, axis = 1)
df


# In[14]:


pes = {'F': 'Fungicides and bactericides',
      'H': 'Herbicides, haulm destructors and moss killers',
      'I': 'Insecticides and acaricides',
      'M': 'Molluscicides',
      'PGR': 'Plant growth regulators',
      'ZR': 'Other plant protection products'}


# Let's add a new column using mapping: **[pandas.Series.map()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01)**.
# 

# In[15]:


df['pesticid_name'] = df['pesticid'].map(pes)
df['pesticid_name'] = df['pesticid_name'].astype('category')
df


# As you can see, we got two new columns.
# 

# ### Grouping data
# 

# In[16]:


df['pesticid_name'].value_counts().to_frame()


# In[17]:


pd.options.display.float_format = '{:,.0f}'.format
df.groupby('pesticid_name')['OBS_VALUE'].sum().sort_values(ascending=False).to_frame()


# As you can see, most of saled pesticides belong to the category "Fungicides and bactericides".
# 

# ### DataSet transformation
# 

# In[18]:


df


# In[19]:


df = df.dropna()
df


# In[31]:


df.info()


# In[30]:


p_df = df.pivot_table(values='OBS_VALUE', index= [ 'country_name', 'TIME_PERIOD'], columns=['pesticid_name'], aggfunc=np.sum, margins=False, dropna=False, fill_value=0)
p_df


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(8, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[24]:


p_df.plot()
plt.xticks(rotation=45)
plt.show()


# In[25]:


p_df = df.pivot_table(values='OBS_VALUE', index= [ 'country_name','TIME_PERIOD'], columns=['pesticid_name'], aggfunc=np.sum, margins=False, dropna=False, fill_value=0)
p_df


# ## Forecasting
# 

# ### Hypothesis creation
# 

# Before making a forecast, you should first determine the target (output) field for which the forecast will be built. The next step is to create a hypothesis that involves determining the input fields which our target depends on. Let's try to make a prediction about Fungicides and bactericides sales. We can propose the following hypothesis: sales depend on the sales of other pesticides.
# 
# To check this hypothesis, we should make a correlation analysis using **[pandas.DataFrame.corr()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01&highlight=corr#pandas-dataframe-corr)**.
# 

# In[26]:


pd.options.display.float_format = '{:,.2f}'.format
p_df.corr()


# Each cell contains the correlation coefficients between two columns. Therefore, diagonal elements are equal to one. As can be seen from the Fungicides and bactericides column (or row), all the correlation coefficients are between 0.43 and 0.81. It means that there are nonlinear dependencies. To begin with, we will test linear models, which accuracy will be compared with the accuracy of nonlinear models.
# 

# ### Splitting the DataSet into training and test sets
# 

# For the model fitting and testing, it is necessary to divide the DataSet into a training and a test set. You can implement this with the classic Python tools such as slices or using a special function with many flexible settings (**[sklearn.model_selection.train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01)**).
# We will take 30% of our DataSet for a test set.
# 

# In[49]:



y=p_df[["Fungicides and bactericides"]]


# In[50]:


x=p_df.drop(["Fungicides and bactericides"],axis=1)


# In[51]:


# sklearn function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)


# As a result, we got training and test DataSets.
# 

# In[52]:


print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# ### Creating a linear model using sklearn
# 

# In[53]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)


# ### Calculation of basic statistical indicators
# 

# In[55]:


from sklearn import metrics
print("Correlation train", regressor.score(X_train, y_train))
print("Correlation test", regressor.score(X_test, y_test))
print("Coefficients:", regressor.coef_)
# pair the feature names with the coefficients
print('Pair the feature names with the coefficients:')
for s in zip(col[1:], regressor.coef_):
    print(s[0], ":", s[1])
print("Intercept", regressor.intercept_)
print('Mean Absolute Error (train):', metrics.mean_absolute_error(y_train, y_pred_train))
print('Mean Absolute Error (test):', metrics.mean_absolute_error(y_test, y_pred_test))
print('Mean Squared Error (train):', metrics.mean_squared_error(y_train, y_pred_train))
print('Mean Squared Error (test):', metrics.mean_squared_error(y_test, y_pred_test))
print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# print('Mean Absolute Error (train):', metrics.mean_absolute_error(y_train, y_pred_train))
# print('Mean Absolute Error (test):', metrics.mean_absolute_error(y_test, y_pred_test))
# print('Mean Squared Error (train):', metrics.mean_squared_error(y_train, y_pred_train))
# print('Mean Squared Error (test):', metrics.mean_squared_error(y_test, y_pred_test))
# print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
# print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
# </code></details>
# 

# ### Creating models using statsmodels
# 

# As you can see, there is a big difference in accuracy between the training and test results. It means that this linear model is not correct.
# Besides, this framework cannot generate a summary report.
# To do this, we can use the **[statsmodels.api](https://www.statsmodels.org/stable/index.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01)** framework.
# 

# In[85]:


import statsmodels.api as sm
model = sm.OLS(y_train, X_train)
results = model.fit()
y_pred_test_OLS = results.predict(X_test)
y_pred_train_OLS = results.predict(X_train)
print(results.summary())


# As you can see, this framework uses the same principles for creating and fitting models. It allows us to build a summary report, also you can get all the other stats coefficients in the same way:

# In[86]:


print('coefficient of determination:', results.rsquared)
print('adjusted coefficient of determination:', results.rsquared_adj)
print('regression coefficients:', results.params, sep = '\n')


# We should join the results to compare these two framework models using **[pandas.DataFrame.join()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01)**:
# 

# In[118]:


dff=pd.DataFrame(y_pred_test,index=y_test.index)
dff.rename(columns={0:'Predicted_test'},inplace=True)
dff


# In[124]:


df_test = pd.DataFrame({'Actual_test': y_test['Fungicides and bactericides'] ,'Predicted_test_OLS': y_pred_test_OLS},index=y_test.index)
df_test=pd.merge(df_test, dff, left_index=True, right_index=True)
df_test


# As you can see, pandas joins and orders data correctly according to the index field automatically. Therefore, it is very important to check the index field datatype, especialy when we deal with datatime.
# 

# Let's visualize the data.
# 

# In[126]:


df_test[['Actual_test',  'Predicted_test_OLS','Predicted_test']].plot()
plt.xticks(rotation=45)
plt.show()


# You can see that the results of these two models are the same. Also you can see that the forecast on the test data is not perfect. To see the difference between our forecast and the real data, we can use **[seaborn.pairplot()](https://seaborn.pydata.org/generated/seaborn.pairplot.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01)**.
# 

# In[127]:


import seaborn as sns
sns.pairplot(df_test, x_vars=['Actual_test'], y_vars='Predicted_test',  kind='reg', height = 8)
plt.show()


# The real data values are plotted on the horizontal axis and the predicted ones are plotted on the vertical axis. The closer the result points are to the diagonal, the better the model forecast is. This plot proves our conclusion about the bad forecast quality under this hypothesis. Moreover, in order to make a forecast for the future, you have to know future data for the sales of other pesticides.

# ## Artificial Neural Network
# 

# ### Creating a linear model using sklearn
# 

# Let's create multilayer percepton that consist of 100 neurons in hidden layer, fit it and test.
# 

# In[56]:


from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(random_state=1, max_iter=500)
regressor.fit(X_train, y_train)
y_pred_test_MLP = regressor.predict(X_test)
y_pred_train_MLP = regressor.predict(X_train)


# In[46]:


y_pred_train_MLP.shape


# Let's calculate the same statistics as in Linear Regression.
# 

# In[57]:


print("Correlation train", regressor.score(X_train, y_train))
print("Correlation test", regressor.score(X_test, y_test))

print('Mean Absolute Error (train):', metrics.mean_absolute_error(y_train, y_pred_train_MLP))
print('Mean Absolute Error (test):', metrics.mean_absolute_error(y_test, y_pred_test_MLP))
print('Mean Squared Error (train):', metrics.mean_squared_error(y_train, y_pred_train_MLP))
print('Mean Squared Error (test):', metrics.mean_squared_error(y_test, y_pred_test_MLP))
print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train_MLP)))
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test_MLP)))


# As you can see, there are much less errors for test than for Linear regression.
# 

# ### Creating a linear model using keras
# 

# Let's try to use a more powerful framework **keras**.
# 

# In[58]:


from sklearn.preprocessing import MinMaxScaler
scaler_x_train = MinMaxScaler(feature_range=(0, 1))
scaler_y_train = MinMaxScaler(feature_range=(0, 1))
scaler_x_test = MinMaxScaler(feature_range=(0, 1))
scaler_y_test = MinMaxScaler(feature_range=(0, 1))

# Normilized data
scaled_x_train = scaler_x_train.fit_transform(X_train.astype('float64')) 
scaled_y_train = scaler_y_train.fit_transform(y_train.astype('float64').values.reshape(-1, 1))
scaled_x_test = scaler_x_test.fit_transform(X_test.astype('float64'))
scaled_y_test = scaler_y_test.fit_transform(y_test.astype('float64').values.reshape(-1, 1))


# In[59]:


def BP_model(X):
    # create model
    model = Sequential()
    #model.add(BatchNormalization(input_shape=tuple([X.shape[1]])))
    model.add(Dense(100, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[60]:


from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
epochs = 10

batch_size=int(y_train.shape[0]*.1)

estimator = KerasRegressor(build_fn=BP_model, X=scaled_x_train, epochs=epochs, batch_size=batch_size, verbose=0)


# Now, let’s train our model for **10** epochs.
# It should be noted, that the fitting process is very slow. Therefore we saved our fitted model to a file.
# To save time, we will upload the fitted model.
# If you like, you can leave the parameter **fitting on True** to refit your model.
# If you like, you can leave the parameter **fitting_save on True** to resave your model.
# 

# In[61]:


fitting = True
fitting_save = True


import pickle

if fitting:
    history=estimator.fit(scaled_x_train, scaled_y_train, validation_data=(scaled_x_test, scaled_y_test))
    if fitting_save:
        estimator.model.save('BP_saved_model.h5')
        print("Saved model to disk")
        with open('history.pickle', 'wb') as f:
            pickle.dump(history.history, f)
# load model 
from keras.models import load_model

# Instantiate the model as you please (we are not going to use this)
estimator = KerasRegressor(build_fn=BP_model, X=scaled_x_train, epochs=epochs, batch_size=batch_size, verbose=0)
# This is where you load the actual saved model into a new variable.
estimator.model = load_model('BP_saved_model.h5')    
with open('history.pickle', 'rb') as f:
    history = pickle.load(f)
print("Loaded model from disk")


# Let's show [**loss and validation loss dynamics**](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagricultureprognosticationusingbyneuralnetwork28167412-2022-01-01).
# 

# In[62]:


plt.figure()
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


# As you can see, the Neural Network is fitted well and no overfitting is observed.
# Let's calculate the prediction of the training (**res_train_ANN**) and test (**res_test_ANN**) sets.
# 

# Let's make a prediction and inverse normalize data.
# 

# In[63]:


res_tr=estimator.predict(scaled_x_train)
res_ts=estimator.predict(scaled_x_test)
res_train_ANN=scaler_y_train.inverse_transform(res_tr.reshape(-1, 1)).flatten()
res_test_ANN=scaler_y_test.inverse_transform(res_ts.reshape(-1, 1)).flatten()


# In[64]:


res_test_ANN.shape


# Let's calculate the same statistics as in Linear Regression.
# 

# In[65]:


print("Correlation train", regressor.score( X_train, res_train_ANN))
print("Correlation test", regressor.score( X_test, res_test_ANN))

print('Mean Absolute Error (train):', metrics.mean_absolute_error( y_train, res_train_ANN))
print('Mean Absolute Error (test):', metrics.mean_absolute_error( y_test, res_test_ANN))
print('Mean Squared Error (train):', metrics.mean_squared_error( y_train, res_train_ANN))
print('Mean Squared Error (test):', metrics.mean_squared_error( y_test, res_test_ANN))
print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error( y_train, res_train_ANN)))
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, res_test_ANN)))


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# print("Correlation train", regressor.score(X_train, res_train_ANN))
# print("Correlation test", regressor.score(X_test, res_test_ANN))
# 
# print('Mean Absolute Error (train):', metrics.mean_absolute_error(y_train, res_train_ANN))
# print('Mean Absolute Error (test):', metrics.mean_absolute_error(y_test, res_test_ANN))
# print('Mean Squared Error (train):', metrics.mean_squared_error(y_train, res_train_ANN))
# print('Mean Squared Error (test):', metrics.mean_squared_error(y_test, res_test_ANN))
# print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, res_train_ANN)))
# print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, res_test_ANN))) </code></details>
# 

# As you can see, the correlation is better than for the previous NN but the error is worse. Let's visualize the data for comparison.
# 

# In[130]:


df_test = pd.DataFrame({'Actual_test': y_test['Fungicides and bactericides'],  'Predicted_test_MLP': y_pred_test_MLP, 'Predicted_test_ANN': res_test_ANN},index=y_test.index)
df_test=pd.merge(df_test, dff, left_index=True, right_index=True)
df_test


# In[131]:


df_test[['Actual_test', 'Predicted_test', 'Predicted_test_MLP', 'Predicted_test_ANN', ]].plot()
plt.xticks(rotation=45)
plt.show()


# In[132]:


import seaborn as sns
sns.pairplot(df_test, x_vars=['Actual_test'], y_vars='Predicted_test_ANN',  kind='reg', height = 8)
plt.show()


# As you can see, an ANN shows better results.
# 

# ## Conclusions
# 

# In this lab work, we have learned how to build hypotheses for forecasting models. We have transformed DataSets for input-output models. We have learned how to divide DataSets into training and test sets and normaliza data. The Linear regression and Neuro Network models have been created, fitted and tested in this lab.
# 
