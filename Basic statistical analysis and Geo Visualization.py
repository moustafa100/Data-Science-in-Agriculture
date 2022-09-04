#!/usr/bin/env python
# coding: utf-8

# ## **Data Science in Agriculture. Basic statistical analysis and Geo Visualization**
# 

# ## Abstract
# 

# This lab is dedicated to downloading, pre-preparing and making statistical analysis of Economic accounts for agriculture and to creating interactive maps showing the dynamics of the prices.
# 

# ## Introduction
# 

# The main problem to be solved in this laboratory is the download, statistical analysis and visualization of a DataSet.
# 
# The basic difficulty of statistical analysis of real data is that it is prepared or presented in a form that is not convenient for machine methods of statistical analysis. Therefore, this lab shows methods of automatic pre-preparation of real data for such cases. The next problem is the ability to competently manipulate and transform big data in order to obtain a convenient statistical report both in tabular form and in the form of graphs.
# 
# Therefore, the main goal that we are to achieve in this lab is learning how to download, pre-process and conduct a basic statistical analysis of Economic accounts for agriculture and present it on interactive maps.
# 

# ## Download data from a .csv file
# 

# Some libraries should be imported before you can begin.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('agric.csv',engine='python')


# Now let's look at our DataSet.
# 

# In[3]:


df


# ## Data preparation
# 

# In[4]:


df.columns


# Let's select only colums 6, 7 and 8 for our future analysis.
# 

# In[5]:


col = df.columns[6:-1]
col


# Remove all the other columns from the DataSet.
# 

# In[6]:


df = df[col]
df


# In[7]:


df.info()


# In[8]:


df.loc[:, 'geo'] = df['geo'].astype('category')
df.info()


# Let's get a list of countries (**[pandas.unique()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**).
# 

# In[9]:


df['geo'].unique()


# It should be noted that there are some nonsandard counry codes for the United Kingdom and Greece.
# We should change the values: UK to GB for the United Kingdom and EL to GR for Greece.
# To do this, we should add new category names using **[pandas.Series.cat.add_categories()](https://pandas.pydata.org/pandas-docs/version/1.0.5/reference/api/pandas.Series.cat.add_categories.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**.
# 

# In[10]:


df['geo'] = df['geo'].cat.add_categories(["GB", "GR"])


# Then we should change the values using a binary mask:
# 

# In[11]:


pd.options.mode.chained_assignment = None  # swich of the warnings
mask = df['geo'] == 'UK' # Binary mask
df.loc[mask, 'geo'] = "GB" # Change the values for mask
df


# Let's do the same for Greece: 'EL'->'GR'
# 

# In[12]:


mask = df['geo'] =='EL'
df.loc[mask, 'geo'] = 'GR'
df


# After that, add a new column that contains full names of countries. To do this, we can use **[pycountry](https://pypi.org/project/pycountry/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)** library.
# 

# In[13]:


import pycountry


# pycountry provides the ISO databases for different standards.
# 
# In order to add a column with full country names we need to create a function that will get a counry code and return a full name.
# Then it should be the function **[pandas.DataFrame.apply()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)** for calculating new column values.
# 

# In[14]:


list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]  # create a list of country codes
print("Country codes", list_alpha_2)

def country_flag(df):
    '''
    df: Series
    return: Full name of country or "Invalide code"
    '''
    if (df['geo'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['geo']).name
    else:
        print(df['geo'])
        return 'Invalid Code'

df['country_name']=df.apply(country_flag, axis = 1)
df.head(5)


# In[15]:


(df['country_name']=='Invalid Code').value_counts()


# As you can see, the column with full country names has been added and this DataSet contains a lot of data with an Invalide Code. Let's remove this data using a binary mask.
# 

# In[16]:


mask = df['country_name'] != 'Invalid Code'
df = df[mask]
df


# ## Statistical analysis
# 

# Let's study this DataSet.
# 

# In[17]:


df.info()


# The summary statistics can be calculated easily with the following command: **[describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01&highlight=describe#pandas.DataFrame.describe)**.
# 

# In[18]:


df.describe()


# As you can see, the result highlights basic statistical information for all the columns except the categorical and object ones.
# The information includes the total, average, standard deviation, minimum, maximum and the values of the main quarters.
# In order to display the summary information of category fields, we have to specify the data types we want to display the statistics for:
# 

# In[19]:


df.describe(include=['category'])


# As you can see, the statistical information consists of the number of unique values, the value of the most popular category and the number of its values.
# The detailed information for a specific column can be obtained as follows (**[value_counts()](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01&highlight=value_counts#pandas.Series.value_counts)**):
# 

# In[20]:


df['country_name'].value_counts()


# You can see that this information is not suitable because the data is not grouped. To get suitable statistics this DataSet should be transformed using a pivot table **[pivot_table()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01&highlight=pivot_table#pandas.DataFrame.pivot_table)**
# 

# In[21]:


pt_country = pd.pivot_table(df, values= 'obs_value', index= ['time_period'], columns=['country_name'], aggfunc='sum', margins=True)
pt_country


# After that we can calculate statistic description for each country.
# 

# In[22]:


pt_country.describe()


# Or we can get statistics for years:
# 

# In[23]:


pt = pd.pivot_table(df, values= 'obs_value', index= ['country_name'], columns=['time_period'], aggfunc='sum', margins=True)
pt


# In[24]:


pt.describe()


# ## Data visualization
# 

# Let's build a plot for the last row ('All') except the last values for column ('All'). Pandas inherits Matplotlib function for plotting.
# 

# In[25]:


pt.iloc[-1][:-1].plot()


# Let's build a bar plot for summary values for each country (the last column 'All' except the last row).
# 

# In[26]:


pt['All'][:-1].plot.bar(x='country_name', y='val', rot=90)


# Let's build a plot for economic accounts dynamics for Sweden.
# 

# In[27]:


pt.loc['Sweden'][:-1].plot()


# Let's compare economic accounts for Germany and France on a bar plot. To do this we should make a lot of preparation:
# 

# In[28]:


import numpy as np

import matplotlib.pyplot as plt

x = np.arange(len(pt.columns)-1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots() # Create subplots
rects1 = ax.bar(x - width/2, pt.loc['Germany'][:-1], width, label='Germany') # parameters of bars
rects2 = ax.bar(x + width/2, pt.loc['France'][:-1], width, label='France')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('OBS_VALUE')
ax.set_xlabel('Years')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(pt.columns[:-1])
ax.legend()

fig.tight_layout()

plt.show()


# Also we can build some specific plots using SeaBorn library.
# 

# In[29]:


import seaborn as sns
sns.set_theme(color_codes=True)
d = pd.DataFrame(pt.loc['Sweden'][:-1])
sns.regplot(x=d.index, y="Sweden", data=d,)


# ## Build a trend line
# 

# Let's make a forecast of dynamics using a linear trend line for Sweden.
# To build a linear model, it is necessary to create the linear model itself, fit it, test it, and make a prediction.
# To do this, use **[sklearn.linear_model.LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**.
# 

# In[30]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.reshape(d.index, (-1, 1)) # transform X values
y = np.reshape(d.values, (-1, 1)) # transform Y values
model.fit(X, y)


# In[31]:


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


# When the model is fitted, we can build our forecast. We should add new values for X and calculate Y.
# 

# In[32]:


X_pred= np.append(X, [2021, 2022, 2023])
X_pred = np.reshape(X_pred, (-1, 1))
# calculate trend
trend = model.predict(X_pred)

plt.plot(X_pred, trend, "-", X, y, ".")


# ## Interactive maps
# 

# ### Data transformation for mapping
# 

# It is convenient to display the changes of economic accounting on a map to visualize it. There are several libraries for this. It is convenient to use the library **[plotly.express](https://plotly.com/python/plotly-express/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**.
# 

# In[33]:


import plotly.express as px


# Let's display our DataSet.
# 

# In[34]:


df


# ### Download polygons of maps
# 

# In[35]:


import json


# In[36]:


get_ipython().system('wget european-union-countries.geojson "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-agriculture-basic-statistical-analysis-and-geo-visualisation/european-union-countries.geojson"')


# In[37]:


with open("european-union-countries.geojson", encoding="utf8") as json_file:
    EU_map = json.load(json_file)


# In[ ]:


fig = px.choropleth(
    df,
    geojson=EU_map,
    locations='country_name',
    featureidkey='properties.name',    
    color= 'obs_value', 
    scope='europe',
    hover_name= 'country_name',
    hover_data= ['country_name', 'obs_value'],
    animation_frame= 'time_period', 
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
)


# Than we should change some map features. For example: showcountries, showcoastline, showland and fitbouns in function: **[plotly.express.update_geos()](https://plotly.com/python/map-configuration/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**.
# Also we can modify the map layout: **[plotly.express.update_layout](https://plotly.com/python/creating-and-updating-figures/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation28110426-2022-01-01)**.
# 

# In[ ]:


fig.update_geos(showcountries=False, showcoastlines=False, showland=True, fitbounds=False)

fig.update_layout(
    title_text ="Agriculture Economic accounts",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)


# In[ ]:


from IPython.display import HTML
HTML(fig.to_html())


# ## Conclusions
# 

# As evidenced in practice, the data obtained in real field experiments is not suitable for direct statistical processing. Therefore, in this lab we learned the basic methods of downloading and preliminary data preparation.
# Unlike the well known classical approaches to statistical data analysis, Python contains many powerful libraries that allow you to manipulate data easily and quickly. Therefore, we have learned the basic methods of automating a library such as Pandas for statistical data analysis. We also learned the basic methods of visualizing the obtained data with the SeaBorn library which also contains effective means of visual data analysis. At the end of the laboratory work, we displayed the DataSet on a dynamic interactive map in \* .html format.
# 
