#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import copy
import numpy as np
from pulp import *


# In[2]:


df_county_adj = pd.read_csv("https://www2.census.gov/geo/docs/reference/county_adjacency.txt", sep='\t',encoding = 'latin1',
                           header = None, names = ['County', 'County GEOID', 'Neighbor', 'Neighbor GEOID'], \
                           dtype = {'County GEOID': 'category', 'Neighbor GEOID': 'category'})

#Filling values down as they appear in an aggregate view
df_county_adj.ffill(inplace = True) 

#Filtering to just TN & removing counties neighboring themselves
df_copy = copy.deepcopy(df_county_adj[df_county_adj['County'].str.contains("TN")])
df_TN_adj = copy.deepcopy(df_copy[df_copy['Neighbor'].str.contains("TN")])
df_TN_adj = copy.deepcopy(df_TN_adj[~(df_TN_adj['County']==df_TN_adj['Neighbor'])])

#Removing extra text from county names
df_TN_adj['County'] = df_TN_adj['County'].str.replace(" County, TN", "")
df_TN_adj['County'] = df_TN_adj['County'].str.replace(" ", "")
df_TN_adj['Neighbor'] = df_TN_adj['Neighbor'].str.replace(" County, TN", "")
df_TN_adj['Neighbor'] = df_TN_adj['Neighbor'].str.replace(" ", "")

#Creating an adjacency matrix
df_TN_adj['Values'] = 1
df_TN_adj = df_TN_adj.pivot(index = 'County', columns = 'Neighbor', values = 'Values')
df_TN_adj.fillna(0, inplace = True)

df_TN_adj.sort_index(axis=1, inplace=True)
df_TN_adj.sort_index(axis=0, inplace=True)
df_TN_adj


# https://data.census.gov/cedsci/table?q=race&g=0400000US47%240500000

# In[3]:


df_race = pd.read_csv("DECENNIALPL2020.P2-2022-05-04T001208.csv")

#Removing white spaces from labels column & set as index
df_race['Label (Grouping)'] = df_race['Label (Grouping)'].str.strip()
df_race.set_index('Label (Grouping)', inplace = True)

#Removing all other races and leaving total, white alone, and other
df_race = df_race.filter(items = ['Total:', 'White alone'], axis = 0)
df_race.loc['Other Races'] = df_race.loc['Total:'] - df_race.loc['White alone']

#Removing extra text from column names
race_cols = df_race.columns.values
new_race_cols = []
for i in range(0,len(race_cols)):
    new_race_cols.append((race_cols[i].replace(" County, Tennessee", "")).replace(" ", ""))
df_race.columns = new_race_cols

df_race.sort_index(axis=1, inplace=True)
df_race


# In[4]:


total_pop = df_race.loc['Total:'].sum()
avg_pop = df_race.loc['Total:'].sum()/9
white_pop =  df_race.loc['White alone'].sum()
white_pop_prop = white_pop/total_pop

print("Total Population: {}".format(total_pop))
print("Avg Population per District: {}".format(round(avg_pop)))
print("{}% of Population is White".format(round(white_pop_prop*100)))


# In[5]:


df_race.loc[:,df_race.loc['Total:']>700000]


# In[6]:


print("{}% of Davidson is white".format(round(df_race.loc['White alone','Davidson']/df_race.loc['Total:', 'Davidson']*100)))
print("{}% of Shelby is white".format(round(df_race.loc['White alone','Shelby']/df_race.loc['Total:', 'Shelby']*100)))


# Since both of these counties can be their own districts based off of population itself, I am going to use them to create a double rep district that has 2 reps with double the population of other districts. This will bring our 9 districts down to 7. 

# In[7]:


n_counties = len(df_TN_adj)
n_districts = 7

var_combs = []
for i in range(1,n_counties+1):
    for j in range (1, n_districts+1):
        var_combs.append(str(i)+"_"+str(j))


# In[8]:


model = LpProblem("model", LpMinimize)

#Binary variable for per county-district combination
DV_y = LpVariable.matrix("Y", var_combs, cat="Binary")
assignment = np.array(DV_y).reshape(n_counties,n_districts)


# In[9]:


#Adjacency constraint

n_adj = np.sum(df_TN_adj,axis = 0)
for j in range(n_districts):
    for i in range(n_counties):
            model += assignment[i][j] <= lpSum(df_TN_adj.iloc[i][k]*assignment[k][j] for k in range(n_counties))


# In[10]:


#Predetermining the districts for Davidson & Shelby respectively

model += assignment[np.where(df_TN_adj.columns.values=='Davidson')][0][0] == 1 #Putting Davidson in District 1
model += assignment[np.where(df_TN_adj.columns.values=='Shelby')][0][1] == 1 #Putting Shelby in District 2

#Setting population ranges, normal districts can be + / - 20% from avg 
for j in range(n_districts):
        if j in [0,1]:
            model += lpSum(df_race.iloc[0,i]*assignment[i][j] for i in range(n_counties)) <= avg_pop*2*1.2 
            model += lpSum(df_race.iloc[0,i]*assignment[i][j] for i in range(n_counties)) >= avg_pop*2*0.8
        else:
            model += lpSum(df_race.iloc[0,i]*assignment[i][j] for i in range(n_counties)) <= avg_pop*1.2
            model += lpSum(df_race.iloc[0,i]*assignment[i][j] for i in range(n_counties)) >= avg_pop*0.8


# In[11]:


#Setting 1 district per county
for i in range(n_counties):
    model += lpSum(assignment[i][j] for j in range(n_districts)) == 1


# ### Objective Function: 
# 
# Minimize
# ### $\sum_{i=1}^{7} |\frac{white\_pop_i}{total\_pop_i} - white\_pop\_prop|$
# 
# Since PuLp does not intake division, the objective function is rewritten to:
# 
# ### $\sum_{i=1}^{7} |{white\_pop_i} - {total\_pop_i} \times white\_pop\_prop|$

# Using absolute value principle:
# 
# $y \leq |x|$
# 
# $-y \leq x \leq y$

# In[12]:


#Create absolute values so that we get total difference regardless of lower or higher

abs_pop_diff = LpVariable.dicts("abs_pop_diff", range(n_districts))

for i in range(n_districts):
    model += abs_pop_diff[i] >= sum(df_race.iloc[1,j]*assignment[j][i] for j in range(n_counties))         - sum(df_race.iloc[0,j]*assignment[j][i] for j in range(n_counties)) * white_pop_prop
    model += abs_pop_diff[i] >= -1 * (sum(df_race.iloc[1,j]*assignment[j][i] for j in range(n_counties))         - sum(df_race.iloc[0,j]*assignment[j][i] for j in range(n_counties)) * white_pop_prop)

#Objective Function: Minimize % difference between each district's white population and state white population
model += lpSum(abs_pop_diff)


# In[13]:


model.solve(GLPK_CMD())
print("Status:", LpStatus[model.status])


# In[14]:


for v in model.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)


# In[15]:


np.where(df_TN_adj.columns.values == 'Van Buren')


# In[16]:


print(model)


# In[17]:


assigned_districts = pd.DataFrame()
for v in model.variables():
    if v.varValue == 1:
        df = pd.DataFrame({
            'var':[v.name],
            'county':[v.name.split('_')[1]],
            'district':[v.name.split('_')[2]],
        })

        assigned_districts = assigned_districts.append(df)
assigned_districts


# In[18]:


assigned_districts = pd.DataFrame()
for v in model.variables():
    if v.varValue == 1:
        df = pd.DataFrame({
            'var':[v.name],
            'county':[df_TN_adj.columns.values[int(v.name.split('_')[1])-1]],
            'district':[v.name.split('_')[2]],
        })

        assigned_districts = assigned_districts.append(df)
assigned_districts


# In[19]:


df_fips = pd.read_csv("tn_fips.csv")


# In[20]:


from urllib.request import urlopen
import json
import random
import plotly.express as px

with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as response:
 counties = json.load(response)

target_states = ['47']
counties['features'] = [f for f in counties['features'] if f['properties']['STATE'] in target_states]


# In[21]:


df_fips['County'] = df_fips['County'].str.replace(" ","")

df_fips = df_fips.merge(
    assigned_districts,
    left_on = 'County',
    right_on = 'county',
    how = 'left'
)


# In[22]:


# df_fips['district'] = df_fips.apply(lambda x: random.randint(1,9),axis = 1)
# df_fips['district'] = df_fips['district'].astype(str)

rank_fig = px.choropleth(df_fips, geojson=counties, locations="code", color="district",
#  color_discrete_map = ['1','2','3','4','5','6','7','8','9'],
#  range_color=(1, 133),
 scope="usa",
#  labels={‘Jul 2019 Rank’:’Rank: 2019 Population’}
 )
rank_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
rank_fig.show()


# In[23]:


#Demographics of our districts

df_race_t = df_race.transpose()

district_pop = assigned_districts.join(df_race_t, lsuffix='1', rsuffix='2', on = 'county')
district_pop = district_pop.groupby('district').agg({"Total:" : "sum", "White alone" : "sum"})
district_pop['White %'] = district_pop['White alone']/district_pop['Total:']
district_pop


# ### References:
# 
# https://towardsdatascience.com/how-to-draw-congressional-districts-in-python-with-linear-programming-b1e33c80bc52

# In[24]:


df_race.sum(axis=1)


# In[25]:


942053/2010594


# In[ ]:




