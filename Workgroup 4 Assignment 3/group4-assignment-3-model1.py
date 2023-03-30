#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import copy
import numpy as np
from pulp import *
import math
# import io


# Counties

# In[2]:


df_counties = pd.read_csv("https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO47.txt")
df_counties.head()


# Distance Matrix
# 
# https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/
# 

# In[3]:


from sklearn.neighbors import DistanceMetric
from math import radians

df_counties['lat'] = np.radians(df_counties['LATITUDE'])
df_counties['lon'] = np.radians(df_counties['LONGITUDE'])

dist = DistanceMetric.get_metric('haversine')

# df_counties[['lat','lon']].to_numpy()
# dist.pairwise(df_counties[['lat','lon']].to_numpy())*6373

df_dist = pd.DataFrame(dist.pairwise(df_counties[['lat','lon']].to_numpy())*6373,  columns=df_counties.COUNAME.unique(), index=df_counties.COUNAME.unique())

df_dist.head()


# Adjacency Data

# In[4]:


df_county_adj = pd.read_csv("https://www2.census.gov/geo/docs/reference/county_adjacency.txt", sep='\t',encoding = 'latin1',
                           header = None, names = ['County', 'County GEOID', 'Neighbor', 'Neighbor GEOID'], \
                           dtype = {'County GEOID': 'category', 'Neighbor GEOID': 'category'})

#Filling values down as they appear in an aggregate view
df_county_adj.ffill(inplace = True) 

#Filtering to just TN
df_copy = copy.deepcopy(df_county_adj[df_county_adj['County'].str.contains("TN")])
df_TN_adj = copy.deepcopy(df_copy[df_copy['Neighbor'].str.contains("TN")])

#Removing extra text from county names
df_TN_adj['County'] = df_TN_adj['County'].str.replace(" County, TN", "")
df_TN_adj['Neighbor'] = df_TN_adj['Neighbor'].str.replace(" County, TN", "")

#Creating an adjacency matrix
df_TN_adj['Values'] = 1
df_TN_adj = df_TN_adj.pivot(index = 'County', columns = 'Neighbor', values = 'Values')
df_TN_adj.fillna(0, inplace = True)

df_TN_adj


# drop counties that will be auto assigned

# In[5]:


df_TN_adj = df_TN_adj.drop(labels=['Shelby','Davidson'])
df_TN_adj = df_TN_adj.drop(labels=['Shelby','Davidson'],axis = 1)
df_dist = df_dist.drop(labels=['Shelby','Davidson'])
df_dist = df_dist.drop(labels=['Shelby','Davidson'],axis = 1)

df_TN_adj


# Demographics

# In[6]:


df_race = pd.read_csv("DECENNIALPL2020.P2-2022-05-04T001048.csv")

#Removing white spaces from labels column & set as index
df_race['Label (Grouping)'] = df_race['Label (Grouping)'].str.strip()
df_race.set_index('Label (Grouping)', inplace = True)

#Removing all other races and leaving total, white alone, and other
df_race = df_race.filter(items = ['Total:', 'White alone'], axis = 0)
df_race.loc['Other Races'] = df_race.loc['Total:'].str.replace(",","").astype(int) - df_race.loc['White alone'].str.replace(",","").astype(int)
#Removing extra text from column names
race_cols = df_race.columns.values
new_race_cols = []
for i in range(0,len(race_cols)):
    new_race_cols.append(race_cols[i].replace(" County, Tennessee", ""))
df_race.columns = new_race_cols

# df_race
df_race


# ### Data Prep

# i. Find the total population and "ideal" characteristics of a district

# In[7]:


total_pop = df_counties.sum()['POPULATION']
df_counties['Prop'] = df_counties['POPULATION']/total_pop
ideal_pop = total_pop/9
ideal_prop = (total_pop/9)/total_pop

print(f'ideal population = {ideal_pop} ({round(ideal_prop,3)*100}%) ' )
# df_counties.sort_values('Prop',ascending=False)

# give one rep to Shelby and Davidson Counties
# filter our large counties
df_counties = df_counties[df_counties['Prop'] < .10]
df_counties.sort_values('Prop',ascending=False)


# In[8]:


df_counties = df_counties.sort_values('COUNAME').reset_index().drop(columns=['index'])


# In[9]:


# find upper and lower bounds
lower_lim = .9 * ideal_pop - (ideal_pop*.9 % 1000)
upper_lim = 1.1 * ideal_pop - (ideal_pop*1.1 % 1000)
ideal = 1.0 * ideal_pop - (ideal_pop*1.0 % 1000)

print(lower_lim)
print(upper_lim)


# In[10]:


n_counties = len(df_counties)
n_districts = 7

var_combs = []
for i in df_counties['COUNAME']:
    for j in range (1, n_districts+1):
        var_combs.append(str(i)+"_"+str(j))

# county_pop = makeDict([df_counties['COUNAME']],df_counties['POPULATION'] )
# population array
county_pop = np.array(df_counties['POPULATION'])
c_range = range(0,n_counties)
d_range = range(0,n_districts)


# In[11]:


# adjacency
adj_mat = np.array(df_TN_adj)
# distanct
dist_mat = np.array(df_dist)

adjacencies = np.sum(adj_mat,axis = 1)

print(max(adjacencies))
print(min(adjacencies))


# In[12]:


model = LpProblem("TN",LpMinimize)

var_combs.sort()
DV_y = LpVariable.matrix("Y", var_combs, cat="Binary")
assignment = np.array(DV_y).reshape(n_counties,n_districts)


# ### Objective Function:
# 
# **Minimize the difference between the average district population and the lower bound of the ideal population. (Hopefully by constraining districts to be >= the lower bound this will work)**
# 
# 
# 

# In[13]:


# model += lpSum(assignment[i][0]*county_pop[i] for i in c_range)*(1/7) + \
#     lpSum(assignment[i][1]*county_pop[i] for i in c_range)*(1/7) + \
#         lpSum(assignment[i][2]*county_pop[i] for i in c_range)*(1/7) + \
#             lpSum(assignment[i][3]*county_pop[i] for i in c_range)*(1/7) + \
#                 lpSum(assignment[i][4]*county_pop[i] for i in c_range)*(1/7) + \
#                     lpSum(assignment[i][5]*county_pop[i] for i in c_range)*(1/7) + \
#                         lpSum(assignment[i][6]*county_pop[i] for i in c_range)*(1/7) \
#                             - lower_lim


# In[14]:


model += (lpSum(assignment[i][0]*county_pop[i] for i in c_range) - lower_lim)*(1/7) +     (lpSum(assignment[i][1]*county_pop[i] for i in c_range)- lower_lim)*(1/7) +         (lpSum(assignment[i][2]*county_pop[i] for i in c_range)- lower_lim)*(1/7) +             (lpSum(assignment[i][3]*county_pop[i] for i in c_range)- lower_lim)*(1/7) +                 (lpSum(assignment[i][4]*county_pop[i] for i in c_range)- lower_lim)*(1/7) +                     (lpSum(assignment[i][5]*county_pop[i] for i in c_range)- lower_lim)*(1/7) +                         (lpSum(assignment[i][6]*county_pop[i] for i in c_range)- lower_lim)*(1/7) 


# Minimize
# 
# 
# $$\sum^7_{j=1}\sum^{95}_{i=1}\,county\_pop_{ij} - lower\_lim$$

# ### Constraints
# 
# **1. One district per county**

# In[15]:


for i in c_range:
    model += lpSum(assignment[i][j] for j in d_range) == 1


# **2. Adjacency (starting with must have at least 1 adjacency)**

# In[16]:


for i in range(3,10):
    print(f'{i}:{np.count_nonzero(adjacencies==i)}')


# In[17]:


for j in d_range:
    for i in c_range:
        if adjacencies[i] == 3:
            model += 3*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)
        elif adjacencies[i] ==4:
            model += 3*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)
        elif adjacencies[i] ==5:
            model += 2*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)
        elif adjacencies[i] ==6:
            model += 2*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)
        elif adjacencies[i] ==7:
            model += 2*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)
        else:
            model += 3*assignment[i][j] <= lpSum(adj_mat[i][k]*assignment[k][j] for k in c_range)


# **3. Population**

# In[18]:


for j in d_range:
    model += lpSum(county_pop[i]*assignment[i][j] for i in c_range) >= lower_lim

for j in d_range:
    model += lpSum(county_pop[i]*assignment[i][j] for i in c_range) <= upper_lim


# **4. Distance**

# In[19]:


max_dist = math.ceil(np.max(dist_mat)*.75)
print(f"maximum allowable distance between counties = {max_dist}km")
for j in d_range:
    for i in c_range:
        for k in c_range:
            model += lpSum(max_dist*assignment[k][j]) >=                 lpSum(dist_mat[i][k]*assignment[i][j] + dist_mat[i][k]*(assignment[k][j]-1))


# if assignment k_j = 1
# 
#     540 >= dist i_k * assignment i_j + dist i_k * 0
# 
#         if assignment i_j = 1
# 
#             360 >= dist_i_k*1 + 0
#             ok
# 
#         if assignment i_j = 0
#             360 >= 0 + 0
#             ok
# 
# if assignment k_j = 0
# 
#     0 >= dist i_k * assignment i_j + dist i_k * -1
# 
#         if assignment i_j = 1
# 
#             0 >= dist_i_k*1 + dist_i_k*-1
#             ok
# 
#         if assignment i_j = 0:
# 
#             0 >= 0 + -1
#             ok

# In[20]:


model.solve()
LpStatus[model.status]


# In[21]:


assigned_districts = pd.DataFrame()
for v in model.variables():
    if v.varValue == 1:
        if "Van_Buren" in v.name:
            df = pd.DataFrame({
                'var':[v.name],
                'county':["VanBuren"],
                'district':[v.name.split('_')[3]],
            })
        else:
            df = pd.DataFrame({
                'var':[v.name],
                'county':[v.name.split('_')[1]],
                'district':[v.name.split('_')[2]],
            })

        assigned_districts = assigned_districts.append(df)


# ### Map

# In[22]:


df_fips = pd.read_csv("tn_fips.csv")


# In[23]:


from urllib.request import urlopen
import json
import random
import plotly.express as px

with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as response:
 counties = json.load(response)

target_states = ['47']
counties['features'] = [f for f in counties['features'] if f['properties']['STATE'] in target_states]


# In[24]:


df_fips['County'] = df_fips['County'].str.replace(" ","")

df_fips = df_fips.merge(
    assigned_districts,
    left_on = 'County',
    right_on = 'county',
    how = 'left'
)

print(df_fips[df_fips['district'].isna()])


# In[25]:


# df_fips.loc[df_fips['County'].str.contains("Van"),'district'] = '2'
df_fips.loc[df_fips['County']=='Davidson','district'] = '8'
df_fips.loc[df_fips['County']=='Shelby','district'] = '9'

rank_fig = px.choropleth(df_fips, geojson=counties, locations="code", color="district",
 scope="usa",
 hover_data=["County","district"]
 )
rank_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
rank_fig.show()


# In[26]:


assignment


# In[ ]:




