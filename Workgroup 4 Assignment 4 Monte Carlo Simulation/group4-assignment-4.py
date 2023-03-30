#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Suppose you are working with a rancher in Central Texas. The rancher is building a rainwater harvesting/recapture system, and he/she wants to ensure that the system will not run out of water. 
# 
# The rancher thinks that a 25-thousand-gallon tank and a 3-thousand-square-foot roof capture area will be sufficient. The rancher asks you to check. The rancher has questions, "What are the chances that the volume in the storage tank will reach zero? Will a larger tank or roof be needed?"

# In[2]:


rain = pd.read_csv("monthly-rainfall-data.csv")
rain


# In[3]:


rain = pd.melt(rain,id_vars='Year',value_vars=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],value_name='Rainfall').rename(columns={'variable':'Month'})
rain


# ### Monthly Rainfall (All Months)

# In[4]:


plt.hist(rain['Rainfall'])
plt.xlabel('Rainfall (in.)')
plt.ylabel('Count')
plt.title('Monthly Rainfall 2000-2011')
plt.show()


# ### Monthly Rainfall (By Month)

# In[5]:


p = sns.FacetGrid(rain,col = 'Month',col_wrap=4)
p.map_dataframe(sns.histplot,x='Rainfall')


# ### Descriptive Stats

# In[6]:


def q75(x):
    return x.quantile(0.75)
def q90(x):
    return x.quantile(0.90)

rain.groupby('Month')['Rainfall'].agg(['mean','median','std',q75,q90,'max']).sort_values('median')


# ### Bootstrapping

# In[7]:


# 1000 bootstrap resamples of the precipitation mean from 2000-2011 for each month
# using the mean and stddev from the bootstrap as the distributions for the simulation

# using the surrounding months to increase the sample size for each target month
# bootstrapping using sample weights = 1 if year = target, .5 otherwise
# so the precipitation values are 2X as likely to get drawn if they are in the target month
def wt(x,month):
    if x['Month'] == month:
        y = 1
    else:
        y = .5

    return(y)

n = 1000
# weight vector
weights = rain[rain['Month'].isin(['Dec','Jan','Feb'])].apply(lambda x: wt(x,'Jan'),axis = 1)
# new monthly samples:
jan = pd.Series([
    rain[rain['Month'].isin(['Dec','Jan','Feb'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Jan','Feb','Mar'])].apply(lambda x: wt(x,'Feb'),axis = 1)
feb = pd.Series([
    rain[rain['Month'].isin(['Jan','Feb','Mar'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Feb','Mar','Apr'])].apply(lambda x: wt(x,'Mar'),axis = 1)
mar = pd.Series([
    rain[rain['Month'].isin(['Feb','Mar','Apr'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Mar','Apr','May'])].apply(lambda x: wt(x,'Apr'),axis = 1)
apr = pd.Series([
    rain[rain['Month'].isin(['Mar','Apr','May'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Apr','May','Jun'])].apply(lambda x: wt(x,'May'),axis = 1)
may = pd.Series([
    rain[rain['Month'].isin(['Apr','May','Jun'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['May','Jun','Jul'])].apply(lambda x: wt(x,'Jun'),axis = 1)
jun = pd.Series([
    rain[rain['Month'].isin(['May','Jun','Jul'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Jun','Jul','Aug'])].apply(lambda x: wt(x,'Jul'),axis = 1)
jul = pd.Series([
    rain[rain['Month'].isin(['Jun','Jul','Aug'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Jul','Aug','Sep'])].apply(lambda x: wt(x,'Aug'),axis = 1)
aug = pd.Series([
    rain[rain['Month'].isin(['Jul','Aug','Sep'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Aug','Sep','Oct'])].apply(lambda x: wt(x,'Sep'),axis = 1)
sep = pd.Series([
    rain[rain['Month'].isin(['Aug','Sep','Oct'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Sep','Oct','Nov'])].apply(lambda x: wt(x,'Oct'),axis = 1)
oct = pd.Series([
    rain[rain['Month'].isin(['Sep','Oct','Nov'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Oct','Nov','Dec'])].apply(lambda x: wt(x,'Nov'),axis = 1)
nov = pd.Series([
    rain[rain['Month'].isin(['Oct','Nov','Dec'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])
weights = rain[rain['Month'].isin(['Nov','Dec','Jan'])].apply(lambda x: wt(x,'Dec'),axis = 1)
dec = pd.Series([
    rain[rain['Month'].isin(['Nov','Dec','Jan'])]['Rainfall'].sample(frac=.5,replace = True, weights = weights).mean()
    for i in range(n)
])


# In[8]:


# plotting some histograms to see the results
plt.hist(jan)
plt.hist(mar)
plt.hist(sep)


# In[9]:


plt.hist(feb)
plt.hist(jun)
plt.hist(oct)


# In[10]:


def simRain(month):
    """
    For a given month, simulate rainfall.

    month: integer from 1-12

    Return: random draw from the normal distribution with mean = [month_array].mean()
    and stddev = [month_array].std() (each month array is an environment variable calculated from 
    the empirical distribution of monthly rainfall).
    """
    
    if month==1:
        precip = random.normalvariate(jan.mean(),jan.std())
    elif month==2:
        precip = random.normalvariate(feb.mean(),feb.std())
    elif month==3:
        precip = random.normalvariate(mar.mean(),mar.std())
    elif month==4:
        precip = random.normalvariate(apr.mean(),apr.std())
    elif month==5:
        precip = random.normalvariate(may.mean(),may.std())
    elif month==6:
        precip = random.normalvariate(jun.mean(),jun.std())
    elif month==7:
        precip = random.normalvariate(jul.mean(),jul.std())
    elif month==8:
        precip = random.normalvariate(aug.mean(),aug.std())
    elif month==9:
        precip = random.normalvariate(sep.mean(),sep.std())
    elif month==10:
        precip = random.normalvariate(oct.mean(),oct.std())
    elif month==11:
        precip = random.normalvariate(nov.mean(),nov.std())
    elif month==12:
        precip = random.normalvariate(dec.mean(),dec.std())

    return(precip)


# ### Simulation

# In[11]:


from tqdm import tqdm

def simRain(month):
    """
    For a given month, simulate rainfall.

    month: integer from 1-12

    Return: random draw from the normal distribution with mean = [month_array].mean()
    and stddev = [month_array].std() (each month array is an environment variable calculated from 
    the empirical distribution of monthly rainfall).
    """
    
    if month==1:
        precip = random.normalvariate(jan.mean(),jan.std())
    elif month==2:
        precip = random.normalvariate(feb.mean(),feb.std())
    elif month==3:
        precip = random.normalvariate(mar.mean(),mar.std())
    elif month==4:
        precip = random.normalvariate(apr.mean(),apr.std())
    elif month==5:
        precip = random.normalvariate(may.mean(),may.std())
    elif month==6:
        precip = random.normalvariate(jun.mean(),jun.std())
    elif month==7:
        precip = random.normalvariate(jul.mean(),jul.std())
    elif month==8:
        precip = random.normalvariate(aug.mean(),aug.std())
    elif month==9:
        precip = random.normalvariate(sep.mean(),sep.std())
    elif month==10:
        precip = random.normalvariate(oct.mean(),oct.std())
    elif month==11:
        precip = random.normalvariate(nov.mean(),nov.std())
    elif month==12:
        precip = random.normalvariate(dec.mean(),dec.std())

    return(precip)
    
def simulation(niter=1000,size=25000,init_fill=10000,capture_area=3000,years = 30):
    """
    Function to simulate rainfall for the ranch over a specified period of time.

    niter: number of iterations
    size: tank size (gal)
    init_fill: starting fill of the tank (gal)
    capture_area: roof capture area (ft^2)
    years: number of years to run simulation

    return: Data frame of results with one row per iteration/year/month.
        'iteration': iteration number
        'year': year number
        'month': month number
        'start_fill': starting tank fill at the beginning of the month (gal)
        'end_fill': tank fill at the end of the month (gal)
        'month_capture': total precipitation captured in the month (gal)
        'shortage': absolute value of the difference in usage and total available water (gal)
        'overflow':difference in the total available water and the size of the tank (gal)
        'usage':amount of water used/water demand for the month,
        'rain':total rainfall for the month (in),
        'efficiency':capture efficiency for the month

    Runtime is about 40 minutes with the default parameter values
    
    """

    tank_size = size
    all_sims = pd.DataFrame()

    for i in tqdm(range(1,niter+1)):

        fill = init_fill

        for n in range(1,years+1):

            for m in range(1,13):
                
                start_fill = fill
                shortage = 0
                overflow = 0
                usage = random.uniform(4000,5200)
                efficiency = random.uniform(.90,.98)
                monthly_precip = simRain(m) # 30 days in a month

                # length x width x inches = total capture
                # using formula from here https://www.watercache.com/resources/rainwater-collection-calculator: roof area X precip X 0.623 = Gallons 
                monthly_capture = efficiency*capture_area*monthly_precip*0.623

                fill = fill - usage + monthly_capture

                if fill < 0:
                    shortage = -1*fill
                    fill = 0
                
                if fill > tank_size:
                    overflow = fill - tank_size
                    fill = tank_size

                df = pd.DataFrame(
                    {
                    'iteration':[i],
                    'year':[n],
                    'month':[m],
                    'start_fill':[start_fill],
                    'end_fill':[fill],
                    'month_capture':[monthly_capture],
                    'shortage':[shortage],
                    'overflow':[overflow],
                    'usage':[usage],
                    'rain':[monthly_precip],
                    'efficiency':[efficiency]
                    }
                )
                all_sims = all_sims.append(df)

    return(all_sims)


# In[12]:


# Simulation 1: Default
final_df = simulation(niter = 1000,size = 25000,init_fill=10000,capture_area = 3000)


# In[13]:


final_df


# In[14]:


#Shortage amounts

final_df['shortage'].plot.hist()


# In[15]:


#Distribution of number of shortages per iteration

df_short = final_df.groupby('iteration')['shortage'].apply(lambda x: (x > 0).sum()).reset_index(name='shortage_count')
df_short.set_index('iteration', inplace = True)

df_short.plot.hist()


# In[16]:


#Proportion of iterations with at least one shortage
len(df_short[df_short['shortage_count'] > 0])/max(final_df['iteration'])


# In[17]:


#Percentage number of shortages
df_short['short_freq'] = df_short['shortage_count']/1000
df_short['short_freq'].describe()


# In[18]:


#Distribution number of overflows per iteration

df_full = final_df.groupby('iteration')['end_fill'].apply(lambda x: (x == 25000).sum()).reset_index(name='full_count')
df_full.set_index('iteration', inplace = True)

df_full.describe()


# In[19]:


#Max fill in the tank per iteration

max_fill = np.array(final_df.groupby('iteration')['end_fill'].max())

df_max = final_df[final_df['end_fill'].isin(max_fill)]
df_max['end_fill'].plot.hist()


# In[20]:


#Average timeseries plot of fill

df_avg = pd.DataFrame(final_df.groupby(['year', 'month'])['end_fill'].mean())
df_avg.plot(figsize=(16, 8))


# In[21]:


df_shortages = final_df[final_df['shortage']>0]
df_shortages.groupby('month')['shortage'].count()


# ### Simulation 2: 
# #### 3500 Capture Area
# Only doing 200 iterations to reduce runtime 

# In[22]:


# Simulation 2: 3500 Capture area, original tank, original fill
df2 = simulation(niter=200,init_fill=10000,size=25000,capture_area=3500)


# In[23]:


#Distribution of number of shortages per iteration
df2_short = df2.groupby('iteration')['shortage'].apply(lambda x: (x > 0).sum()).reset_index(name='shortage_count')
df2_short.set_index('iteration', inplace = True)

print(len(df2_short[df2_short['shortage_count'] > 0])/max(df2['iteration']))
print(len(df2[df2['shortage'] > 0])/len(df2))
df2_short.plot.hist()


# In[24]:


# overflow
df2_over = df2.groupby('iteration')['overflow'].apply(lambda x: (x > 0).sum()).reset_index(name='over_count')
df2_over.set_index('iteration', inplace = True)

print(len(df2_over[df2_over['over_count'] > 0])/max(df2['iteration']*30))
print(len(df2[df2['overflow'] > 0])/len(df2))
df2_over.plot.hist()


# In[25]:


#Average timeseries plot of fill
df2_avg = pd.DataFrame(df2.groupby(['year', 'month'])['end_fill'].mean())
df2_avg.plot(figsize=(16, 8))


# ### Simulation 3: 
# #### 3500 Capture Area & 30000 Tank
# Only doing 200 iterations to reduce runtime 

# In[26]:


#Simulation 3: 3500 sqft capture area w/ 30k tank
df3 = simulation(200,30000,10000,3500)


# In[27]:


#Descriptive statistics of the shortages
df3['shortage'].describe()


# In[28]:


#Distribution of number of shortages per iteration

df3_short = df3.groupby('iteration')['shortage'].apply(lambda x: (x > 0).sum()).reset_index(name='shortage_count')
df3_short.set_index('iteration', inplace = True)

df3_short.plot.hist()
#Proportion of iterations with at least one shortage
print(len(df3_short[df3_short['shortage_count'] > 0])/max(df3['iteration']))
#Frequency of shortages across all iterations and time periods
print(len(df3[df3['shortage'] > 0])/len(df3))


# In[29]:


# overflow
df3_over = df3.groupby('iteration')['overflow'].apply(lambda x: (x > 0).sum()).reset_index(name='over_count')
df3_over.set_index('iteration', inplace = True)

# proportion of iterations with at least one overflow
print(len(df3_over[df3_over['over_count'] > 0])/max(df3['iteration']*30))
# proportion of months with overflows
print(len(df3[df3['overflow'] > 0])/len(df3))
df3_over.plot.hist()


# In[30]:


#Average timeseries plot of fill
df3_avg = pd.DataFrame(df3.groupby(['year', 'month'])['end_fill'].mean())
df3_avg.plot(figsize=(16, 8))


# ### Simulation 4: Climate Change
# - Increased rainfall (10%)
# - Decreased efficiency (5%) due to severity of rain

# In[44]:


def climate_change_simulation(niter=1000,size=25000,init_fill=10000,capture_area=3000,years = 30):
    """
    Function to simulate rainfall for the ranch over a specified period of time.

    niter: number of iterations
    size: tank size (gal)
    init_fill: starting fill of the tank (gal)
    capture_area: roof capture area (ft^2)
    years: number of years to run simulation

    return: Data frame of results with one row per iteration/year/month.
        'iteration': iteration number
        'year': year number
        'month': month number
        'start_fill': starting tank fill at the beginning of the month (gal)
        'end_fill': tank fill at the end of the month (gal)
        'month_capture': total precipitation captured in the month (gal)
        'shortage': absolute value of the difference in usage and total available water (gal)
        'overflow':difference in the total available water and the size of the tank (gal)
        'usage':amount of water used/water demand for the month,
        'rain':total rainfall for the month (in),
        'efficiency':capture efficiency for the month

    Runtime is about 40 minutes with the default parameter values
    
    """

    tank_size = size
    all_sims = pd.DataFrame()

    for i in tqdm(range(1,niter+1)):

        fill = init_fill

        for n in range(1,years+1):

            for m in range(1,13):
                
                start_fill = fill
                shortage = 0
                overflow = 0
                usage = random.uniform(4000,5200)
                
                if m >= 6 & m <=10: #Climate Change during rainy season
                    monthly_precip = simRain(m)*1.1 # Increased 10% rainfall during the rainiest months
                    efficiency = random.uniform(.85,.93) # Decreased 5% efficiency
                else: 
                    monthly_precip = simRain(m)
                    efficiency = random.uniform(.90,.98)

                # length x width x inches = total capture
                # using formula from here https://www.watercache.com/resources/rainwater-collection-calculator: roof area X precip X 0.623 = Gallons 
                monthly_capture = efficiency*capture_area*monthly_precip*0.623

                fill = fill - usage + monthly_capture

                if fill < 0:
                    shortage = -1*fill
                    fill = 0
                
                if fill > tank_size:
                    overflow = fill - tank_size
                    fill = tank_size

                df = pd.DataFrame(
                    {
                    'iteration':[i],
                    'year':[n],
                    'month':[m],
                    'start_fill':[start_fill],
                    'end_fill':[fill],
                    'month_capture':[monthly_capture],
                    'shortage':[shortage],
                    'overflow':[overflow],
                    'usage':[usage],
                    'rain':[monthly_precip],
                    'efficiency':[efficiency]
                    }
                )
                all_sims = all_sims.append(df)

    return(all_sims)


# In[45]:


df4 = climate_change_simulation(200,25000,10000,3000)


# In[46]:


#Descriptive statistics of the shortages
df4['shortage'].describe()


# In[47]:


#Distribution of number of shortages per iteration

df4_short = df4.groupby('iteration')['shortage'].apply(lambda x: (x > 0).sum()).reset_index(name='shortage_count')
df4_short.set_index('iteration', inplace = True)

df4_short.plot.hist()
#Proportion of iterations with at least one shortage
print(len(df4_short[df4_short['shortage_count'] > 0])/max(df4['iteration']))
#Frequency of shortages across all iterations and time periods
print(len(df4[df4['shortage'] > 0])/len(df4))


# In[48]:


# overflow
df4_over = df4.groupby('iteration')['overflow'].apply(lambda x: (x > 0).sum()).reset_index(name='over_count')
df4_over.set_index('iteration', inplace = True)

# proportion of iterations with at least one overflow
print(len(df4_over[df4_over['over_count'] > 0])/max(df4['iteration']*30))
# proportion of months with overflows
print(len(df4[df4['overflow'] > 0])/len(df4))
df4_over.plot.hist()


# In[49]:


#Average timeseries plot of fill
df4_avg = pd.DataFrame(df4.groupby(['year', 'month'])['end_fill'].mean())
df4_avg.plot(figsize=(16, 8))


# In[ ]:




