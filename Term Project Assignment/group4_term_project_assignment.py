#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # for random number distributions
import pandas as pd # for event_log data frame
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
import queue # add FIFO queue data structure
from functools import partial, wraps
import random
import simpy # discrete event simulation %environment
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# ### Global Variables

# In[2]:


# Raw Materials
RM_Order_Qty = 300 # units
RM_Order_Time = 30 # days
RM_Max = 450 # units

# Production
Prod_Speed = 10 # units per day per machine # assume set speed, no variation
Least_Efficiency = 8 # worst-case scenario of min number of units per day
Prod_Machine = 1 # can vary from 1 to 3

# Finished Goods
FG_Max = 100 # units
Delivery = 10 # units / day / door # assume set speed, no variation
Doors = 1 # can vary from 1 to 3

# Customers
Cust_Orders = 10 # orders / day
Order_Max = 15 # balking tolerance
Wait_Max = 2 # days

obtain_reproducible_results = True

simulation_days = 365 
fixed_simulation_time = simulation_days * 24


# In[3]:


'''        
---------------------------------------------------------
set up event tracing of all simulation program events 
controlled by the simulation environment
that is, all timeout and process events that begin with "env."
documentation at:
  https://simpy.readthedocs.io/en/latest/topical_guides/monitoring.html#event-tracing
  https://docs.python.org/3/library/functools.html#functools.partial
'''
def trace(env, callback):
     """Replace the ``step()`` method of *env* with a tracing function
     that calls *callbacks* with an events time, priority, ID and its
     instance just before it is processed.
     note: "event" here refers to simulaiton program events

     """
     def get_wrapper(env_step, callback):
         """Generate the wrapper for env.step()."""
         @wraps(env_step)
         def tracing_step():
             """Call *callback* for the next event if one exist before
             calling ``env.step()``."""
             if len(env._queue):
                 t, prio, eid, event = env._queue[0]
                 callback(t, prio, eid, event)
             return env_step()
         return tracing_step

     env.step = get_wrapper(env.step, callback)

def trace_monitor(data, t, prio, eid, event):
     data.append((t, eid, type(event)))

def test_process(env):
     yield env.timeout(1)

'''
---------------------------------------------------------
set up an event log for recording events
as defined for the discrete event simulation
we use a list of tuples for the event log
documentation at:
  https://simpy.readthedocs.io/en/latest/topical_guides/monitoring.html#event-tracing
'''

def event_log_append(env, rmid, orderid, time, activity, event_log, order_queue, rm_inv=None, fg_inv=None):
    event_log.append((rmid, orderid, time, activity, rm_inv, fg_inv, order_queue.qsize()))
    yield env.timeout(0)
    


# In[4]:


# returns real number of minutes
def random_order_time(Cust_Orders) :
    try_order_time = np.random.normal(loc = 24/Cust_Orders, scale = 1)
    if (try_order_time < 0):
        return(0)
    if (try_order_time > 2 * 24/Cust_Orders):
        return(2 * 24/Cust_Orders)
    if (try_order_time >= 0) and (try_order_time <= 2 * 24/Cust_Orders):
        return(try_order_time)

def RM(env, RM_ID, RM_inv, FG_inv, event_log, order_queue):
    while True:
        print(f'Day {env.now}: {RM_Order_Qty} of Raw Materials units arriving in {RM_Order_Time} days \n')
        yield env.timeout(RM_Order_Time)
        RM_ID += 1
        time = env.now
        activity = 'Raw Materials arrival'
        env.process(event_log_append(
            env,RM_ID, None, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level))
        yield env.timeout(0)
        if RM_inv.level + RM_Order_Qty < RM_inv.capacity:
            RM_inv.put(RM_Order_Qty)
            print(f'Accepted RM order# {RM_ID} @ {env.now}')
            time = env.now
            activity = 'Raw Materials put in storage'
            env.process(event_log_append(
                env,RM_ID, None, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
        else:
            print(f'Rejected RM order# {RM_ID} @ {env.now}')
            time = env.now
            activity = 'Raw Materials rejected'
            env.process(event_log_append(
                env,RM_ID, None, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 

def prod(env, RM_inv, FG_inv, event_log, order_queue):
    while True:
        print(f"Start production for day {env.now}...")
        fg_yield = random.randint(Least_Efficiency, Prod_Speed)
        if RM_inv.level - Prod_Speed * Prod_Machine >= 0 and FG_inv.level + fg_yield * Prod_Machine <= FG_inv.capacity:
            print(f"{fg_yield * Prod_Machine} units produced on day {env.now}\n")
            RM_inv.get(Prod_Speed * Prod_Machine)
            FG_inv.put(fg_yield * Prod_Machine)
        else:
            print(f"0 units produced (insufficienct inventory/capacity) on day {env.now}\n")
            print('End Production.')
        yield env.timeout(1)
        time = env.now
        activity = 'Finished goods production completion'
        env.process(event_log_append(
            env,None, None, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
        yield env.timeout(0)

def inv_report(env, RM_inv, FG_inv, event_log, order_queue):
    while True:
        print(f"Day {env.now}: \n\tRaw Materials Inventory = {RM_inv.level}")
        print(f"\tFinished Goods Inventory = {FG_inv.level}\n")
        time = env.now
        activity = 'inventory report'
        env.process(event_log_append(
            env,None, None, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
        yield env.timeout(1)

def order(env, orderid, order_queue, RM_inv, FG_inv, event_log):
    while True:
        next_order_time = random_order_time(Cust_Orders)/24 #converting hours to days
        yield env.timeout(next_order_time)
        orderid += 1
        time = env.now
        activity = 'order process started'
        env.process(event_log_append(
            env,None, orderid, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
        yield env.timeout(0)
        if order_queue.qsize() < Order_Max:
            order_queue.put(orderid)
            print(f"Order# {orderid} placed @ {round(env.now,2)} \n\t Order queue length = {order_queue.qsize()}")
            time = env.now
            activity = 'order placed'
            env.process(event_log_append(
                env,None, orderid, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
            env.process(fulfillment(env,order_queue, RM_inv, FG_inv, event_log))
        else:
            print(f"Order# {orderid} not placed @ {round(env.now,2)} \n\t Order queue length = {order_queue.qsize()}")
            time = env.now
            activity = 'order not placed'
            env.process(event_log_append(
                env,None, orderid, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 

def fulfillment(env,order_queue, RM_inv, FG_inv, event_log):
    with available_doors.request() as req:
        yield req  # wait until the request can be met.. there must be an available door
        orderid = order_queue.get()
        while FG_inv.level <= 0:
            # just want to 'peek' at front of queue by using .queue[0]
            # .get() will take out the item in front of the queue...!
            print(f"Begin fulfillment of order# {orderid}") 
            print(f"Stop fulliment of order# {orderid} (insufficient FG inventory)\n") 
            time = env.now
            activity = 'fulfillment - insufficient inventory - wait'
            env.process(event_log_append(
                env,None, orderid, time, activity, event_log,order_queue,rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
            yield env.timeout(1) # skip the day/check tomorrow if inventory is empty...?            
        else:
            FG_inv.get(1)
            print(f"Begin fulfillment of order# {orderid}") 
            print(f"Loading  order# {orderid} @ {round(env.now,3)}; order queue length = {order_queue.qsize()}\n")
            time = env.now
            Loading = random.uniform(1/24, 3/24) # loading anywhere between 1 and 3 hours
            env.process(event_log_append(
                env,None, orderid, env.now, 'fulfillment - begin loading', event_log,order_queue,
                rm_inv=RM_inv.level,fg_inv=FG_inv.level)) 
            yield env.timeout(Loading)  # sets generator function
            queue_length_post_loading = order_queue.qsize()
            print(f"Finished loading order# {orderid} @ {round(env.now,3)}; order queue length = {queue_length_post_loading}\n")
            env.process(event_log_append(
                env,None, orderid, env.now, 'fulfillment - finish loading', event_log,order_queue,
                rm_inv=RM_inv.level,fg_inv=FG_inv.level))


# # Simulation #1:

# In[5]:


parameter_string_list = [str(simulation_days),'days',
              str(RM_Order_Qty),str(RM_Order_Time),
              str(RM_Max),str(Prod_Machine),
              str(FG_Max),str(Doors),
              str(Cust_Orders),str(Order_Max),
              str(Wait_Max)]
separator = '-'        
simulation_file_identifier = separator.join(parameter_string_list)

if obtain_reproducible_results: 
    np.random.seed(12345)

# set up simulation trace monitoring for the simulation
data = []
# bind *data* as first argument to monitor()
this_trace_monitor = partial(trace_monitor, data)

env = simpy.Environment()
trace(env, this_trace_monitor)

order_queue = queue.Queue()
available_doors = simpy.Resource(env, Doors)
RM_inv = simpy.Container(env, capacity = RM_Max, init = 300)
FG_inv = simpy.Container(env, capacity = FG_Max, init = 0)

event_log = [(0,0,0,'init',RM_inv.level,FG_inv.level, 0)]
env.process(event_log_append(env, 0, 0, env.now, 'start simulation', event_log, order_queue,
                             RM_inv.level,FG_inv.level))
env.process(inv_report(env, RM_inv, FG_inv, event_log, order_queue))
env.process(RM(env, 0, RM_inv, FG_inv, event_log, order_queue))
env.process(prod(env, RM_inv, FG_inv, event_log, order_queue))
env.process(order(env, 0, order_queue, RM_inv, FG_inv, event_log))
env.run(until = 365)


# In[6]:


simulation_trace_file_name = 'simulation-program-trace-' + simulation_file_identifier + '.txt'
with open(simulation_trace_file_name, 'wt') as ftrace:
    for d in data:
        print(str(d), file = ftrace)

print()        
print('simulation program trace written to file:',simulation_trace_file_name)

event_log_list = [list(element) for element in event_log]
event_log_df = pd.DataFrame(event_log_list,columns = ['rm_id','order_id','time','activity','rm_inventory','fg_inventory','order_queue_size'])
event_log_file_name = 'simulation-event-log-' + simulation_file_identifier + '.csv'
event_log_df.to_csv(event_log_file_name, index = False)  

print()        
print('simulation program trace written to file:',event_log_file_name)


# In[7]:


figure(figsize = (20, 10), dpi = 80)
plt.plot(event_log_df.time, event_log_df.rm_inventory, label='raw_materials')
plt.plot(event_log_df.time, event_log_df.fg_inventory, label='finished goods')
plt.ylabel('count')
plt.xlabel('day')
plt.legend()
plt.show()


# In[8]:


figure(figsize = (20, 10), dpi = 80)
plt.plot(event_log_df.time, event_log_df.order_queue_size, label='queue size')
plt.ylabel('count')
plt.xlabel('day')
plt.legend()
plt.show()


# In[9]:


event_log_df.groupby('activity').size()


# In[10]:


order_time_log = []

ffinish_load_df = event_log_df[(event_log_df['activity'] == 'fulfillment - finish loading')]

for index, row in ffinish_load_df.iterrows():
    order_id = row.order_id
    end = row.time
    start_row = event_log_df[(event_log_df['order_id'] == order_id) & (event_log_df['activity'] == 'order placed')]
    start = start_row.time.values[0]
    duration = end - start
    order_time_log.append((int(order_id), start, end, duration))

order_time_log_list = [list(element) for element in order_time_log]
order_time_log_df = pd.DataFrame(order_time_log_list,columns = ['order_id','start','stop','duration'])

# avg order duration
order_time_log_df.duration.mean()


# # Simulation # 2 (increase # of doors):

# In[11]:


Doors = 2


# In[12]:


parameter_string_list = [str(simulation_days),'days',
              str(RM_Order_Qty),str(RM_Order_Time),
              str(RM_Max),str(Prod_Machine),
              str(FG_Max),str(Doors),
              str(Cust_Orders),str(Order_Max),
              str(Wait_Max)]
separator = '-'        
simulation_file_identifier = separator.join(parameter_string_list)

if obtain_reproducible_results: 
    np.random.seed(12345)

# set up simulation trace monitoring for the simulation
data = []
# bind *data* as first argument to monitor()
this_trace_monitor = partial(trace_monitor, data)

env = simpy.Environment()
trace(env, this_trace_monitor)

order_queue = queue.Queue()
available_doors = simpy.Resource(env, Doors)
RM_inv = simpy.Container(env, capacity = RM_Max, init = 300)
FG_inv = simpy.Container(env, capacity = FG_Max, init = 0)

event_log = [(0,0,0,'init',RM_inv.level,FG_inv.level, 0)]
env.process(event_log_append(env, 0, 0, env.now, 'start simulation', event_log, order_queue,
                             RM_inv.level,FG_inv.level))
env.process(inv_report(env, RM_inv, FG_inv, event_log, order_queue))
env.process(RM(env, 0, RM_inv, FG_inv, event_log, order_queue))
env.process(prod(env, RM_inv, FG_inv, event_log, order_queue))
env.process(order(env, 0, order_queue, RM_inv, FG_inv, event_log))
env.run(until = 365)


# In[13]:


simulation_trace_file_name = 'simulation-program-trace-' + simulation_file_identifier + '.txt'
with open(simulation_trace_file_name, 'wt') as ftrace:
    for d in data:
        print(str(d), file = ftrace)

print()        
print('simulation program trace written to file:',simulation_trace_file_name)

event_log_list = [list(element) for element in event_log]
event_log_df = pd.DataFrame(event_log_list,columns = ['rm_id','order_id','time','activity','rm_inventory','fg_inventory','order_queue_size'])
event_log_file_name = 'simulation-event-log-' + simulation_file_identifier + '.csv'
event_log_df.to_csv(event_log_file_name, index = False)  

print()        
print('simulation program trace written to file:',event_log_file_name)


# In[14]:


figure(figsize = (20, 10), dpi = 80)
plt.plot(event_log_df.time, event_log_df.rm_inventory, label='raw_materials')
plt.plot(event_log_df.time, event_log_df.fg_inventory, label='finished goods')
plt.ylabel('count')
plt.xlabel('day')
plt.legend()
plt.show()


# In[15]:


figure(figsize = (20, 10), dpi = 80)
plt.plot(event_log_df.time, event_log_df.order_queue_size, label='queue size')
plt.ylabel('count')
plt.xlabel('day')
plt.legend()
plt.show()


# In[16]:


event_log_df.groupby('activity').size()


# In[17]:


order_time_log = []

ffinish_load_df = event_log_df[(event_log_df['activity'] == 'fulfillment - finish loading')]

for index, row in ffinish_load_df.iterrows():
    order_id = row.order_id
    end = row.time
    start_row = event_log_df[(event_log_df['order_id'] == order_id) & (event_log_df['activity'] == 'order placed')]
    start = start_row.time.values[0]
    duration = end - start
    order_time_log.append((int(order_id), start, end, duration))

order_time_log_list = [list(element) for element in order_time_log]
order_time_log_df = pd.DataFrame(order_time_log_list,columns = ['order_id','start','stop','duration'])

# avg order duration
order_time_log_df.duration.mean()

