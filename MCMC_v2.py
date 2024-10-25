# -*- coding: utf-8 -*-
"""
Created on Thu May  5 04:23:42 2022

@author: muazi
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# https://github.com/thu-vu92/python-dashboard-panel/blob/main/Interactive_dashboard.ipynb

#%%

# D:/OneDrive - ubd.edu.bn/Dataset/
# C:/Users/Muaz/OneDrive - ubd.edu.bn/Dataset/
# C:/Users/Simulation/muaz/MCMC/

# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--path', default='C:/Users/Simulation/muaz/MCMC/'\
#                  'Population statistics for Seoul UTF-8/'\
#                     'Seoul Synthetic Population (10%).csv',
#                     help='an integer for the accumulator')
# parser.add_argument('--lr', default=0.99,
#         help='an integer for the accumulator')
# parser.add_argument('--exp', default='C:/Users/Simulation/muaz/MCMC/'\
#      'Population statistics for Seoul UTF-8/'\
#         'Seoul Synthetic Population (10%).csv',
#         help='an integer for the accumulator')

# args = parser.parse_args()

# print(args)

df_ipf = pd.read_csv('C:/Users/Simulation/muaz/POP SYNTHESIS/MCMC/'\
                    'Population statistics for Seoul UTF-8/'\
                      'Seoul Synthetic Population (10%).csv')
# df_ipf = df_ipf[df_ipf["ws"]!=4]
print(df_ipf.head(5))

#%%
print('before: ',np.sort(df_ipf['ws'].unique()))
df_ipf.loc[df_ipf['ws']==2, 'ws'] =  1
df_ipf.loc[df_ipf['ws']==3, 'ws'] =  2
df_ipf.loc[df_ipf['ws']==4, 'ws'] =  3
print('after: ',np.sort(df_ipf['ws'].unique()))

#%%
# C:/Users/Muaz/OneDrive - ubd.edu.bn/Dataset/
# D:/OneDrive - ubd.edu.bn/Dataset/
df_sample = pd.read_csv(r'C:/Users/Simulation/muaz/POP SYNTHESIS/MCMC/'\
                        r'Population statistics for Seoul UTF-8/'\
                        r'raw sample.csv')
df_sample.rename(columns = {'gu_code':'home_gu'}, inplace = True)
print(df_sample.head(5))

#%% Get the unique labels for each attribute

x_target = ['sex', 'age_type', 'ws', 'home_gu', 'ws_gu'] 
# x_target = ['ws', 'sex', 'age_type', 'fam_type','home_gu','ws_gu'] 

def get_x_labels(df_sample, df_ipf, x_target):
    x_labels_dict = {}
    for i in x_target:
        # print('attribute: ',i)
        if i != 'ws_gu':
            temp = df_sample[i].unique()
            temp.sort()
            print(str(i)+': '+str(temp))
        else:
            temp = df_ipf[i].unique()
            temp.sort()
            print(str(i)+': '+str(temp))
            
        x_labels_dict.setdefault(str(i)+'_group', temp)
        
    return x_labels_dict

x_labels_dict = get_x_labels(df_sample, df_ipf, x_target)

# #%%

# # Perform a partial conditional for the "ws_gu"
# x_init = ['ws', 'sex', 'age_type', 'fam_type', 'home_gu']
# df_sample = df_sample[x_init]

# # def create_variable_attribute(var):
# #     keys = []
# #     for i in var:
# #         keys.append(i) # 'x'+str(i)
# #     return keys

# # x_target = ['ws', 'sex', 'age_type', 'home_gu','ws_gu']

# # keys_init = create_variable_attribute(x_init)
# # keys_target = create_variable_attribute(x_target)
# # print('keys', keys_target)

# idx_x = list(range(len(x_labels_dict.keys()))) 
# print(idx_x)

#%% create a crosstab

def get_crosstab(df_sample, df_ipf, x_target):
    crosstab_dict = {}
    
    for i in x_target:
        
        if i != 'ws_gu':
            Y_list = [x for x in x_target if i != x and 'ws_gu'!=x]
            print('`'+str(i)+'` given '+str(Y_list))
            
            X = df_sample[i]
            Y = []
            for j in Y_list:
                Y_temp = df_sample[j]
                Y.append(Y_temp)   
        
        else:
            Y_list = [x for x in x_target if i != x]
            print('`'+str(i)+'` given '+str(Y_list))
            
            X = df_ipf[i]
            Y = []
            for j in Y_list:
                Y_temp = df_ipf[j]
                Y.append(Y_temp)   
                
        print(len(X), len(Y))
        
        crosstab_temp = pd.crosstab(X,Y,
                                    margins=True, normalize='columns')       
        
        crosstab_dict.setdefault('crosstab_'+str(i), crosstab_temp)
                
    return crosstab_dict

crosstab_dict = get_crosstab(df_sample, df_ipf, x_target)
    
#%% MCMC Seed initialization using Random sampling based on probabilities 

def get_seed(df_sample, x_target, x_labels_dict, crosstab_dict):
    Y_list = [x for x in x_target if 'ws_gu'!=x]
    X_init = df_sample[Y_list].sample().values.tolist()[0]
    print('initialize: ', X_init)
    
    df_ws_gu = crosstab_dict.get('crosstab_ws_gu')
    ws_gu_group = x_labels_dict.get('ws_gu_group')
    
    # Extracting a set of probabilities from the crosstab
    # get_Xtab_cond = df_ws_gu.loc[df_ws_gu.index.get_level_values('ws_gu')==ws_gu_group,
    #                 (df_ws_gu.columns.get_level_values('ws') == X_init[0]) &
    #                 (df_ws_gu.columns.get_level_values('sex') == X_init[1]) &
    #                 (df_ws_gu.columns.get_level_values('age_type') == X_init[2]) &
    #                 (df_ws_gu.columns.get_level_values('fam_type') == X_init[3]) &
    #                 (df_ws_gu.columns.get_level_values('home_gu') == X_init[4])
    #                 ].values
    
    get_Xtab_cond = df_ws_gu.loc[df_ws_gu.index.get_level_values('ws_gu')==ws_gu_group,
                    (df_ws_gu.columns.get_level_values('sex') == X_init[0]) &
                    (df_ws_gu.columns.get_level_values('age_type') == X_init[1]) &
                    (df_ws_gu.columns.get_level_values('ws') == X_init[2]) &
                    (df_ws_gu.columns.get_level_values('home_gu') == X_init[3])
                    ].values

    p_set = [item for sublist in get_Xtab_cond for item in sublist]
    print(p_set) # this is a set of probabilities
    
    print(np.random.choice(ws_gu_group, 1, p=p_set))  
    
    X_init.append(np.random.choice(ws_gu_group, 1, p=p_set)[0])
    print('initialize_v2: ', X_init)
    
    return X_init

X_init = get_seed(df_sample, x_target, x_labels_dict, crosstab_dict)

#%%

pop_target = 1000538 # 1000538
burn_period = 20000
n_corr = 10 # sampling frequency to avoid correlation
X_sampled = []
X_sampled.append(X_init)
X_sampled_final = []

cnt = 0
agent_idx = 0

# burn in period, skipping iterations
# after warm up, draw an agent from every 20th iteration to avoid any correlation between the successive draws

while len(X_sampled_final) < pop_target:
    # print('\niteration: ',agent_idx) # \n agent_idx
    X_prev_agent = X_sampled[-1]
    X_prev_agent = dict(zip(x_target, X_prev_agent))
    X_new_agent = dict.fromkeys(x_target)
    # print('X_prev_agent: '+str(X_prev_agent))
    # print('X_new_agent: '+str(X_new_agent))
    
    for att_idx in x_target:
        # return value that are not the same as the listed value
        # cond_list = list(set(idx_x)-set([att_idx]))
        # print('\nFor `'+str(att_idx)+'` given '+str(cond_list))
        # cond_list = [x_target[i] for i in cond_list]
        
        cond_list = [x for x in x_target if att_idx != x]
        # print('For `'+att_idx+'` given '+str(cond_list))
        cond_val = [] 
        
        # check each attribute in X_new_agent dictionary
        for cond_idx in cond_list:
            # print('Check ('+str(cond_idx)+') attribute in X_new_agent')
            if not X_new_agent.get(str(cond_idx)): # If empty use the attribute from the previous agent (X_prev_agent)
                # print('`'+str(cond_idx)+'` is not available in X_new_agent, so use attribute from X_prev_agent')
                cond_val.append(X_prev_agent.get(str(cond_idx)))

            else: # if the given attribute is not found in the list
                # print('`'+str(cond_idx)+'` is available in X_new_agent')
                cond_val.append(X_new_agent.get(str(cond_idx)))

        # print('conditional value order: ',cond_val)
        
        crosstab = crosstab_dict.get('crosstab_'+att_idx)
        group = x_labels_dict.get(att_idx+'_group')
        
        # Extracting a set of probabilities from the crosstab
        if att_idx == 'ws_gu': # Full conditionals
            get_Xtab_cond = crosstab.loc[(crosstab.index.get_level_values(att_idx) == group),
                                                    (crosstab.columns.get_level_values(cond_list[0]) == cond_val[0]) &
                                                    (crosstab.columns.get_level_values(cond_list[1]) == cond_val[1]) &
                                                    (crosstab.columns.get_level_values(cond_list[2]) == cond_val[2]) &
                                                    (crosstab.columns.get_level_values(cond_list[3]) == cond_val[3])
                                                    ].values
            
        else: # Partial conditionals without ws_gu
            get_Xtab_cond = crosstab.loc[(crosstab.index.get_level_values(att_idx) == group),
                                                    (crosstab.columns.get_level_values(cond_list[0]) == cond_val[0]) &
                                                    (crosstab.columns.get_level_values(cond_list[1]) == cond_val[1]) &
                                                    (crosstab.columns.get_level_values(cond_list[2]) == cond_val[2])
                                                    ].values
              
        
        if not np.size(get_Xtab_cond):
            # print('if array is empty')
            x_sampled = X_new_agent['home_gu']
        else:
            p_set = [item for sublist in get_Xtab_cond for item in sublist]
            # if att_idx == 'sex':
            #     print(get_Xtab_cond)
            # print('group: {}; p_set: {}'.format(group,p_set)) # this is a set of probabilities
            
            x_sampled = np.random.choice(group, 1, p=p_set)[0] # sample with replacement     
            # print(str(att_idx)+' = '+str(x_sampled)) 
    
        X_new_agent[att_idx] = x_sampled
        # print('Sampled new attribute to an agent: ', X_new_agent)
        
    X_sampled.append([val for key, val in X_new_agent.items()])
    # print(X_sampled)

    # X_sampled_final.append([val for key, val in X_new_agent.items()])
    X_sampled.pop(0)
    agent_idx += 1
    
    if agent_idx > burn_period:
        # print('Stationary #', str(agent_idx))
        # To avoid the correlation between the consecutive draws, 
        # certain number of draws between two recorded draws are skipped.
        cnt += 1
        if cnt == n_corr:
            X_sampled_final.append([val for key, val in X_new_agent.items()])
            cnt = 0
            print('Agent ID#'+str(len(X_sampled_final))+': '+str(X_new_agent)+'')
            
    else:
        print('Burn-in period #',str(agent_idx))
    # print() 
    
    # cnt += 1
    # if cnt == 2:
    #     break

#%%Plot
# ========================================
# pop_sampled = X_sampled_final[burn_period:]
df_sampled = pd.DataFrame(X_sampled_final,columns=x_target)
# ['ws', 'sex', 'age_type', 'home_gu','ws_gu']

fig, axes = plt.subplots(3, 5, figsize=(25, 20))
plt.subplots_adjust(hspace=0.2, wspace=0.4)
# fig.suptitle('Population Synthesis Distribution')


sns.histplot(ax=axes[0,0], data=df_sample['sex'])
sns.histplot(ax=axes[0,1], data=df_sample['age_type'])
sns.histplot(ax=axes[0,2], data=df_sample['ws'])
sns.histplot(ax=axes[0,3], data=df_sample['home_gu'])
sns.histplot(ax=axes[0,4], data=df_ipf['ws_gu'])


sns.histplot(ax=axes[1,0], data=df_ipf['sex'])
sns.histplot(ax=axes[1,1], data=df_ipf['age_type'])
sns.histplot(ax=axes[1,2], data=df_ipf['ws'])
sns.histplot(ax=axes[1,3], data=df_ipf['home_gu'])
sns.histplot(ax=axes[1,4], data=df_ipf['ws_gu'])


sns.histplot(ax=axes[2,0], data=df_sampled['sex'])
sns.histplot(ax=axes[2,1], data=df_sampled['age_type'])
sns.histplot(ax=axes[2,2], data=df_sampled['ws'])
sns.histplot(ax=axes[2,3], data=df_sampled['home_gu'])
sns.histplot(ax=axes[2,4], data=df_sampled['ws_gu'])

df_sampled.to_csv(r'C:/Users/Simulation/muaz/MCMC/'\
                 r'Population statistics for Seoul UTF-8/'\
                     r'MCMC (10%) 10v4.csv', index=False)

# df_ipf.to_csv(r'C:/Users/Simulation/muaz/MCMC/'\
#                  r'Population statistics for Seoul UTF-8/'\
#                      r'IPF (10%) 20v3.csv', index=False)


#%%
s
df_ipf2 = pd.read_csv(r'C:/Users/Simulation/muaz/MCMC/'\
                 r'Population statistics for Seoul UTF-8/'\
                     r'Seoul Synthetic Population (10%).csv')
    
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(hspace=0.2, wspace=0.4)
sns.histplot(ax=axes[0], data=df_ipf2['ws'])
sns.histplot(ax=axes[1], data=df_ipf2['sex'])
sns.histplot(ax=axes[2], data=df_ipf2['age_type'])
sns.histplot(ax=axes[3], data=df_ipf2['home_gu'])
sns.histplot(ax=axes[4], data=df_ipf2['ws_gu'])


#%%

# Slice dataframe when "ws" is 3
# df_ws3 = df[(df["ws"]==3)]

# # Algorithm to get the ratio
# import collections
# ws_gu_dict = {}
# for i in sex_group:
#     # print('sex: ', i)
#     for j in age_type_group:
#         if j == 1 or j == 4: # if children or retired
#             continue
#         else:
#             # print('age_type:', j)
#             for k in home_gu_group:
#                 # print('home_gu', k)
#                 # Slice dataframe, get the frequency/ratio, convert to a dictionary
#                 series_to_dict = df_ws3["ws_gu"].loc[(df_ws3["sex"]==i) & (df_ws3["age_type"]==j) & (df_ws3["home_gu"]==k)].value_counts(normalize=True).to_dict()
#                 series_to_dict = collections.OrderedDict(sorted(series_to_dict.items()))
#                 print('\nsex('+str(i)+') | age_type('+str(j)+') | home_gu('+str(k)+'): '+str(series_to_dict))
                
#                 # Insert to a new dictionary
#                 ws_gu_dict.setdefault('sex('+str(i)+') | age_type('+str(j)+') | home_gu('+str(k)+')', series_to_dict)

# # Draw a random sample based on the given probability
# key = list(ws_gu_dict.get('sex('+str(1)+') | age_type('+str(3)+') | home_gu('+str(10)+')').keys())
# p = list(ws_gu_dict.get('sex('+str(1)+') | age_type('+str(3)+') | home_gu('+str(10)+')').values())
# print('ws_gu: '+str(np.random.choice(key, 1, p=p)[0]))

##%%

# df_ws3_gu = pd.crosstab(df_ws3['ws_gu'],
#                     [df_ws3['sex'], df_ws3['age_type'], df_ws3['home_gu']],
#                     margins=True, normalize='columns')  


# #%% Cross-tabulation

# df_ws = pd.crosstab(df_sample['ws'],
#                         [df_sample['sex'], df_sample['age_type'], df_sample['home_gu']],
#                         margins=True, normalize='columns')      

# df_sex = pd.crosstab(df_sample['sex'],
#                         [df_sample['ws'], df_sample['age_type'], df_sample['home_gu']],
#                         margins=True, normalize='columns')      

# df_age_type = pd.crosstab(df_sample['age_type'],
#                         [df_sample['ws'], df_sample['sex'], df_sample['home_gu']],
#                         margins=True, normalize='columns')      

# df_home_gu = pd.crosstab(df_sample['home_gu'],
#                         [df_sample['ws'], df_sample['sex'], df_sample['age_type']],
#                         margins=True, normalize='columns')   

# # --------------------------
# # ws_gu from the IPF 10% data, get 0,1,2,3
# df_ws_gu = pd.crosstab(df_ipf['ws_gu'],
#                     [df_ipf['ws'], df_ipf['sex'], df_ipf['age_type'], df_ipf['home_gu']],
#                     margins=True, normalize='columns')    

#%%

# def assign_cond(X_init, Y_list, df_ws_gu):
#     cond_list = []
#     idx = 0
#     for i in Y_list:
#         print(i)
#         temp = df_ws_gu.columns.get_level_values(i) == X_init[idx]
#         cond_list.append(temp)
#         idx+=1
#     return cond_list

# cond_list = assign_cond(X_init, Y_list, df_ws_gu)

# print(cond_list)

# check = []
# cond_list_str = [str(item) for item in cond_list]
# # # check = [' & '.join([str(item) for item in cond_list])][0]
# # letters = ['a','b','c','d','e','f','g','h','i','j']
# # print((''.join(l + ' & ' * (n % 1 == 0) for n, l in enumerate(cond_list_str))))


# # print()
# # print(check)


# for n, l in enumerate(cond_list_str):
#     print(n, l)
#     print(type(l))
#     # print(l.astype(np.bool))
    
#     check.append(l)
    
#     if n!=len(cond_list_str)-1:
#         check.append('&')
    
# print(check[0])    

# a = "".join(check)

#%%


















