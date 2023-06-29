# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:20:11 2023

@author: luisg
"""

from MINIPIX_dataframe import *
from MINIPIX_ANALISIS import *
from MINIPIX_ANALISIS_MULTIVARIABLE import *

import MINIPIX_ANALISIS as pixanaly
import MINIPIX_ANALISIS_MULTIVARIABLE as pixanalymult
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")

#%%
df = pd.read_excel('dataframes/df.xlsx')
df_s = pd.read_excel('dataframes/df_s.xlsx')
df_t = pd.read_excel('dataframes/df_t.xlsx')
df_t1 = pd.read_excel('dataframes/df_t1.xlsx')
df_class = pd.read_excel('dataframes/df_class.xlsx')
particle_names=['muons','e','alphas']
particle_name=particle_names[0]


variables=['radiacion solar','lluvia','presion', 'temperatura']
variable=variables[3]

n_targets=2
new_data=0
test_size=0.2
dataframe_name=df_class
#filtrado de los datos buenos
# j=[50,200]
# for i in range(2):
#     dataframe_filtered=dataframe_name[dataframe_name[particle_names[i]]<j[i]][particle_names[i]]
#     dataframe_name=dataframe_name[dataframe_name.index.isin(dataframe_filtered.index)].reset_index()

dataframe_filtered=dataframe_name[dataframe_name[particle_name]<50][particle_name]
dataframe_name=dataframe_name[dataframe_name.index.isin(dataframe_filtered.index)].reset_index()
# dataframe_name=df
y_ydots_lim, y_dots_lim=0.01,1000000 #10000


plt.close('all')
# pixanaly.PCA_analysis(particle_name,dataframe_name, 3, y_ydots_lim,y_dots_lim)
# pixanaly.PCA_dataframe(particle_name,dataframe_name, 3, y_ydots_lim,y_dots_lim)
# pixanaly.heatmap(particle_name,dataframe_name,y_ydots_lim,y_dots_lim)
# pixanaly.box_plot(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)
# pixanaly.pair_plot(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)
# pixanaly.regresion_lineal_univariable(particle_name, dataframe_name,variable, y_ydots_lim, y_dots_lim) 

# pixanaly.class_1d_2d(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.kmean_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.svm_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.svm_uniclass_2(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,new_data,n_targets,test_size)
# pixanaly.randomforest_uniclas(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.neuralnetwork_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim, new_data,n_targets,test_size)
# pixanaly.xgbosting_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,n_targets)
#pixanaly.xgbosting_uniclass_cv(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,n_targets)  #only works with two targets
# pixanaly.naive_bayes_uniclass(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, n_targets)
# pixanaly.neuralnetwork_regression(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.neuralnetwork_regression_2(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim)
# pixanaly.dbscan_uniclass(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets)

def see_data(x):
    accuracy_list=[]
    confmat_list=[]
    conjunto=[]
    for i in particle_names:
        for j in variables:
            plt.close('all')
            particle_name=i
            variable=j
            # accuracy,confmat=x(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,new_data,n_targets,test_size)
            accuracy,confmat=x(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,n_targets)
            accuracy_list.append(accuracy)
            confmat_list.append(confmat)
            conjunto.append(i+j)
             
    for i in range(len(conjunto)):
        print('=============')
        print('conjunto',conjunto[i], ' accuracy: ', accuracy_list[i]) 
        print(confmat_list[i])

# see_data(pixanaly.xgbosting_uniclass_cv)
plt.show()
#%%

#al proyectar los tres s i que da boen
particle_name=['muons','e','alphas']

# particle_name=['muons','e']
variables=['radiacion solar','lluvia','presion', 'temperatura']
variable=variables[3]
n_targets=6
new_data=0
test_size=0.2
dataframe_name=df_class

j=[50,200]
for i in range(2):
    dataframe_filtered=dataframe_name[dataframe_name[particle_name[i]]<j[i]][particle_name[i]]
    dataframe_name=dataframe_name[dataframe_name.index.isin(dataframe_filtered.index)].reset_index()


y_ydots_lim, y_dots_lim=0.01,1000000 #10000

# pixanalymult.svm_bivar(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size)
# pixanalymult.svm_trivar(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size)
# pixanalymult.svm_multivar(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size)
# df_class_multivar=pixanalymult.multilabel(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size)
# pixanalymult.xgboost_multivar_cv(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size)
# pixanalymult.xgbosting_uniclass_multivariable_cv(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, n_targets)


#%%

