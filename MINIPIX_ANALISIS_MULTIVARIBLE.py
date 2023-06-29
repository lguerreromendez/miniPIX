# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 03:24:51 2023

@author: luisg
"""

from MINIPIX_dataframe import *
import MINIPIX_dataframe as pixdata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,mean_squared_error,confusion_matrix,r2_score,accuracy_score

import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold
#DATA
df = pd.read_excel('dataframes/df.xlsx')
df_s = pd.read_excel('dataframes/df_s.xlsx')
df_t = pd.read_excel('dataframes/df_t.xlsx')
df_t1 = pd.read_excel('dataframes/df_t1.xlsx')
df_class = pd.read_excel('dataframes/df_class.xlsx')

# particle_names=['alphas','e','muons']
# particle_name=particle_names[2]


# variables=['radiacion solar','lluvia','presion', 'temperatura']
# variable=variables[3]
# n_targets=2
# new_data=0
# test_size=0.2
# dataframe_name=df_class
# y_ydots_lim, y_dots_lim=0.01,1000000 #10000

lim_newdata='nada'
def data_y_X_2D(particle_name, dataframe_name,y_ydots_lim, y_dots_lim):
    def eliminar_datos_2D(y,ydots):
        index_list_delete=[]
        for i in range(len(y)):
            if y[i]/ydots[i]>y_ydots_lim and ydots[i]>y_dots_lim :
                index_list_delete.append(i)
        return index_list_delete
            
    if len(particle_name)==2:
        y1=dataframe_name[particle_name[0]]
        y2=dataframe_name[particle_name[1]]
        ydots=dataframe_name['dots']
        
        index_list_1=eliminar_datos_2D(y1,ydots)
        index_list_2=eliminar_datos_2D(y2,ydots)
        index_list = np.union1d(index_list_1, index_list_2)
        y2=y2.drop(index_list)
        y1=y1.drop(index_list)
        return y1,y2
    if len(particle_name)==3:
        y1=dataframe_name[particle_name[0]]
        y2=dataframe_name[particle_name[1]]
        y3=dataframe_name[particle_name[2]]
        ydots=dataframe_name['dots']
        
        index_list_1=eliminar_datos_2D(y1,ydots)
        index_list_2=eliminar_datos_2D(y2,ydots)
        index_list_3=eliminar_datos_2D(y3,ydots)
        index_list_4 = np.union1d(index_list_1, index_list_2)
        index_list=np.union1d(index_list_4, index_list_3)
        y2=y2.drop(index_list)
        y1=y1.drop(index_list)
        y3=y3.drop(index_list)
        return y1,y2,y3

def pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim):
    y=dataframe_name[variable] #categoria de la variable del tiempo
    x1=dataframe_name[particle_name[0]] #  cantidad de muones   sin eliminar datos
    if len(particle_name)>2:
        if isinstance(lim_newdata, int)==True:
            x3=dataframe_name[particle_name[2]]
            x2=dataframe_name[particle_name[1]]
            x1=dataframe_name[particle_name[0]]
            y=np.array(y)
            x3=x3[lim_newdata:];x2=x2[lim_newdata:];x1=x1[lim_newdata:];y=y[lim_newdata:]
            return x1,x2,x3,y
        else: 
            x3=dataframe_name[particle_name[2]]
            x2=dataframe_name[particle_name[1]]
            x1=dataframe_name[particle_name[0]]
            x1,x2,x3=data_y_X_2D(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)
            y=y.loc[y.index.isin(x1.index)]
            y=np.array(y)
            return x1,x2,x3,y
    if len(particle_name)>1:
        if isinstance(lim_newdata, int)==True:
            x2=dataframe_name[particle_name[1]]
            x1=dataframe_name[particle_name[0]]
            x1,x2=data_y_X_2D(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)
            y=np.array(y)
            x2=x2[lim_newdata:];x1=x1[lim_newdata:];y=y[lim_newdata:]
            return x1,x2,y
        else: 
            x2=dataframe_name[particle_name[1]]
            x1=dataframe_name[particle_name[0]]
            x1,x2=data_y_X_2D(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)
            
            y=y.loc[y.index.isin(x1.index)]
            y=np.array(y)
            return x1,x2,y

#con eslado pero sin tsne
###sin proyetar las variables en el tsne, da bastante ogual proyectar o no las dos variables
def svm_bivar(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,new_data,n_targets,test_size):
    # particle_name=['alphas','muons']
    x1,x2,y=pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    
    x = np.column_stack((x1, x2))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Entrenar clasificador SVM con kernel RBF
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(x_train, y_train)
    
    # Crear malla de puntos
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicción en cada punto de la malla
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    y_pred = svm.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    if len(np.unique(y)) == 2:
        cmap = ListedColormap(['blue', 'green'])
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Clasificación de los datos con SVM (kernel RBF)')
    plt.colorbar(ticks=list(np.arange(0, len(np.unique(y)), 1)))
    plt.grid(True)
    plt.show()
    
    x = np.column_stack((x1,x2))
    new_data_scaled = scaler.transform(x)
    y_pred = svm.predict(new_data_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()

#%%
#%%
### haciendo lo del tsne y eslacado
def svm_multivar(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,new_data,n_targets,test_size):
    
    # particle_name=['alphas', 'muons']
    if len(particle_name)==3:
        x1,x2,x3,y=pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
        x = np.column_stack((x1, x2,x3))
    
    if len(particle_name)==2:
        x1,x2,y=pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
        x = np.column_stack((x1, x2))
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(data_projected, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Entrenar clasificador SVM con kernel RBF
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(x_train, y_train)
    
    # Crear malla de puntos
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicción en cada punto de la malla
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    y_pred = svm.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    if len(np.unique(y)) == 2:
        cmap = ListedColormap(['blue', 'green'])
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Clasificación de los datos con SVM (kernel RBF)')
    plt.colorbar(ticks=list(np.arange(0, len(np.unique(y)), 1)))
    plt.grid(True)
    plt.show()
    
    new_data_projected = tsne.fit_transform(x)
    new_data_scaled = scaler.transform(new_data_projected)
    y_pred = svm.predict(new_data_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
    #añadir los nuevos datos que se quieren predecir a los anteriores y hacer el tsne:
    # if new_data==int:
    #     new_data=41
    #     #add this new point to the preexisting data
    #     new_data=np.append(data,new_data)
    #     new_data_projected=tsne.fit_transform(new_data.reshape(-1,1))
        
    #     new_data_projected=tsne.fit_transform(new_data.reshape(-1,1))
    #     new_data_scaled=scaler.transform(new_data_projected)
    #     new_data_predicted=svm.predict(new_data_scaled)
    #     y_pred = np.argmax(svm.predict(new_data_predicted), axis=-1)
    #     accuracy = accuracy_score(y, y_pred)
    #     print(accuracy)
    # return accuracy,confusion_mat
def xgbosting_uniclass_multivariable_cv(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, n_targets):
    # particle_name=['alphas', 'muons']
    if len(particle_name)==3:
        x1,x2,x3,y=pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
        x = np.column_stack((x1, x2,x3))
    
    if len(particle_name)==2:
        x1,x2,y=pre_analysis_data_2D(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
        x = np.column_stack((x1, x2))
    
    # Aplicar t-SNE
    data=x
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)

    x_projected = data_projected
    labels = y
    
    # Convertir los datos en una matriz DMatrix de XGBoost
    dtrain = xgb.DMatrix(x_projected, label=labels)
    
    # Definir los parámetros del modelo XGBoost
    params = {
        'objective': 'multi:softmax',  # Clasificación multiclase
        'num_class': n_targets,  # Número de clases
        'eval_metric': 'merror',  # Métrica de evaluación del error de clasificación
        'gamma': 0.1,  # Parámetro de regularización gamma
        'alpha': 0.1,  # Parámetro de regularización L1 (Lasso)
        'lambda': 1.0,  # Parámetro de regularización L2 (Ridge)
        'max_depth': 3,  # Profundidad máxima del árbol
        'min_child_weight': 1,  # Número mínimo de muestras en un nodo hoja
        'learning_rate': 0.1  # Tasa de aprendizaje
    }
    
    # Definir los pliegues para la validación cruzada estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Realizar la validación cruzada
    for train_index, test_index in skf.split(x_projected, labels):
        x_train, x_test = x_projected[train_index], x_projected[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Convertir los datos de entrenamiento y prueba en matrices DMatrix de XGBoost
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        
        # Entrenar el modelo XGBoost
        model = xgb.train(params, dtrain)
        
        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(dtest)
        
        # Calcular métricas de evaluación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calcular promedio de métricas de evaluación
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores, axis=0)
    avg_recall = np.mean(recall_scores, axis=0)
    avg_f1 = np.mean(f1_scores, axis=0)
    
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1-score:", avg_f1)
    
    # Resto del código para la visualización y predicción total de los datos
    # ...

    # Crear malla de puntos
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicciones en cada punto de la malla
    dgrid = xgb.DMatrix(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(dgrid)
    Z = Z.reshape(xx.shape)
    
    # Definir colores para cada categoría
    if n_targets == 2:
        cmap = ListedColormap(['blue', 'green'])
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    
    # Graficar los puntos proyectados, coloreándolos según la categoría original,
    # y trazar las regiones de decisión del clasificador XGBoost
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clasificación de los datos con XGBoost')
    plt.colorbar(ticks=list(np.arange(0, n_targets, 1)))
    plt.grid(True)
    plt.show()
    
    # Predicción sobre el total de los datos
    new_data_projected = tsne.fit_transform(data)
    new_data_dmatrix = xgb.DMatrix(new_data_projected)
    y_pred_total = model.predict(new_data_dmatrix)
    accuracy = accuracy_score(y, y_pred_total)
    print("Accuracy (Total):", accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred_total)
    print("Confusion Matrix:")
    print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(labels)))
    plt.xticks(tick_marks, np.unique(labels))
    plt.yticks(tick_marks, np.unique(labels))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    # Agregar etiquetas a cada celda
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
    
    
    # Predicción sobre el total de los datos
    if isinstance(lim_newdata, int)==True:
        new_data=np.append(data,x_new_data)
        new_data_projected = tsne.fit_transform(new_data.reshape(-1,1))
        new_data_dmatrix = xgb.DMatrix(new_data_projected)
        y_pred_total = model.predict(new_data_dmatrix)
        accuracy_total = accuracy_score(y_new_data, y_pred_total[lim_newdata:])
        print("Accuracy (Total):", accuracy_total)
        
    return accuracy,confusion_mat
#%%

import pandas as pd

def multilabel(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,new_data,n_targets,test_size):
    
# Supongamos que tienes un DataFrame llamado 'df' con las columnas 'col1', 'col2', 'col3', 'col4'

# Crea una nueva columna 'nueva_col' que combine las cuatro columnas en una sola
    df_class=dataframe_name    
    df_class['class'] = df_class[['radiacion solar', 'lluvia', 'presion', 'temperatura']].astype(str).apply(lambda x: ''.join(x), axis=1)
    class_unique=df_class['class'].unique()
    nueva_clase=np.arange(0,len(class_unique),1)
    nueva_clase_aplicada=[]
    for i in range(len(df_class['class'])):
        for j in range(len(class_unique)):
            if df_class['class'][i]==class_unique[j]:
                nueva_clase_aplicada.append(nueva_clase[j])
    
    df_class['class_aplicada']=nueva_clase_aplicada
    plt.close('all')
    particle_name = ['e', 'alphas', 'muons']
    if len(particle_name) == 3:
        x1, x2, x3, y = pre_analysis_data_2D(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
        x = np.column_stack((x1, x2, x3))
    
    if len(particle_name) == 2:
        x1, x2, y = pre_analysis_data_2D(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
        x = np.column_stack((x1, x2))
    
    y = df_class['class_aplicada']
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(data_projected, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Entrenar clasificador SVM con kernel RBF
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(x_train, y_train)
    
    # Crear malla de puntos
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicción en cada punto de la malla
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    y_pred = svm.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    if len(np.unique(y)) == 2:
        cmap = ListedColormap(['blue', 'green'])
    elif len(np.unique(y)) == 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('tab20')
    
    cmap = ListedColormap(['blue', 'green', 'red', 'magenta'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Clasificación de los datos con SVM (kernel RBF)')
    plt.colorbar(ticks=list(np.arange(0, len(np.unique(y)), 1)))
    plt.grid(True)
    plt.show()
    
    new_data_projected = tsne.fit_transform(x)
    new_data_scaled = scaler.transform(new_data_projected)
    y_pred = svm.predict(new_data_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
    
    return df_class
def svm_trivar(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets, test_size):
    # particle_name=['alphas','muons']
    x1, x2, x3, y = pre_analysis_data_2D(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
    
    x = np.column_stack((x1, x2, x3))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Entrenar clasificador SVM con kernel RBF
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(x_train, y_train)
    
    # Crear malla de puntos
    x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    x3_min, x3_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
    
    # Crear malla de puntos
    x1_range = np.arange(x1_min, x1_max, 0.1)
    x2_range = np.arange(x2_min, x2_max, 0.1)
    x3_range = np.arange(x3_min, x3_max, 0.1)
    
    xx, yy, zz = np.meshgrid(x1_range, x2_range, x3_range)
    
    # Aplanar las matrices de la malla de puntos
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    zz_flat = zz.ravel()
    
    # Preparar los datos de entrada para la predicción
    points = np.column_stack((xx_flat, yy_flat, zz_flat))
    points_scaled = scaler.transform(points)
    
    # Realizar predicción en cada punto de la malla
    Z = svm.predict(points_scaled)
    
    # Reshape y reasignar Z para que coincida con las dimensiones de la malla
    Z = Z.reshape(xx.shape)
    
    y_pred = svm.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    if len(np.unique(y)) == 2:
        cmap = ListedColormap(['blue', 'green'])
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=y_train, cmap=cmap)
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.set_zlabel('Variable 3')
    
    # Superficie de decisión
    ax.plot_surface(xx, yy, zz, alpha=0.3, facecolors=cmap(Z))
    
    plt.title('Clasificación de los datos con SVM (kernel RBF)')
    plt.colorbar(ticks=list(np.arange(0, len(np.unique(y)), 1)))
    plt.grid(True)
    plt.show()
    
    x = np.column_stack((x1, x2, x3))
    new_data_scaled = scaler.transform(new_data)
    y_pred = svm.predict(new_data_scaled)
    accuracy = accuracy_score(n_targets, y_pred)
    print(accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(n_targets, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(n_targets)))
    plt.xticks(tick_marks, np.unique(n_targets))
    plt.yticks(tick_marks, np.unique(n_targets))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
