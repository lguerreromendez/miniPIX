# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:58:23 2023

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

lim_newdata='nada'   #-x
#DATA
df = pd.read_excel('dataframes/df.xlsx')
df_s = pd.read_excel('dataframes/df_s.xlsx')
df_t = pd.read_excel('dataframes/df_t.xlsx')
df_t1 = pd.read_excel('dataframes/df_t1.xlsx')
df_class = pd.read_excel('dataframes/df_class.xlsx')
def data_y_X(particle_name, dataframe_name,y_ydots_lim, y_dots_lim):
    weather_features=['radiacion solar', 'lluvia','presion', 'temperatura']
    X=df[weather_features]
    def y_ydots(particle,df):
        y=df[particle]
        ydots=df['dots']
        return y,ydots    
    index_list_delete=[]
    y,ydots=y_ydots(particle_name,dataframe_name)    
    #### eliminate some rows of t analysis data
    for i in range(len(y)):
        if y[i]/ydots[i]>y_ydots_lim and ydots[i]>y_dots_lim :
            index_list_delete.append(i)
    y=y.drop(index_list_delete)
    X=X.drop(index_list_delete)
    return X,y,index_list_delete, weather_features
def pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim):
    x=dataframe_name[particle_name] #  cantidad de muones   sin eliminar datos
    y=dataframe_name[variable] #categoria de la variable del tiempo
    print('puta')
    print(x)
    #eliminar los datos que estan mal
    x=data_y_X(particle_name, dataframe_name,y_ydots_lim, y_dots_lim)[1]
    y=y.loc[y.index.isin(x.index)]
    if isinstance(lim_newdata, int)==True:
        xx=np.array(x[:lim_newdata])
        yy=np.array(y[:lim_newdata])
        x_new_data=np.array(x[lim_newdata:])
        y_new_data=np.array(y[lim_newdata:])
    else:
        xx=np.array(x)
        yy=np.array(y)
        x_new_data=[]
        y_new_data=[]
    return xx,yy,x_new_data,y_new_data

#TRY PCA
def PCA_analysis(particle_name,dataframe_name, n_components,y_ydots_lim, y_dots_lim):

    X,y,index_list_delete, weather_features=data_y_X(particle_name,dataframe_name,y_ydots_lim, y_dots_lim)
    ###elimminate rows of the particle data measurement and the corresponding weather rows 
    ### in order to do the heatma, PCA...
    # Step 2: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Step 3: Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    # Step 4: Transform the data
    X_transformed = pca.transform(X_scaled)
    # Step 5: Analyze the results
    # Access the principal components and their corresponding explained variance ratios
    principal_components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    # Plot the cumulative explained variance ratio
    plt.figure(51)
    plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio')
    plt.show()
    # Analyze the weights or contributions of each weather feature to the principal components
    for i, pc in enumerate(principal_components):
        print(f"Principal Component {i+1}:")
        for feature, weight in zip(X.columns, pc):
            print(f"{feature}: {weight}")
        print()
    columns_principal_components=[]
    for i in range(n_components):
        columns_principal_components.append('PCA'+ str(i+1))
    principal_df=pd.DataFrame(principal_components,columns_principal_components,weather_features)
    principal_df['exp varience']=explained_variance_ratio
    principal_df=principal_df.T
    ###DATAFRAME WITH PCA DATA OF THE FEATURES
    print(i)
    return X,y, columns_principal_components
def PCA_dataframe(particle_name,dataframe_name, n_components,y_ydots_lim, y_dots_lim):       
    ###DATAFRAME WITH THE PCA DATA OF THE PARTICLES DATA
    X,y,columns_principal_components=PCA_analysis(particle_name, dataframe_name, n_components, y_ydots_lim, y_dots_lim)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca=PCA(n_components=n_components)
    principalComponents=pca.fit_transform(X)
    principalComponents_df=pd.DataFrame(principalComponents,columns=columns_principal_components)
    principalComponents_df['muons']=y
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    fig = plt.figure(15)
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    pca1=principalComponents_df['PCA1']
    pca2=principalComponents_df['PCA2']
    #pca3=principalComponents_df['PCA3']
    ax.scatter(pca1, pca2, y)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Number of Alpha Particles')
    ax.set_title('3D Scatter Plot of Alpha Particles and PCA Components')
    
    plt.show()
    ###REGRESSION
    # Assuming you have the two principal components (PCA1 and PCA2) in arrays 'pca1' and 'pca2',
    # and the number of alpha particles in an array 'alpha_particles'
    
    # Prepare the data
    X = np.column_stack((pca1, pca2))  # Combine PCA1 and PCA2 into a feature matrix
    y = y  # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a linear regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    # Evaluate the model
    # Evaluate the model
    y_pred = regression_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2):", r2)
    
    train_score = regression_model.score(X_train, y_train)
    test_score = regression_model.score(X_test, y_test)
    
    print("Train Score (R-squared):", train_score)
    print("Test Score (R-squared):", test_score)
    
    # Interpret the coefficients
    coefficients = regression_model.coef_
    intercept = regression_model.intercept_
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)
    
    # Make predictions on new data
    new_data = np.array([[2.5, 1.8], [1.0, -0.5]])  # Example new data with PCA1 and PCA2 values
    predicted_values = regression_model.predict(new_data)
    print("Predicted values for new data:", predicted_values)

def heatmap(particle_name,dataframe_name,y_ydots_lim, y_dots_lim): 
#HEATMAP
    dataframe_name=dataframe_name.iloc[:-6]
    X,y,index_list_delete, weather_features=data_y_X(particle_name,dataframe_name,y_ydots_lim, y_dots_lim)
    import seaborn as sns
    correlation_matrix=dataframe_name.drop(index_list_delete).corr()  ###le he eliminado las filas al df para hacer la matriz
    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # Set plot title
    plt.title('Correlation Heatmap')
    # Display the plot
    plt.show()
def box_plot(particle_name,dataframe_name,y_ydots_lim, y_dots_lim):
    import seaborn as sns
    X,y,index_list_delete, weather_features=data_y_X(particle_name,dataframe_name,y_ydots_lim, y_dots_lim)
    data_subset=dataframe_name.drop(index_list_delete).drop('dots',axis=1)
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data_subset)
    plt.title('Box Plot')
    plt.show()
def pair_plot(particle_name,dataframe_name,y_ydots_lim, y_dots_lim):
    import seaborn as sns
    X,y,index_list_delete, weather_features=data_y_X(particle_name,dataframe_name,y_ydots_lim, y_dots_lim)
    data_subset=dataframe_name.drop(index_list_delete).drop('dots',axis=1)
    sns.pairplot(data_subset)
    plt.title('Pair Plot')
    plt.show()
    
    
def regresion_lineal_univariable(particle_name, dataframe_name, variable,y_ydots_lim, y_dots_lim):

    plt.figure()
    #eliminar los datos
    x=data_y_X(particle_name, dataframe_name,y_ydots_lim, y_dots_lim)[0]
    print(len(x))
    y=data_y_X(particle_name, dataframe_name,y_ydots_lim, y_dots_lim)[1]
    print(len(y))
    x=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)[0]
    y=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)[1]
    print('hola', len(x))
    # x=x[variable]
    X=np.array(x).reshape((-1,1))
    model = LinearRegression()
    x_intervalo=np.linspace(min(x),max(x), 1000)
    # Ajustar el modelo a los datos
    model.fit(X, np.array(y))
    r2_score = model.score(X, y)
    print('==========================')
    print('Coeficiente de pendiente (w1):', model.coef_)
    print('Término de intersección (w0):', model.intercept_)
    print('Coeficiente de regresión (R²):', r2_score)
    
    plt.plot(x,y,'*')
    plt.plot(x_intervalo, x_intervalo*model.coef_+model.intercept_)
    plt.xlabel(variable)
    plt.ylabel(particle_name)

def class_1d_2d(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    plt.close('all')
    
    # Datos de cantidad de muones y categoría de temperatura
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    x=x.reshape(-1,1)
    plt.figure(1)
    plt.plot(x[:, 0][y==0], np.zeros(len(x[:, 0][y==0])), "bs", label='Categoria 0')
    plt.plot(x[:, 0][y==1], np.zeros(len(x[:, 0][y==1])), "g^", label='Categoria 1')
    plt.plot(x[:, 0][y==2], np.zeros(len(x[:, 0][y==2])), "r*", label='Categoria 2')
    plt.gca().get_yaxis().set_ticks([])
    plt.xlabel(particle_name, fontsize=20)
    plt.legend(loc='best')
    x2= np.c_[x, x**3]
    plt.figure(2)
    plt.plot(x2[:, 0][y==0], x2[:, 1][y==0], "bs", label='Categoria 0')
    plt.plot(x2[:, 0][y==1],  x2[:, 1][y==1], "g^", label='Categoria 1')
    plt.plot(x2[:, 0][y==2],  x2[:, 1][y==2], "r*", label='Categoria 2')
    plt.xlabel(particle_name, fontsize=20)
    plt.legend(loc='best')
def kmean_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    plt.close('all')
    particle_name='muons'
    variable='temperatura'
    dataframe_name=df_class
    y_ydots_lim, y_dots_lim=0.01,10000  #10000
    
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    # Datos
    data = x
    
    # Reshape de los datos para que sea un array 2D
    data_2D = data.reshape(-1, 1)
    
    # Aplicar Kernel PCA
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data_2D)
    
    # kpca = KernelPCA(n_components=2, kernel='rbf')
    # data_projected = kpca.fit_transform(data_2D)
    
    x=data_2D;x2=data_projected
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(x2)
    # Plot de la proyección
    plt.plot(x2[:, 0][y==0], x2[:, 1][y==0], "bs", label='Categoria 0')
    plt.plot(x2[:, 0][y==1],  x2[:, 1][y==1], "g^", label='Categoria 1')
    plt.plot(x2[:, 0][y==2],  x2[:, 1][y==2], "r*", label='Categoria 2')
    #plt.scatter(x2[:,0],x2[:,1],c=labels,cmap='viridis')
    plt.xlabel(particle_name, fontsize=20)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Proyección de Kernel PCA')
    plt.grid(True)
    plt.legend()
    plt.show()

def svm_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    plt.close('all')
    # Datos
    data = x
    
    # Reshape de los datos para que sea un array 2D
    data_2D = data.reshape(-1, 1)
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data_2D)
    x = data_projected
    labels = y
    svm = SVC(kernel='linear')
    svm.fit(x, labels)
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap = ListedColormap(['blue', 'green', 'red'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    # plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
    plt.plot(x[:, 0][y==0], x[:, 1][y==0], "bs", label='Categoria 0')
    plt.plot(x[:, 0][y==1],  x[:, 1][y==1], "g^", label='Categoria 1')
    plt.plot(x[:, 0][y==2],  x[:, 1][y==2], "r*", label='Categoria 2')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clasificación de los datos con SVM')
    plt.colorbar()
    plt.grid(True)
    plt.legend()
    plt.show()

def svm_uniclass_2(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim, new_data,n_targets, test_size):
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    print(len(x))
    print(len(y))
    print('===========')
    data=x.reshape(-1,1)
    # Reshape de los datos para que sea un array 2D
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    # tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
    data_projected = tsne.fit_transform(data)
    
    x_projected = data_projected
    labels = y
    
    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test=scaler.fit_transform(x_test)
    # Datos
    
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
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
  
    # Definir colores para cada categoría

    if n_targets==2:
        cmap = ListedColormap(['blue', 'green'])
        
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    # cmap = ListedColormap(['blue', 'green', 'red'])
    
    # Graficar los puntos proyectados, coloreándolos según la categoría original, y trazar las regiones de decisión del clasificador SVM
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clasificación de los datos con SVM (kernel RBF)')
    plt.colorbar(ticks=list(np.arange(0,n_targets,1)))
    plt.grid(True)
    plt.show()
    #prediccion sobre el total de los datos
    new_data_projected=tsne.fit_transform(data)
    new_data_scaled=scaler.transform(new_data_projected)
    # new_data_predicted=svm.predict(new_data_scaled)
    y_pred = svm.predict(new_data_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)
    
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred)  #y_test,y_pred
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
    # Add labels to each cell
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
    
    #añadir los nuevos datos que se quieren predecir a los anteriores y hacer el tsne:
    
    if isinstance(lim_newdata, int)==True:
        new_data=x_new_data
        #add this new point to the preexisting data
        new_data=np.append(data,new_data)
        new_data_projected=tsne.fit_transform(new_data.reshape(-1,1))
        new_data_projected=tsne.fit_transform(new_data.reshape(-1,1))
        new_data_scaled=scaler.transform(new_data_projected)
        new_data_predicted=svm.predict(new_data_scaled)
        accuracy_total = accuracy_score(y_new_data, new_data_predicted[lim_newdata:])
        print(accuracy_total)
    return accuracy,confusion_mat
def randomforest_uniclas(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    
    data = x.reshape(-1, 1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)
    
    x_projected = data_projected
    labels = y
    
    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    # Train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(x_train, y_train)
    
    # Create meshgrid for visualization
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on each point of the meshgrid
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    y_pred = rf.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Define colors for each category
    cmap = ListedColormap(['blue', 'green', 'red'])
    
    # Plot the projected points, color them based on the original category, and plot the decision regions of the Random Forest classifier
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    #plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmap)
    plt.xlabel('Component 1')
def neuralnetwork_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim, new_data,n_targets, test_size):
    plt.close('all')
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    data = x.reshape(-1, 1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)
    
    x_projected = data_projected
    labels = y
    
    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(n_targets, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1)
    
    # Make predictions on test data
    y_pred = np.argmax(model.predict(x_test), axis=-1)  ##argmax por la funcion de salida softmax
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Create meshgrid for visualization
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on each point of the meshgrid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision regions of the classifier
    if n_targets==2:
        cmap = ListedColormap(['blue', 'green'])
        
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot the projected points, color them based on the original category
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Data Classification with Neural Network (t-SNE)')
    plt.colorbar(ticks=list(np.arange(0,n_targets,1)))
    plt.grid(True)
    plt.show()
    #prediccion sobre el total de los datos
    new_data_projected=tsne.fit_transform(data)
    new_data_scaled=scaler.transform(new_data_projected)
    new_data_predicted=model.predict(new_data_scaled)
    y_pred = np.argmax(new_data_predicted, axis=-1)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)
    plt.figure()
    confusion_mat = confusion_matrix(y, y_pred)  #y_test,y_pred
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
    
    # Add labels to each cell
    thresh = confusion_mat.max() / 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")
    plt.show()
    #añadir los nuevos datos que se quieren predecir a los anteriores y hacer el tsne:

    
    if isinstance(lim_newdata, int)==True:
        # new_data=x_new_data
        #add this new point to the preexisting data
        new_data=np.append(data,x_new_data)
        new_data_projected=tsne.fit_transform(new_data.reshape(-1,1))
        new_data_scaled=scaler.transform(new_data_projected)
        new_data_predicted=model.predict(new_data_scaled)
        y_pred = np.argmax(new_data_predicted, axis=-1)
        accuracy_total = accuracy_score(y_new_data, y_pred[lim_newdata:])
        print(accuracy_total)
    return accuracy,confusion_mat

def xgbosting_uniclass(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim,n_targets):
    x, y = pre_analysis_data(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
    
    data = x.reshape(-1, 1)
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)
    
    x_projected = data_projected
    labels = y
    
    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)
    
    # Convertir los datos en una matriz DMatrix de XGBoost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    
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
    
    # Entrenar el modelo XGBoost
    model = xgb.train(params, dtrain)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(dtest)
    
    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
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
    accuracy_total = accuracy_score(y, y_pred_total)
    print("Accuracy (Total):", accuracy_total)
    
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
    
    return accuracy,confusion_mat
def xgbosting_uniclass_cv(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, n_targets):
    x, y, x_new_data, y_new_data = pre_analysis_data(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
    
    data = x.reshape(-1, 1)
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
def naive_bayes_uniclass(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, n_targets):
    x, y,x_new_data,y_new_data = pre_analysis_data(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)
    
    data = x.reshape(-1, 1)
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)
    
    x_projected = data_projected
    labels = y
    
    # Dividir los datos en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)
    
    # Definir el modelo de clasificación Naive Bayes
    model = GaussianNB()
    
    # Entrenar el modelo Naive Bayes
    model.fit(x_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(x_test)
    
    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Crear malla de puntos
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicciones en cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Definir colores para cada categoría
    # Definir colores para cada categoría
    if n_targets == 2:
        cmap = ListedColormap(['blue', 'green'])
    else:
        cmap = ListedColormap(['blue', 'green', 'red'])
    # Graficar los puntos proyectados, coloreándolos según la categoría original,
    # y trazar las regiones de decisión del clasificador Naive Bayes
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clasificación de los datos con Naive Bayes')
    plt.colorbar(ticks=list(np.arange(0, n_targets, 1)))
    plt.grid(True)
    plt.show()
    
    # Predicción sobre el total de los datos
    new_data_projected = tsne.fit_transform(data)
    y_pred_total = model.predict(new_data_projected)
    accuracy_total = accuracy_score(y, y_pred_total)
    print("Accuracy (Total):", accuracy_total)
    
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
"""
def dbscan_uniclass(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim, new_data, n_targets):

    x, y = pre_analysis_data(particle_name, dataframe_name, variable, y_ydots_lim, y_dots_lim)

    data = x.reshape(-1, 1)
    # Reshape de los datos para que sea un array 2D
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_projected = tsne.fit_transform(data)

    x_projected = data_projected
    labels = y

    x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)

    x_train_scaled=x_train

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(x_train_scaled)

    # Crear malla de puntos
    x_min, x_max = x_train_scaled[:, 0].min() - 1, x_train_scaled[:, 0].max() + 1
    y_min, y_max = x_train_scaled[:, 1].min() - 1, x_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Obtener las etiquetas de los grupos generados por DBSCAN
    Z = dbscan.fit_predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Calcular la precisión
    accuracy = accuracy_score(y_train, cluster_labels)
    print("Accuracy:", accuracy)

    # Definir colores para cada categoría
    cmap = ListedColormap(['blue', 'green', 'red'])

    # Graficar los puntos proyectados, coloreándolos según la categoría original,
    # y trazar las regiones de decisión de DBSCAN
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap=cmap)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clustering de los datos con DBSCAN')
    plt.colorbar(ticks=list(np.arange(0, np.max(y_train) + 1, 1)))
    plt.grid(True)
    plt.show()

    # Predicción sobre el total de los datos
    new_data_projected = tsne.fit_transform(data)
    cluster_labels_total = dbscan.fit_predict(new_data_projected)
    accuracy_total = accuracy_score(y, cluster_labels_total)
    print("Accuracy (Total):", accuracy_total)

    plt.figure()
    confusion_mat = confusion_matrix(y, cluster_labels_total)
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

"""
"""
#%%
plt.close('all')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

particle_name = 'muons'
variable = 'temperatura'
dataframe_name = df_class
y_ydots_lim, y_dots_lim = 0.01, 100000

x = dataframe_name[particle_name]
y = dataframe_name[variable]

# Remove faulty data
x = data_y_X(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)[1]
y = y.loc[y.index.isin(x.index)]

x = np.array(x[:])
y = np.array(y[:])
x = x.reshape(-1, 1)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_projected = tsne.fit_transform(x)

x_projected = data_projected
labels = y

x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)


# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions on test data
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create meshgrid for visualization
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on each point of the meshgrid
Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
Z = Z.reshape(xx.shape)

# Plot the decision regions of the classifier
cmap = ListedColormap(['blue', 'green', 'red'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot the projected points, color them based on the original category
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Data Classification with Neural Network (t-SNE)')
plt.colorbar(ticks=[0, 1, 2])
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

particle_name = 'muons'
variable = 'temperatura'
dataframe_name = df_class
y_ydots_lim, y_dots_lim = 0.01, 100000

x = dataframe_name[particle_name]
y = dataframe_name[variable]

# Remove faulty data
x = data_y_X(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)[1]
y = y.loc[y.index.isin(x.index)]

x = np.array(x[:])
y = np.array(y[:])

# Reshape the data for t-SNE
data = x.reshape(-1, 1)

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_projected = tsne.fit_transform(data_scaled)

x_projected = data_projected
labels = y

x_train, x_test, y_train, y_test = train_test_split(x_projected, labels, test_size=0.2, random_state=42)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1)

# Make predictions on test data
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create meshgrid for visualization
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on each point of the meshgrid
Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
Z = Z.reshape(xx.shape)

# Plot the decision regions of the classifier
cmap = ListedColormap(['blue', 'green', 'red'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)


# Plot the projected points, color them based on the original category
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmap)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Data Classification with Neural Network (t-SNE)')
plt.colorbar(ticks=[0, 1, 2])
plt.grid(True)
plt.show()

"""

def neuralnetwork_regression(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    x,y, x_new_data, y_new_data=pre_analysis_data(particle_name,dataframe_name, variable, y_ydots_lim,y_dots_lim)
    
    # Normalizar los datos
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x[:, np.newaxis])
    
    # Definir la arquitectura de la red neuronal
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_dim=1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenar el modelo
    history = model.fit(x_scaled, y, epochs=1000, verbose=0)
    
    # Generar puntos para la predicción
    x_pred = np.linspace(min(x), max(x), 100)
    x_pred_scaled = scaler.transform(x_pred[:, np.newaxis])
    
    # Realizar la predicción
    y_pred = model.predict(x_pred_scaled)
    
    # Obtener los resultados de entrenamiento
    loss = history.history['loss']
    
    # Graficar la pérdida durante el entrenamiento
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el entrenamiento')
    plt.grid(True)
    plt.show()
    
    # Graficar los datos y la regresión no lineal
    plt.figure()
    plt.scatter(x, y, label='Datos')
    plt.plot(x_pred, y_pred, color='red', label='Regresión no lineal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regresión no lineal con redes neuronales')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calcular el coeficiente de determinación (R2 score)
    y_pred_train = model.predict(x_scaled)
    r2 = r2_score(y, y_pred_train)
    print("Coeficiente de determinación (R2 score):", r2)
  
def neuralnetwork_regression_2(particle_name, dataframe_name, variable,y_ydots_lim,y_dots_lim):
    particle_name = 'muons'
    variable = 'presion'
    dataframe_name = df
    y_ydots_lim, y_dots_lim = 0.01, 10000
    
    x = dataframe_name[particle_name]
    y = dataframe_name[variable]
    
    # Remove faulty data
    x = data_y_X(particle_name, dataframe_name, y_ydots_lim, y_dots_lim)[1]
    y = y.loc[y.index.isin(x.index)]
    
    x=np.array(x)
    y=np.array(y)
    
    
    # Normalizar los datos
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x[:, np.newaxis])
    
    # Reshape de los datos para que sean compatibles con la red LSTM
    x_reshaped = x_scaled.reshape(-1, 1, 1)
    
    # Definir la arquitectura de la red LSTM
    model = keras.Sequential([
        keras.layers.LSTM(16, activation='relu', input_shape=(1, 1)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenar el modelo
    history = model.fit(x_reshaped, y, epochs=1000, verbose=0)
    
    # Generar puntos para la predicción
    x_pred = np.linspace(min(x), max(x), 100)
    x_pred_scaled = scaler.transform(x_pred[:, np.newaxis])
    x_pred_reshaped = x_pred_scaled.reshape(-1, 1, 1)
    
    # Realizar la predicción
    y_pred = model.predict(x_pred_reshaped)
    
    # Obtener los resultados de entrenamiento
    loss = history.history['loss']
    
    # Graficar la pérdida durante el entrenamiento
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el entrenamiento')
    plt.grid(True)
    plt.show()
    
    # Graficar los datos y la regresión no lineal
    plt.figure()
    plt.scatter(x, y, label='Datos')
    plt.plot(x_pred, y_pred, color='red', label='Regresión no lineal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regresión no lineal con redes neuronales LSTM')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    from sklearn.metrics import r2_score
    
    # Predict the y values using the trained model
    y_pred = model.predict(x_reshaped)
    
    # Calculate the R2 score
    r2 = r2_score(y, y_pred)
    
    print("R2 Score:", r2)
