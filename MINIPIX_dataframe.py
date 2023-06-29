# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:55:18 2023

@author: luisg
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def all_dataframe():
    plt.close('all')
    #tipo de muestra puesta para medir
    
    def data(hist_data):
        with open(hist_data, 'r') as archivo:
            # Leer todas las líneas del archivo
            lineas = archivo.readlines()
        
            # Crear una lista vacía para almacenar las listas de valores numéricos
            lista = []
        
            # Iterar sobre cada línea del archivo y convertirla en una lista de valores numéricos
            for linea in lineas:
                # Dividir la línea en una lista de cadenas
                valores_str = linea.split()
        
                # Convertir cada cadena en un valor numérico y agregarlo a una lista
                valores_num = []
                for valor in valores_str:
                    valores_num.append(float(valor))
        
                # Agregar la lista de valores numéricos a la lista de listas
                lista.append(valores_num)
        
            # Imprimir la lista de listas resultante
            #print(listas)
        
        #alphas_E=lista[0]   
        alphas=lista[1]
        #e_E=lista[2]
        e=lista[3]
        #muons_E=lista[4]
        
        muons=lista[5]
        # for i in range(len(muons)):
        #     if muons[i]>5:
        #             muons[i]=0

        #dots_E=lista[6]
        dots=lista[7]
        # print('alphas:', np.sum(alphas))
        # print('e:', np.sum(e))
        # print('muons', np.sum(muons))
        # print('dots', np.sum(dots))
        return alphas, e, muons, dots
        
    def time(hist_data):
        with open(hist_data, 'r') as archivo:
            # Leer todas las líneas del archivo
            lineas = archivo.readlines()
        
            # Crear una lista vacía para almacenar las listas de valores numéricos
            lista = []
        
            # Iterar sobre cada línea del archivo y convertirla en una lista de valores numéricos
            for linea in lineas:
                # Dividir la línea en una lista de cadenas
                valores_str = linea.split()
        
                # Convertir cada cadena en un valor numérico y agregarlo a una lista
                valores_num = []
                for valor in valores_str:
                    valores_num.append(float(valor))
        
                # Agregar la lista de valores numéricos a la lista de listas
                lista.append(valores_num)
        
            # Imprimir la lista de listas resultante
            #print(listas)
        
        timeline=lista[0]   
        alphas=lista[1]
        e=lista[3]
        #muons_E=lista[4]
        muons=lista[5]
        #dots_E=lista[6]
        dots=lista[7]
        # print('alphas:', np.sum(alphas))
        # print('e:', np.sum(e))
        # print('muons', np.sum(muons))
        # print('dots', np.sum(dots))
        return alphas, e, muons, dots, timeline
  
    data_energy_list=[]
    alphas_list=[]
    e_list=[]
    muons_list=[]
    dots_list=[]
    data_list=['04-27','04-28','04-30','05-02','05-03', '05-04', '05-07', '05-08', '05-10', '05-11',
               '05-13','05-14', '05-17','05-18','05-19','05-22', '05-23', '05-24', '05-25', '05-26',
               '05-27','05-28','05-28','05-29','05-30','05-31','06-02','06-03','06-05','06-06','06-07','06-08','06-09',
               '06-13','06-14','06-16','06-19','06-23','06-24']
    
    time_list=['18:00','18:30', '18:30','18:20','18:20', '21:00', '18:40', '19:30', '18:20', '18:50',
               '18:40','18:10','18:30','18:10','18:10','18:40','18:20','18:40','18:20','17:10',
               '18:30','16:40','18:30','18:50','17:40','18:20','20:30','11:00','21:20','18:20','18:30','18:20','18:30',
               '18:30', '18:30','18:40','18:30','18:30','19:00']
    n_dias=0
    variable=True
    def eliminar(x, variable):
        for i in range(len(x)):
            if x[i]>5:
                    x[i]=0
        if sum(x)>100:
            if variable==True:
                x_list=[]
                for i in x:
                    if i>1:
                        x_list.append(i)
                x_2=sum(x_list)-len(x_list)
                return x_2
                    
        return x
    for i in range(10,len(data_list)+10):
        data_energy="mia/1800_1_"+str(i)+"_energy.txt"  #DATA  FROM ENERGY TXT
        alphas, e, muons, dots=data(data_energy)
        alphas=eliminar(alphas,variable)
        e=eliminar(e, variable)
        muons=eliminar(muons,variable)
        
                
        alphas=np.sum(alphas)
        e=np.sum(e)
        muons=np.sum(muons)
        dots=np.sum(dots)
        
        alphas_list.append(alphas)
        e_list.append(e)
        muons_list.append(muons)
        dots_list.append(dots)
        data_energy_list.append(i)
    
    #CREATE DATAFRAME OF PARTICLES OF ENERGY
    df=pd.DataFrame()
    df['Medida']=data_energy_list
    df['alphas']=alphas_list
    df['e']=e_list
    df['muons']=muons_list
    df['dots']=dots_list
    
    data_size_list=[]
    alphas_list=[]
    e_list=[]
    muons_list=[]
    dots_list=[]
    for i in range(10,len(data_list)+10):
        data_size="mia/1800_1_"+str(i)+"_size.txt"  #DATA  FROM SIZE TXT
        alphas, e, muons, dots=data(data_size)
        alphas=np.sum(alphas)
        e=np.sum(e)
        muons=np.sum(muons)
        dots=np.sum(dots)
        
        alphas_list.append(alphas)
        e_list.append(e)
        muons_list.append(muons)
        dots_list.append(dots)
        data_size_list.append(i)
    
    df_s=pd.DataFrame()
    df_s['Medida']=data_size_list
    df_s['alphas']=alphas_list
    df_s['e']=e_list
    df_s['muons']=muons_list
    df_s['dots']=dots_list
   
    #weather features
    
    ###meteogalicia1=2023-04-27 00:10:00.0  |   2023-05-27 00:00:00.0
    ###meteogalicia2=2023-05-27 00:10:00.0  |   2023-06-10 00:00:00.0
    ###meteogalicia3=2023-05-27 00:10:00.0  |   2023-06-25
    meteogalicia1=pd.read_csv('meteogalicia/resultadoCSV_Columnas_1.csv')
    #meteogalicia2=pd.read_csv('meteogalicia/resultadoCSV_Columnas_2.csv')
    meteogalicia3=pd.read_csv('meteogalicia/resultadoCSV_Columnas_3.csv')
    meteogalicia = pd.concat([meteogalicia1, meteogalicia3], ignore_index=True)   #combined
    #meteogalicia=pd.read_csv('meteogalicia/resultadoCSV_Columnas_1.csv')
    variables=meteogalicia.columns.tolist()
    instante_lectura=meteogalicia[variables[0]]
    
    
    indice_list=[]
    
    for i in range(len(data_list)):
        indice=-1
        for data in instante_lectura:
            indice+=1
            if data[5:10]==data_list[i] and data[11:16]==time_list[i]:
                indice_list.append(indice+n_dias*144)  ##+144 añadir un dia mas
        
    
    radiacion_list=[]   ; var_rad=5
    lluvia_list=[]      ; var_llu=1
    presion_list=[]     ; var_pres=3
    temperatura_list=[]  ; var_T=6
    for i in indice_list:
        radiacion=(meteogalicia.loc[i][var_rad]+meteogalicia.loc[i+1][var_rad]+meteogalicia.loc[i+2][var_rad]+meteogalicia.loc[i+3][var_rad])/4
        radiacion_list.append(radiacion)
        
        lluvia=(meteogalicia.loc[i][var_llu]+meteogalicia.loc[i+1][var_llu]+meteogalicia.loc[i+2][var_llu]+meteogalicia.loc[i+3][var_llu])/4
        lluvia_list.append(lluvia)
        
        presion=(meteogalicia.loc[i][var_pres]+meteogalicia.loc[i+1][var_pres]+meteogalicia.loc[i+2][var_pres]+meteogalicia.loc[i+3][var_pres])/4
        presion_list.append(presion)
        
        temperatura=(meteogalicia.loc[i][var_T]+meteogalicia.loc[i+1][var_T]+meteogalicia.loc[i+2][var_T]+meteogalicia.loc[i+3][var_T])/4
        temperatura_list.append(temperatura)
    
    #ADD WEATHER FEATURES TO THE DATAFRAME
    df['radiacion solar']=radiacion_list
    df['lluvia']=lluvia_list
    df['presion']=presion_list
    df['temperatura']=temperatura_list
    
    
    df_s['radiacion solar']=radiacion_list
    df_s['lluvia']=lluvia_list
    df_s['presion']=presion_list
    df_s['temperatura']=temperatura_list
    
    #df: energy dataframe
    #df_t timeline txt datafrme with timelapse applied
    #df_t1 original timeline txt dataframe
    
    #CREATE DATAFRAME OF PARTICLES
    df_t=df.copy() #data of timeline changing the timelapse getting only 15 minutes
    # the df_t dataframe must have lower values for some paticles compared with df_t1
    df_t1=df.copy() #without the timelapse applied
    
    
    
    #CHANGE DATA BECAUSE ERROR GETTING VALUES BY MINIPIX
    ### TIMELINE
    
    
    timeline_errors={}
    timeline_errors_solutions={}
    def time_analysis(hist_time, numero, grafica):
        
        #####ver analisis
        # print('=================================================================================')
        # print('=================================================================================')
        # print('=================================================================================')
        # print('timeline', numero)
        alphas_t, e_t, muons_t, dots_t, timeline=time(hist_time)
        #######    PRINT SOME INFORMATION ABOUT THE DATA 
        ###### ver analisis
        # print('alphas_t', sum(alphas_t))
        # print('e_t', sum(e_t))
        # print('muons_t',sum(muons_t))
        # print('dots_t', sum(dots_t))
        # print('=====================')
    
        #original timeline data in df_t1 dataframe
        df_t1['alphas'][numero-10]=sum(alphas_t)  #coger los datos sin tratar del timeline
        df_t1['muons'][numero-10]=sum(muons_t)
        df_t1['e'][numero-10]=sum(e_t)
        df_t1['dots'][numero-10]=sum(dots_t)
        
        # alphas=df['alphas'][numero-10]
        # e=df['e'][numero-10]
        # muons=df['muons'][numero-10]
        # dots=df['dots'][numero-10]
        # print('diferencia de alphas medidas', alphas-sum(alphas_t))
        # print('diferencia de muons medidas', muons-sum(muons_t))
        # print('diferencia de e medidas', e-sum(e_t))
        # print('diferencia de dots medidas', dots-sum(dots_t))
        
        #grafica en la que se ve el timelne.txt, lo mismo que en el software
        if grafica==1:
            plt.figure(numero)
            plt.title('timeline_'+str(numero)+'.txt')
            plt.xlabel('timeline')
            plt.ylabel('nº particles')
            plt.plot(timeline,alphas_t, color='blue', label='alphas_t')
            plt.plot(timeline, e_t, color='green', label='e_t')
            plt.plot(timeline, muons_t, color='red', label='muons_t')
            plt.plot(timeline,dots_t, color='violet', label='dots_t')
            plt.legend(loc='best')
            #plt.savefig('analisis/timeline_'+str(numero)+'.png')
        #cantidad de muones unicos diferentes que mide,  que medidas diferentes hace
        muons_unique=set(muons_t)  
        alphas_unique=set(alphas_t)
        e_unique=set(e_t)
        dots_unique=set(dots_t)
        
        # diccionario  con los valores unicos existente y la cantidad de particulas que
        # mide de cada uno de estos valores
        muons_t_dict={} ;alphas_t_dict={}; e_t_dict={}; dots_t_dict={}
        def unique_valor(x_unique,x_t,x_t_dict):  #create the dictionary
            for valor in x_unique:
                cantidad_x_unique=x_t.count(valor)
                x_t_dict[valor]=cantidad_x_unique       
        unique_valor(muons_unique,muons_t,muons_t_dict) 
        unique_valor(alphas_unique,alphas_t,alphas_t_dict) 
        unique_valor(e_unique,e_t,e_t_dict) 
        unique_valor(dots_unique,dots_t,dots_t_dict) 
                
    
        #### cantida de valores unicos
        # print('cantidad de valores unicos de muons', len(muons_unique),'valores unicos de muons', muons_t_dict)
        # print('cantidad de valores unicos de alphas', len(alphas_unique),'valores unicos de alphas', alphas_t_dict)
        # print('cantidad de valores unicos de e', len(e_unique),'valores unicos de e', e_t_dict)
        # print('cantidad de valores unicos', len(dots_unique),'valores unicos de dots', dots_t_dict)
        
        
        
        ####saber que particulas tienen error en los datos
        intervalo_mal=[]
        intervalo_bien=[]
        def intervalo_mal_bien(x,y,umbral,nombre, malbien=True):
            if len(x)<(y+2) and list(x)[-1]<(umbral+3):
                # print('intervalo de',nombre, 'no sobrepasa el umbral de', y,
                #        '--> tiene', len(x),'valores diferentes medidos' )
                # print(list(x)[-1])
                intervalo='bien'
                if malbien==True:
                    intervalo_bien.append(nombre)
                
                return intervalo
            if len(x)>=(y+2):
                # print('intervalo de',nombre, 'sobrepasa el umbral de', y,
                #       '--> tiene', len(x),'valores diferentes medidos' )
                intervalo='mal'
                if malbien==True:
                    intervalo_mal.append(nombre)
                return intervalo
            
            # else:
            #     intervalo_mal.append(nombre)
        
        
        
        #ver si el intervalo esta bien
        # print('=================')
        # print('ver que intervalo esta bien y cual mal')
        intervalo_mal_bien(muons_unique,4,3,'muons')
        intervalo_mal_bien(alphas_unique,2,2,'alphas')
        intervalo_mal_bien(e_unique,4,3,'e')
        intervalo_mal_bien(dots_unique,6,9,'dots')
        #ahora ya se si hay algunas particulas que las mida mal, ahora que
        #timelapse 900 frames 900 segundos
        
        # print('intervalo bbien', intervalo_bien)
        # print('intervalo mal',intervalo_mal)
        
        def timelapse_funtion(timeline,x_t,y,umbral, nombre, valor_intervalo):
            for i in range(valor_intervalo):
                timelapse=timeline[i:i+valor_intervalo]
                x_t_timelapse=x_t[i:i+valor_intervalo]
                
                x_unique_timelapse=set(x_t_timelapse) 
                x_t_dict_timelapse={}
                unique_valor(x_unique_timelapse,x_t_timelapse,x_t_dict_timelapse) 
                intervalo=intervalo_mal_bien(x_unique_timelapse,y,umbral,nombre, malbien=False)
                ##malbien=False para no añadir mas cosas a la lista de intervalo_mal / intervalo_bien
                if intervalo=='bien':  #que si encuentra un intervalo que pare
                    break
            return timelapse, intervalo
                
        #los que esten mal arreglarlos
        ###     intervalo_mal
        # print('=================')
        # print('los que estan mal arreglarlos')
        
        
        ### implementacion del nuevo timelapse
        ### si esta bien ponerlos en la base de datos, si no mantener los del df_t1
        def timelapse_implementation(timeline,x_t,x,y,nombre, valor_intervalo):
            timelapse_x=timelapse_funtion(timeline, x_t,x,y,nombre,valor_intervalo)[0]
            index_0_x=timeline.index(timelapse_x[0]); index_1_x=timeline.index(timelapse_x[-1]);
            x_t_timelapse=x_t[index_0_x:index_1_x+1] #coger todo el rango de valores del intervalo
            x_unique_timelapse=set(x_t_timelapse)  #ver los valores unicos
            x_t_timelapse_dict={}
            unique_valor(x_unique_timelapse,x_t_timelapse,x_t_timelapse_dict)  #diccionatio con valores unicos nuevos
            #print('dict', nombre, ':', x_t_timelapse_dict)
            df_t[nombre][numero-10]=sum(x_t_timelapse)*(round(len(timeline)/valor_intervalo))
            return timelapse_x,x_t_timelapse
        
        solution=[]
        pd.options.mode.chained_assignment = None
        for i in intervalo_mal:
            if i=='alphas':
                intervalo=timelapse_funtion(timeline, alphas_t,2,2,'alphas', 856)[1]
                if intervalo=='bien':
                    timelapse_implementation(timeline, alphas_t, 2, 2, 'alphas',856) 
                    #comprobar que el modelo esta funcionando bieny ha cambiado alguno
                    # print('OJITOOOOOOOOO')
                    # print('----------timelapse 856 implementado con exito en alphas')
                    solution.append('alpha 856')
                else: 
                    df_t['alphas'][numero-10]=sum(alphas_t)
                    
            if i=='muons':
                intervalo=timelapse_funtion(timeline, muons_t,4,3,'muons',856)[1]
                if intervalo=='bien':
                    timelapse_implementation(timeline, muons_t, 4, 3, 'muons',856)
                    # print('OJITOOOOOOOOO')
                    # print('----------timelapse 856 implementado con exito en muons')
                    solution.append('muons 856')
                    
                else:
                    intervalo=timelapse_funtion(timeline, muons_t,6,9,'muons',570)[1]
                    if intervalo=='bien':
                        timelapse_implementation(timeline, muons_t, 6, 9, 'muons', 570)
                        # print('----------timelapse 570 implementado con exito en muons')
                        solution.append('muons 570')
                    else:
                        df_t['muons'][numero-10]=sum(muons_t)
                     
            if i=='e':
                intervalo=timelapse_funtion(timeline, e_t,4,3,'e',856)[1]
                if intervalo=='bien':
                    timelapse_implementation(timeline, e_t, 4, 3, 'e',856)
                    # print('----------timelapse 856 implementado con exito en e')
                    solution.append('e 856')
                    
                    # #ver cual es el nuevo
                    # plt.figure()
                    # plt.plot(timelapse_implementation(timeline, e_t, 4, 3, 'e',856)[0],
                    #          timelapse_implementation(timeline, e_t, 4, 3, 'e',856)[1])
                else:
                    intervalo=timelapse_funtion(timeline, e_t,6,9,'e',570)[1]
                    if intervalo=='bien':
                        timelapse_implementation(timeline, e_t, 6, 9, 'e', 570)
                        # print('----------timelapse 570 implementado con exito en e')
                        solution.append('e 570')
                    else:
                        intervalo=timelapse_funtion(timeline, e_t,6,9,'e',427)[1]
                        if intervalo=='bien':
                            timelapse_implementation(timeline, e_t, 6, 9, 'e', 427)
                            # print('----------timelapse 427 implementado con exito en e')
                            solution.append('e 470')
                        else:
                            df_t['e'][numero-10]=sum(e_t)
            if i=='dots':
                intervalo=timelapse_funtion(timeline, dots_t,6,9,'dots',856)[1]
                if intervalo=='bien':
                    timelapse_implementation(timeline, dots_t, 6, 9, 'dots', 856)
                    # print('----------timelapse 856 implementado con exito en dots')
                    solution.append('dots 856')
                else:
                    intervalo=timelapse_funtion(timeline, dots_t,6,9,'dots',570)[1]
                    if intervalo=='bien':
                        timelapse_implementation(timeline, dots_t, 6, 9, 'dots', 570)
                        # print('----------timelapse 570 implementado con exito en dots')
                        solution.append('dots 570')
                    else:
                        intervalo=timelapse_funtion(timeline, dots_t,6,9,'dots',427)[1]
                        if intervalo=='bien':
                            timelapse_implementation(timeline, dots_t, 6, 9, 'dots', 427)
                            # print('----------timelapse 427 implementado con exito en dots')
                            solution.append('dots 470')
                        else:
                            df_t['dots'][numero-10]=sum(dots_t)
               
        for i in intervalo_bien:
            if i=='alphas':
                df_t['alphas'][numero-10]=sum(alphas_t)
                
            elif i=='muons':
                df_t['muons'][numero-10]=sum(muons_t)
                
            elif i=='e':
                df_t['e'][numero-10]=sum(e_t)
                
            elif i=='dots':
                df_t['dots'][numero-10]=sum(dots_t)
         
        timeline_errors[numero]=intervalo_mal
        timeline_errors_solutions[numero]=solution
        
        #ahora ya se si hay algunas particulas que las mida mal, ahora que
        #timelapse 900 frames 900 segundos
       
        #try to see if they follow a linear ascendent growth
        e_t_sum=[]; e_t0=0
        alphas_t_sum=[]; alphas_t0=0
        muons_t_sum=[]; muons_t0=0
        dots_t_sum=[]; dots_t0=0
        for i,j,k,l in zip(e_t,alphas_t,muons_t, dots_t):
            e_t0+=i ;e_t_sum.append(e_t0)
            alphas_t0+=j ;alphas_t_sum.append(alphas_t0)
            muons_t0+=k ;muons_t_sum.append(muons_t0)
            dots_t0+=l ;dots_t_sum.append(dots_t0)
            
        
        #grafica en la que se ve el crecimiento del numero de particulas detectadas
        if grafica==2:
            plt.figure(numero)
            plt.title('timeline growth.txt')
            plt.xlabel('timeline')
            plt.ylabel('nº particles')
            plt.plot(timeline,alphas_t_sum,'o', color='blue', label='alphas_t')
            plt.plot(timeline, e_t_sum,'*', color='green', label='e_t')
            plt.plot(timeline, muons_t_sum,'>', color='red', label='muons_t')
            plt.plot(timeline,dots_t_sum, '^', color='violet', label='dots_t')
            plt.legend(loc='best')
            plt.show()
    
    
    #PONER TODAS JUNTAS
    def time_analysis_subplot(hist_time, numero, grafica,ax):
        alphas_t, e_t, muons_t, dots_t, timeline=time(hist_time)
        print('alphas_t', sum(alphas_t))
        print('e_t', sum(e_t))
        print('muons_t',sum(muons_t))
        print('dots_t', sum(dots_t))
        print('=====================')
        
        alphas=df['alphas'][numero-10]
        e=df['e'][numero-10]
        muons=df['muons'][numero-10]
        dots=df['dots'][numero-10]
        print('diferencia de alphas medidas', alphas-sum(alphas_t))
        print('diferencia de muons medidas', muons-sum(muons_t))
        print('diferencia de e medidas', e-sum(e_t))
        print('diferencia de dots medidas', dots-sum(dots_t))
        
        
        #grafica en la que se ve el timelne.txt, lo mismo que en el software
        if grafica==1:
            ax.set_title('timeline.txt')
            ax.set_xlabel('timeline')
            ax.set_ylabel('nº particles')
            ax.plot(timeline,alphas_t,'o', color='blue', label='alphas_t')
            ax.plot(timeline, e_t,'*', color='green', label='e_t')
            ax.plot(timeline, muons_t,'>', color='red', label='muons_t')
            ax.plot(timeline,dots_t, '^', color='violet', label='dots_t')
            ax.legend(loc='best')
        
        #try to see if they follow a linear ascendent growth
        e_t_sum=[]; e_t0=0
        alphas_t_sum=[]; alphas_t0=0
        muons_t_sum=[]; muons_t0=0
        dots_t_sum=[]; dots_t0=0
        for i,j,k,l in zip(e_t,alphas_t,muons_t, dots_t):
            e_t0+=i ;e_t_sum.append(e_t0)
            alphas_t0+=j ;alphas_t_sum.append(alphas_t0)
            muons_t0+=k ;muons_t_sum.append(muons_t0)
            dots_t0+=l ;dots_t_sum.append(dots_t0)
         
        #grafica en la que se ve el crecimiento del numero de particulas detectadas
        if grafica==2:
            plt.title('timeline growth.txt')
            plt.xlabel('timeline')
            plt.ylabel('nº particles')
            plt.plot(timeline,alphas_t_sum,'o', color='blue', label='alphas_t')
            plt.plot(timeline, e_t_sum,'*', color='green', label='e_t')
            plt.plot(timeline, muons_t_sum,'>', color='red', label='muons_t')
            plt.plot(timeline,dots_t_sum, '^', color='violet', label='dots_t')
            plt.legend(loc='best')
            plt.show()
    
    
    
    
    plt.close('all')
    ### arange with the plots 
    valores=np.arange(10,30,1)
    for numero in valores:
        hist_time='mia/1800_1_'+str(numero)+'_timeline.txt'
        time_analysis(hist_time, numero, grafica=0)
        
    return df, df_s, df_t, df_t1

###cambiar los datos de los 4 weather features para que sea una clase.
### de esta forma podremos hacer una clasificacion

def weather_class(df):
    import numpy as np
    import pandas as pd
    
    weather_features=df.columns[5:]
    weather_features_values=df[weather_features]

    valores_weather=[] #valores de cada feature sin clasificar
    valores_weather_intervalo=[] #valores de los intervlos de cada feature usado para la clasificacion
    valores_clasificados_weather=[] #lista con los valores clasificados para cada weather feature
    # for i in range(len(weather_features)):
    for i in range(4):
        print('rango de',weather_features[i],'[', min(weather_features_values[weather_features[i]]),'--',max(weather_features_values[weather_features[i]]),']')

        #radiacin solar, lluvia, presion, temperatura
        valores=weather_features_values[weather_features[i]]
        valores_weather.append(valores)
        
        valores_clasificados=[]
        if i==0:
            # etiquetas = [0, 1, 2]
            # intervalos = [min(valores)-1,np.mean(valores)/2,np.mean(valores)*(3/2),max(valores)+1]
            etiquetas = [0, 1]
            intervalos = [min(valores)-1,np.mean(valores)/2,max(valores)+1]
            clasificacion = pd.cut(valores, bins=intervalos, labels=etiquetas, right=False)
            # Iterar sobre los valores clasificados y etiquetarlos
            for valor, etiqueta in zip(valores, clasificacion):
                #print(f"Valor: {valor}, Etiqueta: {etiqueta}")
                valores_clasificados.append(etiqueta)
            valores_clasificados_weather.append(valores_clasificados)
        if i==1:
            etiquetas=[0,1]
            intervalos=[0,0.0000000000001,1000]
            clasificacion = pd.cut(valores, bins=intervalos, labels=etiquetas, right=False)
            
            # Iterar sobre los valores clasificados y etiquetarlos
            for valor, etiqueta in zip(valores, clasificacion):
                #print(f"Valor: {valor}, Etiqueta: {etiqueta}")
                valores_clasificados.append(etiqueta)
            valores_clasificados_weather.append(valores_clasificados)
            
        if i==2:
            etiquetas = [0, 1]
            intervalos=[min(valores), np.mean(valores),max(valores)+1]
            clasificacion = pd.cut(valores, bins=intervalos, labels=etiquetas, right=False)

            # Iterar sobre los valores clasificados y etiquetarlos
            for valor, etiqueta in zip(valores, clasificacion):
                #print(f"Valor: {valor}, Etiqueta: {etiqueta}")
                valores_clasificados.append(etiqueta)
            valores_clasificados_weather.append(valores_clasificados)

        if i==3:
            etiquetas = [0, 1, 2]
            intervalos = np.linspace(min(valores)-0.1, max(valores)+0.1, 4)
            
            etiquetas = [0, 1]
            # intervalos = np.linspace(13.4-0.1,26.6+0.1, 3)
            intervalos=[13.3,17,30]
            # intervalos = np.linspace(min(valores)-0.1, max(valores)+0.1, 3)
            
            clasificacion = pd.cut(valores, bins=intervalos, labels=etiquetas, right=False)
            
            # Iterar sobre los valores clasificados y etiquetarlos
            for valor, etiqueta in zip(valores, clasificacion):
                #print(f"Valor: {valor}, Etiqueta: {etiqueta}")
                valores_clasificados.append(etiqueta)
            valores_clasificados_weather.append(valores_clasificados)
        
        valores_weather_intervalo.append(intervalos)
        

    df_class=df.copy()
    df_class=df_class.drop(weather_features, axis=1)
    for i in range(len(weather_features)):
        df_class[weather_features[i]]=valores_clasificados_weather[i]
        
    return df_class
def dataframe_classification():
    df, df_s, df_t, df_t1=all_dataframe()
    return df, df_s, df_t, df_t1

df, df_s, df_t, df_t1=dataframe_classification()

df_class=weather_class(df)

# Especifica el nombre de la carpeta que deseas crear
carpeta = 'dataframes'

# Verifica si la carpeta ya existe
if not os.path.exists(carpeta):
    # Crea la carpeta
    os.makedirs(carpeta)
    print(f'Se ha creado la carpeta "{carpeta}".')
else:
    print(f'La carpeta "{carpeta}" ya existe.')

df.to_excel('dataframes/df.xlsx', index=False)
df_s.to_excel('dataframes/df_s.xlsx', index=False)
df_t.to_excel('dataframes/df_t.xlsx', index=False)
df_t1.to_excel('dataframes/df_t1.xlsx', index=False)
df_class.to_excel('dataframes/df_class.xlsx', index=False)

df = pd.read_excel('dataframes/df.xlsx')
df_s = pd.read_excel('dataframes/df_s.xlsx')
df_t = pd.read_excel('dataframes/df_t.xlsx')
df_t1 = pd.read_excel('dataframes/df_t1.xlsx')
df_class = pd.read_excel('dataframes/df_class.xlsx')
