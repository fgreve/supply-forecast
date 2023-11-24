import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import os
from datetime import datetime
from dateutil import relativedelta
from tqdm import tqdm
from pathlib import Path
import json
from multiprocessing import Pool
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import *

import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator



def plot_ts(part_number, date, dda, fc_bp, fc_aa, first_date='all', meses_historia="all", enteros=False, figsize=(20,5), fc_months=6):  

    """
    Plot del forecast fc_aa de BigQuery
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(part_number) 

    if meses_historia=="all" and first_date=='all':
        df = pd.DataFrame(dda.loc[part_number][:date_add_months(date, fc_months+1)])
    if meses_historia!="all" and first_date=='all':
        df = pd.DataFrame(dda.loc[part_number][date_add_months(date, -meses_historia):date_add_months(date, fc_months+1)])
    if meses_historia=="all" and first_date!='all':
        df = pd.DataFrame(dda.loc[part_number][first_date:date_add_months(date, fc_months+1)])
        
    df.columns = ["y"]
    df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
    df.index = df["ds"]
    df = df[["y"]]
    ts = df.copy()

    if enteros:
        ax.plot(ts["y"].round(), 'g', label='Demand', color="black")
    else:
        ax.plot(ts["y"], 'g', label='Demand', color="black") 

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # FC_BP
    future = fc_bp[(fc_bp["REF_CODE"] == part_number) & (fc_bp["date"] == date)]
    future = future[['_'+"%.2d" % i for i in range(1, fc_months+1)]]
    future.columns = months_between_list(date_add_months(date, 2), date_add_months(date, fc_months+1))
    future = future.T
    future.columns = ["y"]    
    future["ds"] = pd.to_datetime(future.index, format='%Y-%m-%d')
    future.index = future["ds"]
    if enteros:
        ax.plot(future["y"].round(), 'g', label='Forecast BP', color="blue")    
    else:
        ax.plot(future["y"], 'g', label='Forecast BP', color="blue")

    # FC_AA
    future = fc_aa[(fc_aa["REF_CODE"] == part_number) & (fc_aa["date"] == date)]
    future = future[['_'+"%.2d" % i for i in range(1, fc_months+1)]]
    future.columns = months_between_list(date_add_months(date, 2), date_add_months(date, fc_months+1))
    future = future.T
    future.columns = ["y"]    
    future["ds"] = pd.to_datetime(future.index, format='%Y-%m-%d')
    future.index = future["ds"]
    if enteros:
        ax.plot(future["y"].round(), 'g', label='Forecast AA', color="red")   
    else:
        ax.plot(future["y"], 'g', label='Forecast AA', color="red")

    plt.legend(loc='best')
    plt.grid(True)   
    plt.ylim(bottom=0)
    plt.show()



def plot_fc(part_number, date, dda, fc_bp, fc_future, fc_past, first_date='all', meses_historia="all", enteros=False, figsize=(20,5), fc_months=6):  
    """
    Plot de del forecast local guardado en FC/linea/fc_past/date/
    """
        
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(part_number) 
    
    if meses_historia=="all" and first_date=='all':
        df = pd.DataFrame(dda.loc[part_number][:date_add_months(date, fc_months+1)])
    if meses_historia!="all" and first_date=='all':
        df = pd.DataFrame(dda.loc[part_number][date_add_months(date, -meses_historia):date_add_months(date, fc_months+1)])
    if meses_historia=="all" and first_date!='all':
        df = pd.DataFrame(dda.loc[part_number][first_date:date_add_months(date, fc_months+1)])
    
    df.columns = ["y"]
    df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
    df.index = df["ds"]
    df = df[["y"]]
    ts = df.copy()
    if enteros:
        ax.plot(ts["y"].round(), 'g', label='Demand', color="black")
    else:
        ax.plot(ts["y"], 'g', label='Demand', color="black") 

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
          
    future = fc_bp[(fc_bp["REF_CODE"] == part_number) & (fc_bp["date"] == date)]
    future = future[['_'+"%.2d" % i for i in range(1, fc_months+1)]]
    future.columns = months_between_list(date_add_months(date, 2), date_add_months(date, fc_months+1))
    future = future.T
    future.columns = ["y"]    
    future["ds"] = pd.to_datetime(future.index, format='%Y-%m-%d')
    future.index = future["ds"]
    if enteros:
        ax.plot(future["y"].round(), 'g', label='Forecast BP', color="blue")    
    else:
        ax.plot(future["y"], 'g', label='Forecast BP', color="blue")
        
    future = pd.DataFrame(fc_future.yhat.loc[:date_add_months(date, fc_months+1)])
    future.columns = ["y"]    
    future["ds"] = pd.to_datetime(future.index, format='%Y-%m-%d')
    if enteros:
        ax.plot(future["y"].round(), 'g', label='Forecast AA', color="red")   
    else:
        ax.plot(future["y"], 'g', label='Forecast AA', color="red")
            
    plt.legend(loc='best')
    plt.grid(True) 
    plt.ylim(bottom=0)
    plt.show()

    
    
def plot(part_number, date, dda, fc_bp, fc_aa=None, fc_future=None, fc_past=None, FC=False, first_date='all', meses_historia="all", enteros=False, figsize=(20,5), fc_months=6):
    if FC:
        plot_fc(part_number, date, dda, fc_bp, fc_future, fc_past, first_date=first_date, meses_historia=meses_historia, enteros=enteros, figsize=figsize, fc_months=fc_months)
    if not FC:
        plot_ts(part_number, date, dda, fc_bp, fc_aa, first_date=first_date, meses_historia=meses_historia, enteros=enteros, figsize=figsize, fc_months=fc_months)


        
def get_local_fc(linea, fc_type, date, new=True):
    """retorna los forecast desde la carpeta local FC de la linea (linea) y el tipo de FC (fc_type)"""

    if new:
        folder_path = Path.cwd().joinpath("FC").joinpath(linea).joinpath(fc_type).joinpath(date.replace("-", "")) 
        folder_files = [file for file in os.listdir(folder_path) if file.find('FC_')==0 if file.find('-checkpoint.csv')==-1]        
        df_fc = pd.DataFrame()
        for file in folder_files:
            df = pd.read_csv(folder_path.joinpath(file))
            df_fc = df_fc.append(df)
        return df_fc
   
    if ~new:
        folder_path = Path.cwd().joinpath("FC").joinpath(linea).joinpath(fc_type) 
        folder_files = [file for file in os.listdir(folder_path) if file.find('FC_')==0]
        fc_df = pd.DataFrame()
        for file in folder_files:
            df = pd.read_csv(folder_path.joinpath(file))
            fc_df = fc_df.append(df)
        return fc_df

    
    
def trans_fc_sem(df, colname):
    df[colname] = df[["_01", "_02", "_03", "_04", "_05", "_06"]].sum(axis=1)
    df = df[["REF_CODE", colname]]
    df.set_index(["REF_CODE"], inplace=True)
    return df



def get_fc_sem(linea, date, FC='AA'):
    if not FC=='AA':
        df = get_local_fc(linea, FC, date)
        colname = FC
    if FC=='AA':
        df = get_fc(linea, partner='AA')#, AC=False
        colname = 'AA'
    df = trans_fc_sem(df, colname)
    return df



def get_dda_sem(linea, date, months=6):
    dda = get_dda(linea, base=False)#, AC=False
    df = pd.DataFrame(dda[months_between_list(date_add_months(date, 2), date_add_months(date, months+1))].sum(axis=1))
    df.columns = ["DA"]   
    return df



# def get_last_nonzero_date(part_number):
#     ts = dda.loc[part_number]
#     last_nonzero_index = [i for i, e in enumerate(ts) if e != 0][-1]
#     return ts.index[last_nonzero_index]



def get_masterdata(linea, date):
    
    if linea=='SSC':
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_masterdata`
        """
        client = bigquery.Client()
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    
    if linea=='MRO':
        df_rot = get_rot(linea)
        df_rot = df_rot[[date]]
        df_rot.columns = ['RC_ROTACION']

        df_rot['date'] = date
        df_rot.reset_index(inplace=True)
        
        df_rot.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace=True)

        df_price = pd.read_csv('gs://supply-forecast-aa/price_mro.csv')

        df = df_rot.set_index('REF_CODE').merge(df_price.set_index('Pref_Compra'), left_index=True, right_index=True)
        df = df[['RC_ROTACION', 'date', 'PMM']]
        df.rename(columns={"PMM": "RC_AVG_PRICE"}, inplace=True)
        
        df['REF_CODE'] = df.index
        df.reset_index(inplace=True, drop=True)
        
        return df
    
    if linea=='BRA':
        df_rot = get_rot(linea)
        df_rot = df_rot[[date]]
        df_rot.columns = ['RC_ROTACION']

        df_rot['date'] = date
        df_rot.reset_index(inplace=True)

        df_rot.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace=True)

        df_price = pd.read_csv('gs://supply-forecast-aa/price_bra.csv')
        df_price = df_price[['Material', 'Classe_FFF', 'Família', 'Classificação', 'Preço USD']]

        df_familia = df_price.groupby("Família").mean()
        df_familia.columns = ['P_FAMILIA']
        df_familia.reset_index(inplace=True)

        df = df_price.merge(df_familia, left_on='Família', right_on='Família')

        df1 = df[df['Preço USD'].notnull()]
        df1['RC_AVG_PRICE'] = df1['Preço USD']

        df2 = df[df['Preço USD'].isnull()] 
        df2['RC_AVG_PRICE'] = df2['P_FAMILIA']

        df = df1.append(df2)
        df = df.dropna()
        df_price = df.copy()

        df = df_rot.merge(df_price, left_on='REF_CODE', right_on='Classe_FFF')
        df = df[['REF_CODE', 'date', 'RC_ROTACION', 'Classificação', 'RC_AVG_PRICE']]
        df['RC_CLASS_SUPPLY'] = 'OTROS'
        df.loc[df['Classificação'] == "Consumível", 'RC_CLASS_SUPPLY'] = 'CONSUMO'
        df.loc[df['Classificação'] == "Componente", 'RC_CLASS_SUPPLY'] = 'COMPONENTE'
        
        return df


    
def get_err(linea, date, FC='AA', PQ=False, output='DATA', metric='ABS_ERROR', aggr='ROT', REL_ERROR=False):
    
    """
    metric=['ABS_ERROR', 'ERROR']
    output=['DATA', 'PLOT']
    aggr=['ROT', 'CLASS']
    
    """

    fc_cv = get_fc_sem(linea, date, FC)

    dda = get_dda_sem(linea, date)

    df = get_fc(linea, partner='BP', AC=False)
    df = df[df.date==date]
    fc_bp = trans_fc_sem(df, 'BP')

    masterdata = get_masterdata(linea, date)
    masterdata.drop_duplicates(subset='REF_CODE', keep='first', inplace=True)
    masterdata.set_index('REF_CODE', inplace=True)

    data = fc_cv.merge(dda, left_index=True, right_index=True).merge(fc_bp, left_index=True, right_index=True).merge(masterdata, left_index=True, right_index=True, how='left')
    print('PNs :', data.shape[0])

    if not PQ:
        if metric=='ABS_ERROR':
            data['ERRABS_' + FC] = abs(data[FC] - data['DA']) 
            data['ERRABS_BP'] = abs(data['BP'] - data['DA']) 
        if metric=='ERROR':
            data['ERRABS_' + FC] = data[FC] - data['DA'] 
            data['ERRABS_BP'] = data['BP'] - data['DA'] 

    if PQ:
        if metric=='ABS_ERROR':
            data['ERRABS_' + FC] = abs(data[FC] - data['DA']) * data['RC_AVG_PRICE']
            data['ERRABS_BP'] = abs(data['BP'] - data['DA']) * data['RC_AVG_PRICE']
        if metric=='ERROR':
            data['ERRABS_' + FC] = (data[FC] - data['DA']) * data['RC_AVG_PRICE']
            data['ERRABS_BP'] = (data['BP'] - data['DA']) * data['RC_AVG_PRICE']
            
    if output=='PLOT':
        if aggr=='ROT':
            data_group = 'RC_ROTACION'
        if aggr=='CLASS':
            data_group = 'RC_CLASS_SUPPLY'           
        
        if REL_ERROR:
            df = data.groupby(data_group).sum()
            if PQ:
                df['DA'] = df['DA'] * df['RC_AVG_PRICE']
                df['ERRABS_'+ FC] = df['ERRABS_'+ FC] / df['DA']
                df['ERRABS_BP'] = df['ERRABS_BP'] / df['DA']
            if not PQ:
                df['ERRABS_'+ FC] = df['ERRABS_'+ FC] / df['DA']
                df['ERRABS_BP'] = df['ERRABS_BP'] / df['DA']

        if not REL_ERROR:
            df = data.groupby(data_group).sum()

        if aggr=='ROT':
            df = df.reindex(['FM', 'MM', 'SM', 'NM'])
        if aggr=='CLASS':
            df = df.reindex(['COMPONENTE', 'CONSUMO', 'OTROS'])        

        rot=df.index
        fig = go.Figure(data=[
            go.Bar(name='AA', x=rot, y=df['ERRABS_'+ FC], text=df['ERRABS_'+ FC], textposition='auto'),
            go.Bar(name='BP', x=rot, y=df['ERRABS_BP'], text=df['ERRABS_BP'], textposition='auto'),
        ])
        fig.update_layout(barmode='group')
        
        if metric=='ABS_ERROR':
            title_error = 'Error Absoluto'
        if metric=='ERROR':
            title_error = 'Error'
       
        if PQ:
            fig.update_layout(
                title= title_error + ' en relación a la Demanda real: Monto Capital $USD (P*Q) (' + linea +', 6 meses a continuación del mes ' + date + ')',
                yaxis_title="$USD")
        if not PQ:
            fig.update_layout(
                title= title_error + " en relación a la Demanda real: Cantidad de Materiales (" + linea +', 6 meses a continuación del mes ' + date + ')',
                yaxis_title="Unidades")
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.show() 

    if output=='DATA':
        return data


def get_cantidad(linea, date, FC='AA', PQ=False, output='DATA', aggr='ROT'):
    
    """
    Calcula el Forecast para un PN
    FC=['AA', 'MPARAMS', 'CV_OUTLIERS'] define Forecast (AA devuelve el Forecast oficial de Asvanced Analytics desde BigQuery)
    output=['DATA', 'PLOT', 'DEMAND'] define Output
    aggr=['ROT', 'CLASS']
    
    """
    
    fc_cv = get_fc_sem(linea, date, FC=FC)
    dda = get_dda_sem(linea, date)

    df = get_fc(linea, partner='BP', AC=False)
    df = df[df.date==date]
    fc_bp = trans_fc_sem(df, 'BP')

    masterdata = get_masterdata(linea, date)
    masterdata.drop_duplicates(subset='REF_CODE', keep='first', inplace=True)
    masterdata.set_index('REF_CODE', inplace=True)

    data = fc_cv.merge(dda, left_index=True, right_index=True).merge(fc_bp, left_index=True, right_index=True).merge(masterdata, left_index=True, right_index=True, how='left')
        
    print('PNs :', data.shape[0])

    if not PQ:
        data['CANT_' + FC] = data[FC] 
        data['CANT_BP'] = data['BP']
        data['CANT_DA'] = data['DA']
    if PQ:
        data['CANT_' + FC] = data[FC]  * data['RC_AVG_PRICE']
        data['CANT_BP'] = data['BP'] * data['RC_AVG_PRICE']
        data['CANT_DA'] = data['DA'] * data['RC_AVG_PRICE']
        
    if output=='PLOT':
        if aggr=='ROT':
            data_group = 'RC_ROTACION'
        if aggr=='CLASS':
            data_group = 'RC_CLASS_SUPPLY'        
        df = data.groupby(data_group).sum()

        if aggr=='ROT':
            df = df.reindex(['FM', 'MM', 'SM', 'NM'])
        if aggr=='CLASS':

            df = df.reindex(['COMPONENTE', 'CONSUMO', 'OTROS'])

        rot=df.index
        
        if not output=='DEMAND':
            fig = go.Figure(data=[
                go.Bar(name='AA', x=rot, y=df['CANT_'+ FC], text=df['CANT_'+ FC], textposition='auto'),
                go.Bar(name='BP', x=rot, y=df['CANT_BP'], text=df['CANT_BP'], textposition='auto'),
                go.Bar(name='DA', x=rot, y=df['CANT_DA'], text=df['CANT_DA'], textposition='auto')
            ])    
            
        if output=='DEMAND':
            fig = go.Figure(data=[
                go.Bar(name='DA', x=rot, y=df['CANT_DA'], text=df['CANT_DA'], textposition='auto')
            ])        
            
            
        fig.update_layout(barmode='group')
        if PQ:
            fig.update_layout(
                title='Monto Capital $USD (P*Q) (' + linea +', 6 meses a continuación del mes ' + date + ')',
                yaxis_title="$USD")
        if not PQ:
            fig.update_layout(
                title="Cantidad de Materiales (" + linea +', 6 meses a continuación del mes ' + date + ')',
                yaxis_title="Unidades")
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.show() 

    if output=='DATA':
        return data


    
    
def get_obsoletos(linea, date, FC='AA', PQ=False, output_data=True, output_plot=False, saving_plot_only=True, metric='ABS_ERROR'):

    data = get_err(linea, date, FC, metric=metric) 

    dda = get_dda(linea, base=False, AC=False)
    dda = dda[dda.sum(axis=1)>0]
    PNs = list(set(dda.index).intersection(set(data.index)))
    CPU_n = os.cpu_count()
    with Pool(CPU_n) as p:
        result = p.map(get_last_nonzero_date, PNs)

    df = pd.DataFrame()
    df['REF_CODE'] = list(PNs)
    df['OBS_DATE'] = result
    df_last_nonzero_date = df.copy()
    df_last_nonzero_date = df_last_nonzero_date.set_index('REF_CODE')

    data = data.merge(df_last_nonzero_date, left_index=True, right_index=True, how='left')

    data['date'] = pd.to_datetime(date)
    data['OBS_DATE'] = pd.to_datetime(data['OBS_DATE'])
    data['DIAS_PARA_OBS'] = data['OBS_DATE'] - data['date']
    data['DIAS_PARA_OBS'] = data['DIAS_PARA_OBS'] / np.timedelta64(1,'D')

    if not PQ:
        data['CANT_' + FC] = data[FC] 
        data['CANT_BP'] = data['BP']
        data['BP_' + FC] = data['BP'] - data[FC]
    if PQ:
        data['CANT_' + FC] = data[FC]  * data['RC_PRECIO']
        data['CANT_BP'] = data['BP'] * data['RC_PRECIO']
        data['BP_' + FC] = (data['BP'] - data[FC])  *  data['RC_PRECIO']

    data = data[np.logical_and(data['DIAS_PARA_OBS']<0, data['BP_' + FC]>0)]
    print('Pns Obsoletos :', data.shape[0])
    print('Ahorro en materiales por obsolescencia : ', int(data['BP_' + FC].sum()))

    if output_plot:
        df = data.groupby('RC_ROTACION').sum()  
        df_agg = pd.DataFrame(df.sum()).T
        df_agg.index = ['AGG']
        df = df.append(df_agg)
        df = df.reindex(['AGG', 'FM', 'MM', 'SM', 'NM'])

        rot=df.index

    if output_data:
        return data3 

    

def accuracy(part_number, date, dda, fc_bp, fc_aa=None, fc_future=None, fc_past=None, fc_months=12, FC=False, metric='ABS_ERROR'):  
    """
    metric=['ABS_ERROR', 'ERROR']
    """
    
    df = pd.DataFrame(dda.loc[part_number][date_add_months(date, 2):date_add_months(date, fc_months+1)])
    df.columns = ["da"]
    df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
    df.index = df["ds"]
    df = df[["da"]]
    da = df.copy()

    # FC_BP
    df = fc_bp[(fc_bp["REF_CODE"] == part_number) & (fc_bp["date"] == date)]
    df = df[['_'+"%.2d" % i for i in range(1, fc_months+1)]]
    df.columns = months_between_list(date_add_months(date, 2), date_add_months(date, fc_months+1))
    df = df.T
    df.columns = ["bp"]    
    df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
    df.index = df["ds"]
    df = df[["bp"]]
    bp = df.copy()

    # FC_AA
    if not FC:
        df = fc_aa[(fc_aa["REF_CODE"] == part_number) & (fc_aa["date"] == date)]
        df = df[['_'+"%.2d" % i for i in range(1, fc_months+1)]]
        df.columns = months_between_list(date_add_months(date, 2), date_add_months(date, fc_months+1))
        df = df.T
        df.columns = ["aa"]    
        df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
        df.index = df["ds"]
        df = df[["aa"]]
        aa = df.copy()
    
    if FC:
        df = fc_future[['yhat']]
        df = df.loc[date_add_months(date, 2): date_add_months(date, fc_months+1)]
        df.columns = ["aa"]    
        df["ds"] = pd.to_datetime(df.index, format='%Y-%m-%d')
        df.index = df["ds"]
        df = df[["aa"]]
        aa = df.copy()
    
    df = da.merge(aa, left_index=True, right_index=True).merge(bp, left_index=True, right_index=True)
    
    if metric=='ABS_ERROR':
        err_aa = abs(df.sum().aa - df.sum().da)
        err_bp = abs(df.sum().bp - df.sum().da)
    if metric=='ERROR':
        err_aa = df.sum().aa - df.sum().da
        err_bp = df.sum().bp - df.sum().da   
    
    diff = err_bp - err_aa
    
    return {'AA': err_aa, 'BP': err_bp, 'diff': diff}