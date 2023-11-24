import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import os
from datetime import datetime
from dateutil import relativedelta
from pathlib import Path
import glob


def get_dda(linea, base=False): #, AC=True
    
#     if linea=="SSC" and AC:
#         query = """
#             SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_dda_ac`
#         """

#     if linea=="BRA" and AC:
#         query = """
#             SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.bra_dda_ac`
#         """
    
#     if linea=="MRO" and AC:
#         query = """
#             SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_dda_ac`
#         """

    if linea=="SSC":# and not AC:
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_dda`
        """

    if linea=="BRA":# and not AC:
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.bra_dda`
        """
    
    if linea=="MRO":# and not AC:
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_dda`
        """

        
    client = bigquery.Client()
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    df.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace=True)
    
    if linea=="SSC" and not base:
        df = df.groupby(df["REF_CODE"]).sum()       
    if linea=="SSC" and base:
        df.set_index(["REF_CODE", "BASE"], inplace=True)       
    if linea=="BRA" and not base:
        df = df.groupby(df["REF_CODE"]).sum()       
    if linea=="BRA" and base:
        df.set_index(["REF_CODE", "BASE"], inplace=True)
    if linea=="MRO": 
        df = df.groupby(df["REF_CODE"]).sum()  
        
    df.columns = [col[1:].replace("_", "-") for col in df.columns]
    

    return df



def get_fc_bp(linea):

    if linea=="SSC":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_fc_bp`
        """
        client = bigquery.Client()
        query_job = client.query(query)
        df = query_job.to_dataframe()
        df.rename({'forecast_date': 'date'}, axis=1, inplace=True)
        df.set_index(['REF_CODE', 'date'], inplace=True)
        df.columns = [col.replace('m', '') for col in df.columns]
        df.reset_index(inplace=True)

    if linea=="BRA":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.bra_fc_bp`
        """
        client = bigquery.Client()
        query_job = client.query(query)
        df = query_job.to_dataframe()

    if linea=="MRO":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_fc_bp`
        """
        client = bigquery.Client()
        query_job = client.query(query)

        df = query_job.to_dataframe()
        df.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace = True)
      
    return df



def get_fc_aa(linea): #, AC=True
    
#     if linea=="SSC" and AC:
#         client = bigquery.Client()
#         query = """
#             SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_fc_aa_ac`
#         """

#         query_job = client.query(query)
#         df = query_job.to_dataframe()
#         df.rename({'forecast_date': 'date'}, axis=1, inplace=True)

        
    if linea=="SSC": #and not AC
        client = bigquery.Client()
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_fc_aa`
        """

        query_job = client.query(query)
        df = query_job.to_dataframe()
        df.rename({'forecast_date': 'date'}, axis=1, inplace=True)
        
        
    if linea=="BRA":
        client = bigquery.Client()

        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.bra_fc_aa_ac`
        """

        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        
    if linea=="MRO" and AC:
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_fc_aa_ac`
        """

        client = bigquery.Client()
        query_job = client.query(query)

        df = query_job.to_dataframe()
        df.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace = True)

        
    if linea=="MRO" and not AC:
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_fc_aa`
        """

        client = bigquery.Client()
        query_job = client.query(query)

        df = query_job.to_dataframe()
        df.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace = True)
        
    
    return df
        

    
def get_fc(linea, partner="AA"):#, AC=True
    if partner=="AA": #and AC
        df = get_fc_aa(linea)
#     if partner=="AA" and not AC:
#         df = get_fc_aa(linea)#, AC=False
    if partner=="BP":
        df = get_fc_bp(linea)        
    return df
       


def months_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    
    r = relativedelta.relativedelta(d1, d2)
    
    m = np.abs(r.months)
    y = np.abs(r.years)

    diff_months = y*12 + m +1
    return diff_months



def months_between_list(date_t0, date_t1):
    months_list = []
    months_list.append(date_t0)

    for t in range(1, months_between(date_t0, date_t1)):
        date_t = pd.Timestamp(months_list[t-1]) + pd.DateOffset(months=1)
        date_t = date_t.date().strftime('%Y-%m-%d')
        months_list.append(date_t)
    
    return months_list



def date_add_months(date, n):
    date = datetime.strptime(date, "%Y-%m-%d")
    date = pd.Timestamp(date) + pd.DateOffset(months=n - np.sign(n))
    date = date.date().strftime('%Y-%m-%d')
    
    return date



def get_ac(linea):
    df = pd.read_csv("gs://supply-forecast/supply-forecast/" + linea.lower() + "_ambiente_controlado.csv")
    ac = df["0"].to_list()
    return ac



def get_input_dda(linea, RC_BASE=False):
    if linea=="SSC":
        df = pd.read_csv("gs://supply-forecast/" + linea + "/INPUT/DEMAND/" + dda_list[0])
        df.rename(columns={"REFERENCE_CODE": "REF_CODE"}, inplace = True)
        if not RC_BASE:
            df = df.groupby(df["REF_CODE"]).sum()
        if RC_BASE:
            df.set_index(['REF_CODE', 'BASE'], inplace=True)
        df.columns = [c.replace(" 00:00:00", "") for c in df.columns]
        return df

    if linea=="BRA":
        df = pd.read_csv("gs://supply-forecast/" + linea + "/INPUT/DEMAND/" + dda_list[0]) 
        df.drop(columns=["PN", "DESCRICAO", "TIPO_MATERIAL", "CLASSIFICACAO", "D24M", "D12M", "MEDIA12M", "FREQ12M", "CLASS_FREQ"], inplace=True)
        df.rename(columns={"PPN": "REF_CODE"}, inplace = True)
        
        df.set_index(['REF_CODE', 'BASE'], inplace=True)
        df[df < 0] = 0
        df.reset_index(inplace=True)
        if not RC_BASE:
            df = df.groupby(df["REF_CODE"]).sum()
        if RC_BASE:
            df.set_index(['REF_CODE', 'BASE'], inplace=True)
        return df

    if linea=="MRO":
        df = pd.read_csv("gs://supply-forecast/" + linea + "/INPUT/DEMAND/" + dda_list[0], sep=',')
        df = df.groupby(df["REF_CODE"]).sum()
        df.columns = [c.split("/")[2] + "-" + c.split("/")[1] + "-" + c.split("/")[0] for c in df.columns]
        df = df[["2021-01-01", "2021-02-01", "2021-03-01"]]
        return df


    
def get_rot(linea):
    
    if linea=="SSC":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.ssc_rot_aa`
        """

    if linea=="BRA":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.bra_rot_aa`
        """
    
    if linea=="MRO":
        query = """
            SELECT * FROM `bc-ed-advaa-dev-nc0h.supply_forecast.mro_rot_aa`
        """
        
    client = bigquery.Client()
    query_job = client.query(query)
    df = query_job.to_dataframe()
        
    df.set_index('REFERENCE_CODE', inplace=True)
    df.columns = [col[1:].replace("_", "-") for col in df.columns]
    
    return df



def movecol(df, col_to_move='REF_CODE', position_to_move=0):
    
    cols = df.columns.tolist()
    cols.remove(col_to_move)
    cols.insert(position_to_move, col_to_move)
    df_moved = df[cols]

    return df_moved



def ordenar_fechas(dates):
    dates.sort(key = lambda date: datetime.strptime(date, "%Y-%m-%d"))
    return dates



def get_fh():
    fh = pd.read_csv('DATA/fh.csv')
    fh['ds'] = pd.to_datetime(fh['date'])
    fh.drop('date', axis='columns', inplace=True)
    return fh



def download_from_bucket(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)

    output_file_name = blob_name
    blob.download_to_filename(blob_name)

    print("Downloaded blob {} to {}.".format(blob.name, output_file_name))
    
    
    
def get_fc_from_folder(linea, fc_type):
    """retorna los forecast desde la carpeta local FC de la linea (linea) y el tipo de FC (fc_type)"""
    
    folder_path = Path.cwd().joinpath("FC").joinpath(linea).joinpath(fc_type) 
    folder_files = [file for file in os.listdir(folder_path) if file.find('FC_')==0]

    fc_df = pd.DataFrame()
    for file in folder_files:
        df = pd.read_csv(folder_path.joinpath(file))
        fc_df = fc_df.append(df)

    return fc_df



def get_fc_all(linea):
    """retorna los forecast desde la carpeta local FC para la linea"""
    
    folder_path = Path.cwd().joinpath("FC").joinpath(linea)
    types = [file for file in os.listdir(folder_path) if file.find('.ipynb_checkpoints')<0]
    types

    fc_df = pd.DataFrame()
    for fc_type in types:
        df = get_fc_from_folder(linea, fc_type)
        df['fc_type'] = fc_type
        fc_df = fc_df.append(df)

    return fc_df



def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path, date):
    assert os.path.isdir(local_path)
    
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file), date)
        else:
            blob_name = os.path.join('FC_RESPALDO', date, local_path, local_file[1 + len(local_path):])
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
            
            

def hacer_respaldo_fc_local():
    local_path = 'FC'
    bucket_name = 'supply-forecast-aa'
    gcs_path = 'supply-forecast-aa'
    date = str(datetime.now()).replace('-','').replace(' ','_').replace(':','').replace('.','_')
    upload_local_directory_to_gcs(local_path, bucket_name, bucket_name, date)
    
    
    
if __name__ == '__main__':
    hacer_respaldo_fc_local()
    
    