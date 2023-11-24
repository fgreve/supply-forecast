import multiprocessing
from itertools import product
from utils import *
from supply_forecast import *
import argparse
import os
import sys
from pathlib import Path
from shutil import copyfile


my_parser = argparse.ArgumentParser(prog='supply_forecast', 
                                    usage='%(prog)s [options] path',
                                    description='Genera una simulación de Forecast',
                                    epilog='Advanced Analytics - LATAM Airlines')

my_parser.version = '1.0'
my_parser.add_argument('-linea', action='store', default='SSC', help='Linea de forecast')
my_parser.add_argument('-date', action='store', default='2019-01-01', help='Mes para la simulación de forecast')
my_parser.add_argument('-pn', action='store', default='006605100001-R5', help='PartNumber / RefCode al que que quiere calcular Forecast')
my_parser.add_argument('-pn1', action='store_true', help='Para correr 1 part_number')
my_parser.add_argument('-pnall', action='store_true', help='Para correr una simulación con todos los PNs')
my_parser.add_argument('-folder', action='store', default='FC_CV', help='Folder donde se guarda el forecast')
my_parser.add_argument('-mparams', action='store_true', help='Para correr una simulación con todos los PNs')

args = my_parser.parse_args()
print(vars(args))

linea = vars(args)['linea']
date = vars(args)['date']
part_number = vars(args)['pn']
fc_folder = vars(args)['folder']


if vars(args)['mparams']:
    from fc_params import parametros
    
    def run_prophet_pn(pn_rot):
        try:
            with timeout(60*5, exception=RuntimeError):
                return multiparams_prophet(pn_rot, date, dda, parametros)
        except:
            return run_prophet(pn_rot[0], date, dda, growth='flat')   


from fc_params import CV
     
if CV:
    from fc_params import model_type
    
    def run_prophet_pn(pn_rot):
        try:
            with timeout(60*2, exception=RuntimeError):
                return cv_prophet(pn_rot, date, dda, model_type)
        except:
            return run_prophet(part_number, date, dda, growth='flat')    
    
dda = get_dda(linea, base=False) 


if vars(args)['pn1']:
    fc = run_prophet_pn(part_number)
    print(fc)

if vars(args)['pnall']:
    path = os.path.join('FC', linea, fc_folder, date.replace("-", ""))
    print('Simulation folder : ', path)
    
    os.makedirs(path, exist_ok=False)
    copyfile('fc_params.py', os.path.join(path, 'fc_params.py'))
    
    fh = get_fh()

    with open('DATA/PNs_' + linea + '.json') as f:
        PNs = json.load(f)
        
    print('PNs a simular : ', str(len(PNs[date])))
    
    rot = get_rot(linea)
    pns_rots = zip(PNs[date], list(rot.reindex(PNs[date])[date]))
    pns_rots = list(pns_rots)

    CPU_n = os.cpu_count()
    print('N nucleos: ', CPU_n)

    a_list = list(range(0, len(pns_rots), 1000))
    a_list.append(len(pns_rots))

    for index, elem in enumerate(a_list):
        if index+1 < len(a_list):
            start_time = time.time() 

            curr_el = str(elem)
            next_el = str(a_list[index+1])
            pns = pns_rots[int(curr_el): int(next_el)]
            
            filepath = os.path.join(path, "FC_" + linea + "_" + fc_folder + "_" + date.replace("-", "") + '_' + curr_el + '_' + next_el + ".csv")

            with Pool(CPU_n) as p:
                result = p.map(run_prophet_pn, pns) 
            df = fc_to_df(result, [i[0] for i in pns], date)    
            df.to_csv(filepath, index=False)

            print("date: ", date)
            print("filepath: ", filepath)
            end_time = time.time()
            print("Tiempo proceso (minutos): ", (end_time - start_time)/60)

            
            