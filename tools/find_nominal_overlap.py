import pandas as pd
import sqlite3
import numpy as np
import os
from copy import deepcopy
import multiprocessing

def get_sys_database(folder):
    if os.path.isdir('/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_%s'%folder):
        return '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_%s/data/dev_lvl7_robustness_muon_neutrino_%s.db'%(folder,folder)
    elif os.path.isdir('/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_neutrino_%s'%folder):
        return '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_neutrino_%s/data/dev_lvl7_robustness_neutrino_%s.db'%(folder,folder)
    else:
        print('SYSTEMATIC DATABASE NOT FOUND FOR %s'%folder)

def get_sql_overlap(data_older, folder, events,key):
    database_nominal = '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db'
    database_sys = get_sys_database(folder) #'/remote/ceph/user/o/oersoe/databases/dev_lvl7_robustness_muon_neutrino_%s/data/dev_lvl7_robustness_muon_neutrino_%s.db'%(folder,folder)
    print(database_sys)
    # Create unique ID based on truth energy, zenith, and azimuth
    # -- Nominal dataset
    with sqlite3.connect(database_nominal) as con:
        df_lookup_nomimal = pd.read_sql(
    """
    SELECT (CAST(energy AS str) || '-' || CAST(zenith AS str) || '-' || CAST(azimuth AS str)) as UID, event_no as event_no_nominal
    FROM truth
    """, con)

    # -- Systematic variation
    with sqlite3.connect(database_sys) as con:
        df_lookup_variation = pd.read_sql(
    """
    SELECT (CAST(energy AS str) || '-' || CAST(zenith AS str) || '-' || CAST(azimuth AS str)) as UID, event_no as event_no_variation
    FROM truth
    """, con)
    # Get the event_no's that are in both the nominal and systematic variation dataset
    df_event_nos_overlap = df_lookup_nomimal.merge(df_lookup_variation, on="UID", how="inner").drop(columns=["UID"])

    # Dummy reco selection to prove the point
    #reco_selection = df_lookup_nomimal[['event_no_nominal']].rename(columns={'event_no_nominal': 'event_no'}).sample(frac=0.5, random_state=21, replace=False)

    # Get the UIDs for event_no's in `reco_selection`
    df_lookup_nominal_reco = df_lookup_nomimal.merge(events, left_on="event_no_nominal", right_on="event_no", how="inner").drop(columns=["event_no"])

    # Get the event_no's that are in both the nominal and systematic variation dataset *and* that are also in `reco_selection`
    df_event_nos_overlap_reco = df_lookup_nominal_reco.merge(df_lookup_variation, on="UID", how="inner").drop(columns=["UID"])

    with sqlite3.connect(database_nominal) as con:
        query = 'select event_no from truth where event_no in %s and event_no not in %s'%(str(tuple(events['event_no'])),str(tuple(df_event_nos_overlap_reco['event_no_nominal'])))
        nominal_not_in_sys = pd.read_sql(query,con)
        
    nominal_not_in_sys.columns = ['event_no_nominal']
    nominal_not_in_sys['%s_overlap'%folder] = 0
    nominal_in_sys = df_event_nos_overlap_reco.loc[:,['event_no_nominal']]
    nominal_in_sys['%s_overlap'%folder] = 1
    nominal_labels = nominal_not_in_sys.append(nominal_in_sys, ignore_index = True).sort_values('event_no_nominal').reset_index(drop = 'True')


    with sqlite3.connect(database_sys) as con:
        query = 'select event_no from truth where event_no not in %s'%(str(tuple(df_event_nos_overlap_reco['event_no_variation'])))
        sys_not_in_nominal = pd.read_sql(query,con)
    sys_not_in_nominal.columns = ['event_no_variation']
    sys_not_in_nominal['%s_overlap'%key] = 0

    sys_in_nominal = df_event_nos_overlap_reco.loc[:,['event_no_variation']]
    sys_in_nominal['%s_overlap'%key] = 1
    sys_labels = sys_not_in_nominal.append(sys_in_nominal, ignore_index = True).sort_values('event_no_variation').reset_index(drop = 'True')

    print('sys in nominal %s' % len(sys_in_nominal))
    print('sys not in nominal %s' % len(sys_not_in_nominal))
    return nominal_labels, sys_labels


def get_overlap(settings):
    nominal_dict, data_folder, folder,  id, outdir = settings
    os.makedirs(outdir + '/data_with_overlap_labels', exist_ok= True)
    os.makedirs(outdir + '/data_with_overlap_labels/%s'%folder, exist_ok= True)
    os.makedirs(outdir + '/data_with_overlap_labels/%s/tmp'%folder, exist_ok= True)
    sys = pd.read_csv(data_folder + '/' + folder + '/' + 'everything.csv').sort_values('event_no').reset_index(drop = 'True')
    print('---sys---')
    print(len(sys))
    print(len(pd.unique(sys['event_no'])))
    print('-----')
    for key in nominal_dict.keys():
        nominal_df = nominal_dict[key].sort_values('event_no').reset_index(drop = 'True')
        nominal_labels, sys_labels = get_sql_overlap(data_folder,folder, nominal_df, key)

        nominal_df['%s_overlap'%folder] = nominal_labels['%s_overlap'%folder]

        sys['%s_overlap'%key] = sys_labels['%s_overlap'%key]
        print(sum(nominal_df['%s_overlap'%folder] == 1))
        print(sum(sys['%s_overlap'%key] == 1))
        print('----')
        nominal_df.to_csv(outdir + '/data_with_overlap_labels/%s/tmp/%s_in_%s.csv'%(folder, key, folder))

    sys.to_csv(outdir + '/data_with_overlap_labels/%s/everything.csv'%folder)
    return 


def make_overlapping_event_labels(data_folder, outdir):
    folders = os.listdir(data_folder)
    nominal_dict = {'reco': pd.read_csv(data_folder + '/' + '0000' + '/' + 'reconstruction.csv').reset_index(drop = True),
                    'signal': pd.read_csv(data_folder + '/' + '0000' + '/' + 'signal.csv').reset_index(drop = True), 
                    'track': pd.read_csv(data_folder + '/' + '0000' + '/' + 'track_cascade.csv').reset_index(drop = True)}
    settings = []
    id = 0
    for folder in folders:
        if '0000' != folder:
            settings.append([deepcopy(nominal_dict), data_folder, folder, id, outdir])
            id +=1
    print(folders)
    p = multiprocessing.Pool(processes = len(settings))
    async_result = p.map_async(get_overlap, settings)
    p.close()
    p.join()
    #print(settings[0])
    #get_overlap(settings[0])
    
    return

def merge_nominal_files(outdir):
    os.makedirs(outdir +'/data_with_overlap_labels/0000', exist_ok=True)
    folders = os.listdir(outdir + '/data_with_overlap_labels')
    is_first = True
    for folder in folders:
        print(folder)
        if folder != '0000':
            if is_first:
                reco = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/reco_in_%s.csv'%folder)
                signal = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/signal_in_%s.csv'%folder)
                track = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/track_in_%s.csv'%folder)
                is_first = False
            else:
                reco['%s_overlap'%folder] = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/reco_in_%s.csv'%folder)['%s_overlap'%folder]
                signal['%s_overlap'%folder] = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/signal_in_%s.csv'%folder)['%s_overlap'%folder]
                track['%s_overlap'%folder] = pd.read_csv(outdir +'/data_with_overlap_labels/' +folder + '/tmp/track_in_%s.csv'%folder)['%s_overlap'%folder]
    reco.to_csv(outdir +'/data_with_overlap_labels/0000/reconstruction.csv')
    signal.to_csv(outdir +'/data_with_overlap_labels/0000/signal.csv')
    track.to_csv(outdir +'/data_with_overlap_labels/0000/track_cascade.csv')
    

data_folder = '/mnt/scratch/rasmus_orsoe/paper_data_pass2/data'
outdir = '/mnt/scratch/rasmus_orsoe/paper_data_pass2'

if __name__ == '__main__':
    make_overlapping_event_labels(data_folder, outdir)
    merge_nominal_files(outdir)

