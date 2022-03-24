import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from time import strftime,gmtime
import os
import matplotlib as mpl
mpl.use('pdf')
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import multiprocessing
from pathlib import Path
from scipy import stats
from scipy import stats
from copy import deepcopy
from os.path import exists
import pickle

def parallel_50th_error(settings):
    queue, n_samples, batch_size, diff = settings
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        new_sample = rng.choice(diff, size = batch_size, replace = True)
        queue.put(np.percentile(new_sample,50))
    multiprocessing.current_process().close()

def parallel_width_error(settings):
    queue, n_samples, batch_size, diff = settings
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        new_sample = rng.choice(diff, size = batch_size, replace = True)
        queue.put([np.percentile(new_sample,84),np.percentile(new_sample,16)])
    multiprocessing.current_process().close()

def AddSignature(db, df):
    events  = df['event_no']
    with sqlite3.connect(db) as con:
        query = 'select event_no, pid, interaction_type from truth where event_no in %s'%str(tuple(events))
        data = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
        
    df = df.sort_values('event_no').reset_index(drop = 'True')
    df['signature'] =  int((abs(data['pid']) == 14) & (data['interaction_type'] == 1))
    return df


def CalculateWidth(bias_tmp):
    return (np.percentile(bias_tmp,84) - np.percentile(bias_tmp,16))/2
    #return (np.percentile(bias_tmp,75) - np.percentile(bias_tmp,25))/1.365

def gauss_pdf(mean, std, x):
    pdf =  1/(std*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((x-mean)/std)**2)
    return (pdf).reset_index(drop = True)

def gaussian_pdf(x,diff):
    dist = getattr(stats, 'norm')
    parameters = dist.fit(diff)
    pdf = gauss_pdf(parameters[0],parameters[1],diff)[x]
    #print(pdf)
    return pdf

def laplacian_pdf(x,diff):
    return stats.laplace.pdf(diff)[x]

def add_50th_error(diff, laplace = False):
    if __name__ == '__main__':
        manager = multiprocessing.Manager()
        q = manager.Queue()
        total_samples = 10000
        batch_size = len(diff)
        n_workers = 100
        samples_pr_worker = int(total_samples/n_workers)
        settings = []
        for i in range(n_workers):
            settings.append([q, samples_pr_worker, batch_size, diff])
        p = multiprocessing.Pool(processes = len(settings))
        async_result = p.map_async(parallel_50th_error, settings)
        p.close()
        p.join()
        p50 = []
        queue_empty = q.empty()
        while(queue_empty == False):
            queue_empty = q.empty()
            if queue_empty == False:
                p50.append(q.get())
        return np.std(p50)

def CalculateWidthError(diff, laplace = False):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    total_samples = 10000
    batch_size = len(diff)
    n_workers = 100
    samples_pr_worker = int(total_samples/n_workers)
    settings = []
    for i in range(n_workers):
        settings.append([q, samples_pr_worker, batch_size, diff])
    p = multiprocessing.Pool(processes = len(settings))
    async_result = p.map_async(parallel_width_error, settings)
    p.close()
    p.join()
    p16 = []
    p84 = []
    queue_empty = q.empty()
    while(queue_empty == False):
        queue_empty = q.empty()
        if queue_empty == False:
            item = q.get()
            p84.append(item[0])
            p16.append(item[1])
    return np.sqrt(np.std(p16)**2 + np.std(p84)**2)
   
def convert_to_unit_vectors(data, post_fix):
    
    data['x'] = np.cos(data['azimuth'])*np.sin(data['zenith'])
    data['y'] = np.sin(data['azimuth'])*np.sin(data['zenith'])
    data['z'] = np.cos(data['zenith'])

    data['x' + post_fix] = np.cos(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['y' + post_fix] = np.sin(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['z' + post_fix] = np.cos(data['zenith' + post_fix])
    return data

def calculate_angular_difference(data, is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    data = convert_to_unit_vectors(data, post_fix)
    dotprod = (data['x']*data['x' + post_fix].values + data['y']*data['y'+ post_fix].values + data['z']*data['z'+ post_fix].values)
    norm_data = np.sqrt(data['x'+ post_fix]**2 + data['y'+ post_fix]**2 + data['z'+ post_fix]**2).values
    norm_truth = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2).values

    cos_angle = dotprod/(norm_data*norm_truth)

    return np.arccos(cos_angle).values*(360/(2*np.pi))


def calculate_xyz_difference(data,is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    #if is_retro == False:
    diff = np.sqrt((data['position_x'] - data['position_x%s'%post_fix])**2 + (data['position_y'] - data['position_y%s'%post_fix])**2 + (data['position_z'] - data['position_z%s'%post_fix])**2)
    return diff

def add_unbinned_width(data,key):
    if key == 'angular_res':
        residual = calculate_angular_difference(data, is_retro = False)
        width = np.percentile(residual,50)
        width_error = add_50th_error(residual, laplace = False)
    elif key == 'XYZ':
        residual = calculate_xyz_difference(data,is_retro = False)
        width = np.percentile(residual,50)
        width_error = add_50th_error(residual, laplace = False)
    elif key == 'energy':
        residual = ((10**(data[key + '_pred'])- 10**(data[key]))/10**(data[key]))*100
        width = CalculateWidth(residual)
        width_error = CalculateWidthError(residual)
    else:
        residual = data[key + '_pred'] - data[key]
        if key == 'azimuth':
            residual[residual>= 180] = 360 - residual[residual>= 180]
            residual[residual<= -180] = -(residual[residual<= -180] + 360)

        width = CalculateWidth(residual)
        width_error = CalculateWidthError(residual)

    return width, width_error



def ExtractStatistics(data_raw,keys, key_bins, is_retro):
    data_raw = data_raw.sort_values('event_no').reset_index(drop = 'True')
    pids = [14]
    interaction_types = [1.0]
    biases = {}
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    for key in keys:
        data = deepcopy(data_raw)
        print(key)
        print(data.columns)
        biases[key] = {}
        if key not in ['energy', 'angular_res', 'XYZ', 'interaction_time']:
            data[key] = data[key]*(360/(2*np.pi))
            data[key + post_fix] = data[key + post_fix]*(360/(2*np.pi))
        if key == 'angular_res':
            data[key] = calculate_angular_difference(data, is_retro)
        if key == 'XYZ':
            data[key] = calculate_xyz_difference(data,is_retro)
        for pid in pids:
            biases[key][str(pid)] = {}
            data_pid_indexed = data.loc[abs(data['pid']) == pid,:].reset_index(drop = True)
            for interaction_type in interaction_types:
                biases[key][str(pid)][str(interaction_type)] = {'mean':         [],
                                                                '16th':         [],
                                                                '50th':         [],
                                                                '84th':         [],
                                                                'count':        [],
                                                                'width':        [],
                                                                'width_error':  [],
                                                                'predictions' : [],
                                                                'bias': []}
                data_interaction_indexed = data_pid_indexed.loc[data_pid_indexed['interaction_type'] == interaction_type,:]
                if len(data_interaction_indexed) > 0:
                    if key not in ['angular_res', 'XYZ']:
                        biases[key][str(pid)][str(interaction_type)]['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
                    if key == 'angular_res':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = data_interaction_indexed['angular_res']
                    if key == 'energy':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
                    if key == 'zenith' or key == 'interaction_time':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
                bins = key_bins['energy']

                for i in range(1,(len(bins))):
                    bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
                    data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
                    
                    if key == 'energy':
                        bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/10**(data_interaction_indexed_sliced[key]))*100
                        bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
                    if key in ['zenith', 'azimuth', 'interaction_time']:
                        bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
                        if key == 'azimuth':
                            bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                            bias_tmp[bias_tmp<= -180] = -(bias_tmp[bias_tmp<= -180] + 360)
                    if key in ['angular_res', 'XYZ']:
                        bias_tmp = data_interaction_indexed_sliced[key]
                    if len(data_interaction_indexed_sliced)>0:
                        biases[key][str(pid)][str(interaction_type)]['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                        
                        #biases[key][str(pid)][str(interaction_type)]['count'].append(len(bias_tmp))
                        #biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                        #biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                        if key == 'energy':
                            biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp_percent))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp_percent))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp_percent,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp_percent,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp_percent,84))
                        elif key == 'angular_res':
                            biases[key][str(pid)][str(interaction_type)]['width'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
                        elif key == 'XYZ':
                            biases[key][str(pid)][str(interaction_type)]['width'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
                        else:
                            biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
            unbinned_width, unbinned_width_error = add_unbinned_width(data_interaction_indexed,key)
            biases[key][str(pid)][str(interaction_type)]['unbinned_width'] = unbinned_width
            biases[key][str(pid)][str(interaction_type)]['unbinned_width_error'] = unbinned_width_error
        biases[key]['cascade'] = {}
        biases[key]['cascade']                       = {'mean':         [],
                                                        '16th':         [],
                                                        '50th':         [],
                                                        '84th':         [],
                                                        'count':        [],
                                                        'width':        [],
                                                        'width_error':  [],
                                                        'predictions': []}
        data_interaction_indexed = data.loc[((data['pid'] != 14.0) | (data['interaction_type'] != 1.0)) ,:]
        if len(data_interaction_indexed) > 0:
            if key not in ['angular_res', 'XYZ']: 
                biases[key]['cascade']['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
            if key in ['angular_res', 'XYZ']:
                biases[key]['cascade']['bias'] = data_interaction_indexed[key]
            if key == 'energy':
                biases[key]['cascade']['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
            if key not in ['angular_res', 'XYZ']:
                biases[key]['cascade']['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
        bins = key_bins['energy']
        for i in range(1,(len(bins))):
            bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
            data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
            
            if key == 'energy':
                bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/(10**(data_interaction_indexed_sliced[key])))*100
                bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
            if key not in ['angular_res', 'XYZ']:
                bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
            else:
                bias_tmp = data_interaction_indexed_sliced[key]
            if key == 'azimuth':
                bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                bias_tmp[bias_tmp<= -180] = (bias_tmp[bias_tmp<= -180] + 360)
                if np.max(bias_tmp) > 180:
                    print(np.max(bias_tmp))
            if len(data_interaction_indexed_sliced)>0:
                biases[key]['cascade']['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                biases[key]['cascade']['count'].append(len(bias_tmp))
                if key == 'energy':
                    biases[key]['cascade']['width'].append(CalculateWidth(bias_tmp_percent))
                    biases[key]['cascade']['width_error'].append(CalculateWidthError(bias_tmp_percent))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp_percent,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp_percent,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp_percent,84))
                elif key == 'angular_res':
                    biases[key]['cascade']['width'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp,84))
                elif key == 'XYZ':
                    biases[key]['cascade']['width'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp,84))
                else:
                    biases[key]['cascade']['width'].append(CalculateWidth(bias_tmp))
                    biases[key]['cascade']['width_error'].append(CalculateWidthError(bias_tmp))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp,84))
        unbinned_width, unbinned_width_error = add_unbinned_width(data_interaction_indexed,key)
        biases[key]['cascade']['unbinned_width'] = unbinned_width
        biases[key]['cascade']['unbinned_width_error'] = unbinned_width_error
    return biases

def get_angular_res():
    data_zenith = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_zenith/results.csv').sort_values('event_no').reset_index(drop = True)
    data_azimuth = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_azimuth/results.csv').sort_values('event_no').reset_index(drop = True)
    data_zenith['azimuth'] = data_azimuth['azimuth']
    data_zenith['azimuth_pred'] = data_azimuth['azimuth_pred']
    data_merged_pulses_zenith = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/merged_pulses/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_zenith_merged_pulses/results_has_merged_pulse_labels.csv').sort_values('event_no').reset_index(drop = True)
    data_merged_pulses_azimuth = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/merged_pulses/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_azimuth_merged_pulses/results_has_merged_pulse_labels.csv').sort_values('event_no').reset_index(drop = True)
    data_merged_pulses_zenith['azimuth'] = data_merged_pulses_azimuth['azimuth']
    data_merged_pulses_zenith['azimuth_pred'] = data_merged_pulses_azimuth['azimuth_pred']
    return data_zenith, data_merged_pulses_zenith

def add_xyz(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select event_no, position_x, position_y, position_z from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    for key in ['position_x', 'position_y', 'position_z']:
        data[key] = truth[key]
    return data



def reshape_data(data, data_merged_pulses, database):
    print(data.columns)
    data = add_xyz(data, database)
    #data = data.drop('XYZ', axis=1, inplace=True)
    data_merged_pulses = add_xyz(data_merged_pulses, database)
    #data_merged_pulses = data_merged_pulses.drop('XYZ', axis=1, inplace=True)
    return data, data_merged_pulses

    
def CalculateStatistics(targets, key_bins, database):
    is_first = True
    for target in targets:
        if target != 'angular_res':
            data = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_%s/results.csv'%target).sort_values('event_no').reset_index(drop = True)
            data_merged_pulses = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/merged_pulses/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_%s_merged_pulses/results_has_merged_pulse_labels.csv'%target).sort_values('event_no').reset_index(drop = True)
        else:
            data, data_merged_pulses = get_angular_res()

        data['has_merged_pulse'] = data_merged_pulses['has_merged_pulse']
        data = data.loc[data['has_merged_pulse'] == 1,:].reset_index(drop = True)
        data_merged_pulses = data_merged_pulses.loc[data_merged_pulses['has_merged_pulse'] == 1, :].reset_index(drop = True)
            
        data_merged_pulses = add_pid_and_interaction(data_merged_pulses, database)
        data_merged_pulses = transform_energy(data_merged_pulses,database)
        data_merged_pulses = remove_muons(data_merged_pulses)
        data = add_pid_and_interaction(data, database)
        data = transform_energy(data,database)
        data = remove_muons(data)

        if target == 'XYZ':
            data, data_merged_pulses = reshape_data(data, data_merged_pulses, database)
        if is_first:
            biases = {'dynedge': ExtractStatistics(data,[target], key_bins, is_retro = False),
                    'dynedge_merged_pulses': ExtractStatistics(data_merged_pulses,[target], key_bins, is_retro = False)}
            is_first = False
        else:
            biases['dynedge'].update(ExtractStatistics(data,[target], key_bins, is_retro = False))
            biases['dynedge_merged_pulses'].update(ExtractStatistics(data_merged_pulses,[target], key_bins, is_retro = False))
    return biases
    

def CalculateRelativeImprovementError(relimp, w1, w1_sigma, w2, w2_sigma):
    w1 = np.array(w1)
    w2 = np.array(w2)
    w1_sigma = np.array(w1_sigma)
    w2_sigma = np.array(w2_sigma)
    sigma = np.sqrt(((1/w2)*w1_sigma)**2  + ((w1/w2**2)*w2_sigma)**2)
    return sigma

def add_energy(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select event_no, energy from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    data['energy'] = truth['energy']
    return data

def add_pid_and_interaction(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select event_no, pid, interaction_type from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    data['pid'] = truth['pid']
    data['interaction_type'] = truth['interaction_type']
    return data


def transform_energy(data,database):
    try:
        data['energy'] = np.log10(data['energy'])
    except:
        data = add_energy(data, database)
        data['energy'] = np.log10(data['energy'])
    try:
        data['energy_pred'] = np.log10(data['energy_pred'])
    except:
        pass
    return data

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data



def get_axis(key, fig, gs):
    if key == 'energy':
        ax1 = fig.add_subplot(gs[0:6, 0:6])
        #ax2 = fig.add_subplot(gs[4:6, 0:6])
    if key == 'zenith':
        ax1 = fig.add_subplot(gs[6:12, 0:6])
        #ax2 = fig.add_subplot(gs[10:12, 0:6])
    if key == 'angular_res':
        ax1 = fig.add_subplot(gs[0:6, 6:12])
        #ax2 = fig.add_subplot(gs[4:6, 6:12])
    if key == 'XYZ':
        ax1 = fig.add_subplot(gs[6:12, 6:12])
        #ax2 = fig.add_subplot(gs[10:12, 6:12])
    return ax1


def make_combined_resolution_plot(targets, plot_config, include_retro, track_cascade = False):
    #data = pd.read_csv(plot_config['data'])
    #data_merged_pulses = pd.read_csv(plot_config['data_merged_pulses'])
    width = 2*3.176
    height = 2*3.176

    key_limits = plot_config['width']
    key_bins = plot_config['key_bins']
    if exists('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/merged_pulses_statistics.pickle'):
        with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/merged_pulses_statistics.pickle', 'rb') as handle:
            biases = pickle.load(handle)
    else:
        biases = CalculateStatistics(targets, key_bins,  plot_config['database'])
        with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/merged_pulses_statistics.pickle', 'wb') as handle:
            pickle.dump(biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fig = plt.figure(constrained_layout = True)
    fig.set_size_inches(width, height)
    gs = fig.add_gridspec(6*2, 6*2)
    for key in targets:
        print(key)
        ax1 = get_axis(key, fig, gs)
        plot_data_track = biases['dynedge'][key][str(14)][str(1.0)]
        plot_data_cascade = biases['dynedge'][key]['cascade']
        plot_data_track['width'] = np.array(plot_data_track['width'])
        plot_data_track['width_error'] = np.array(plot_data_track['width_error'])
        plot_data_cascade['width'] = np.array(plot_data_cascade['width'])
        plot_data_cascade['width_error'] = np.array(plot_data_cascade['width_error'])

        plot_data_track_merged = biases['dynedge_merged_pulses'][key][str(14)][str(1.0)]
        plot_data_cascade_merged  = biases['dynedge_merged_pulses'][key]['cascade']
        plot_data_track_merged['width'] = np.array(plot_data_track_merged['width'])
        plot_data_track_merged['width_error'] = np.array(plot_data_track_merged['width_error'])
        plot_data_cascade_merged['width'] = np.array(plot_data_cascade_merged['width'])
        plot_data_cascade_merged['width_error'] = np.array(plot_data_cascade_merged['width_error'])

    
        print('--------')
        print('--TRACK VARIATION--')
        ri = (1 - plot_data_track_merged['unbinned_width']/plot_data_track['unbinned_width'])*100
        ri_e = CalculateRelativeImprovementError('', plot_data_track_merged['unbinned_width'],plot_data_track_merged['unbinned_width_error'],plot_data_track['unbinned_width'],plot_data_track['unbinned_width_error'])*100
        print('%s +- %s'%(ri, ri_e))
        print('--------')
        print('---cascade----')
        print('%s +- %s'%((1 - plot_data_cascade_merged['unbinned_width']/plot_data_cascade['unbinned_width'])*100, CalculateRelativeImprovementError('', plot_data_cascade_merged['unbinned_width'],plot_data_cascade_merged['unbinned_width_error'],plot_data_cascade['unbinned_width'],plot_data_cascade['unbinned_width_error'])*100))
        print('--------')


        if len(plot_data_track['mean']) != 0:
            ax1.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'black', lw = 1)
            ax1.plot(plot_data_track['mean'],(1 - np.array(plot_data_track_merged['width'])/np.array(plot_data_track['width']))*100, color = 'black', lw = 0.5, alpha = 1,linestyle='solid')
            l5 = ax1.fill_between(plot_data_track['mean'], (1 - np.array(plot_data_track_merged['width'])/np.array(plot_data_track['width']))*100 - CalculateRelativeImprovementError(1 - np.array(plot_data_track_merged['width'])/np.array(plot_data_track['width']), plot_data_track_merged['width'], plot_data_track_merged['width_error'], plot_data_track['width'], plot_data_track['width_error'])*100, (1 - np.array(plot_data_track_merged['width'])/np.array(plot_data_track['width']))*100 + CalculateRelativeImprovementError(1 - np.array(plot_data_track_merged['width'])/np.array(plot_data_track['width']), plot_data_track_merged['width'], plot_data_track_merged['width_error'], plot_data_track['width'], plot_data_track['width_error'])*100, label = 'Track', color = 'tab:olive')
            
            ax1.plot(plot_data_cascade['mean'],(1 - np.array(plot_data_cascade_merged['width'])/np.array(plot_data_cascade['width']))*100 , color = 'black', alpha = 0.5, lw = 0.5,linestyle='solid')
            l6 = ax1.fill_between(plot_data_cascade['mean'], (1 - np.array(plot_data_cascade_merged['width'])/np.array(plot_data_cascade['width']))*100 -  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade_merged['width'])/np.array(plot_data_cascade['width']), plot_data_cascade_merged['width'], plot_data_cascade_merged['width_error'], plot_data_cascade['width'], plot_data_cascade['width_error'])*100, (1 - np.array(plot_data_cascade_merged['width'])/np.array(plot_data_cascade['width']))*100 +  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade_merged['width'])/np.array(plot_data_cascade['width']), plot_data_cascade_merged['width'], plot_data_cascade_merged['width_error'], plot_data_cascade['width'], plot_data_cascade['width_error'])*100,  label = 'Cascade', color = 'tab:green')
            #ax1.legend(frameon=False, fontsize = 6)
        
    
            ax1.tick_params(axis='x', labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)
            ax1.set_xlim(key_limits[key]['x'])
            ax1.set_ylim([-15,15])

            if key == 'energy':
                unit_tag = '(%)'
                ax1.legend([l5,l6], ['Track', 'Cascade'], ncol = 1, fontsize = 8, frameon = False)
            else:
                unit_tag = '(deg.)'
            if key == 'angular_res':
                key = 'direction'
                #plt.tick_params(right=False,labelright=False)
                #ax1.yaxis.set_label_position("right")
                #ax1.yaxis.tick_right()
                labels = [item.get_text() for item in ax1.get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_yticklabels(empty_string_labels)  
                #ax1.set_ylim([-60,30])
            if key == 'XYZ':
                key = 'vertex'
                unit_tag = '(m)'
                #plt.tick_params(right=False,labelright=False)

                labels = [item.get_text() for item in ax1.get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_yticklabels(empty_string_labels)  
                #ax1.yaxis.set_label_position("right")
                #ax1.yaxis.tick_right()
            if key not in ['vertex', 'direction']:
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('Rel. Impro. (%)', size = 12)
            if key == 'energy' or key == 'direction':
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_xticklabels(empty_string_labels)  
            else:
                ax1.set_xlabel('Energy' + ' (log10 GeV)', size = 10)
            
            plt.text(1, -10, key.title(), size = 12)

    fig.suptitle('Resolution Performance', size = 12)
    fig.savefig('plots/merged_pulses_relative_improvement.pdf',bbox_inches="tight")

    return fig


width_limits = {'energy':{'x':[0,3], 'y':[-0.5,1.5]},
                'zenith': {'x':[0,3], 'y':[-100,100]},
                'azimuth': {'x':[0,3], 'y':[-100,100]},
                'XYZ': {'x':[0,3], 'y':[-100,100]},
                'angular_res': {'x':[0,3], 'y':[-100,100]}}

key_bins = { 'energy': np.arange(0, 3.25, 0.25),
                'zenith': np.arange(0, 180, 10),
                'azimuth': np.arange(0, 2*180, 20) }


plot_config = {'database': '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db',
                'width': width_limits,
                'key_bins': key_bins}
if __name__  == '__main__':
    make_combined_resolution_plot(targets =[ 'XYZ','zenith', 'energy','angular_res'], plot_config = plot_config, include_retro= True, track_cascade = True)
