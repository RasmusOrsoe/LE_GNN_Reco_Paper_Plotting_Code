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

indir = '/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle'

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
    #N = len(diff)
    #x_50 = abs(diff-np.percentile(diff,50,interpolation='nearest')).argmin() #int(0.16*N)
    #if len(diff)>0:
    #    if laplace == True:
    #        error = np.sqrt((1/laplacian_pdf(x_50, diff)**2)*(0.5*(1-0.5)/N))
    #    else:
    #        error = np.sqrt((1/gaussian_pdf(x_50, diff)**2)*(0.5*(1-0.5)/N))
    #else:
    #    error = np.nan
    #return error
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
    #N = len(diff)
    #x_16 = abs(diff-np.percentile(diff,16,interpolation='nearest')).argmin() #int(0.16*N)
    #x_84 = abs(diff-np.percentile(diff,84,interpolation='nearest')).argmin() #int(0.84*N)
    #fe_16 = sum(diff <= diff[x_16])/N
    #fe_84 = sum(diff <= diff[x_84])/N
    #n_16 = sum(diff <= diff[x_16])
    #n_84 = sum(diff <= diff[x_84]) 
    #error_width = np.sqrt((0.16*(1-0.16)/N)*(1/fe_16**2 + 1/fe_84**2))*(1/2)
    #n,bins,_ = plt.hist(diff, bins = 30)
    #plt.close()
    #if len(diff)>0:
    #    error_width = np.sqrt((1/gaussian_pdf(x_84, diff)**2)*(0.84*(1-0.84)/N) + (1/gaussian_pdf(x_16, diff)**2)*(0.16*(1-0.16)/N))*(1/2)
    #else:
    #    error_width = np.nan
    #return error_width
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

def ExtractStatistics(data_raw,keys, key_bins, is_retro):
    data_raw = data_raw.sort_values('event_no').reset_index(drop = 'True')
    pids = pd.unique(abs(data_raw['pid']))
    interaction_types = data_raw['interaction_type'].unique()
    biases = {}
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    for key in keys:
        data = deepcopy(data_raw)
        print(key)
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
        
        biases[key]['all_pid'] = {}
        for interaction_type in interaction_types:
            biases[key]['all_pid'][str(interaction_type)] = {'mean':         [],
                                                            '16th':         [],
                                                            '50th':         [],
                                                            '84th':         [],
                                                            'count':        [],
                                                            'width':        [],
                                                            'width_error':  [],
                                                            'predictions': []}
            data_interaction_indexed = data.loc[data['interaction_type'] == interaction_type,:]
            if len(data_interaction_indexed) > 0:
                if key not in ['angular_res', 'XYZ']: 
                    biases[key]['all_pid'][str(interaction_type)]['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
                if key in ['angular_res', 'XYZ']:
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = data_interaction_indexed[key]
                elif key == 'energy':
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
                else:
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
            bins = key_bins['energy']
            for i in range(1,(len(bins))):
                bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
                data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
                
                if key == 'energy':
                    bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/(10**(data_interaction_indexed_sliced[key])))*100
                    bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
                elif key not in ['angular_res', 'XYZ']:
                    bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
                else:
                    bias_tmp = data_interaction_indexed_sliced[key]

                if key == 'azimuth':
                    bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                    bias_tmp[bias_tmp<= -180] = (bias_tmp[bias_tmp<= -180] + 360)
                    if np.max(bias_tmp) > 180:
                        print(np.max(bias_tmp))
                if len(data_interaction_indexed_sliced)>0:
                    biases[key]['all_pid'][str(interaction_type)]['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                    biases[key]['all_pid'][str(interaction_type)]['count'].append(len(bias_tmp))
                    if key == 'energy':
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp_percent))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp_percent))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp_percent,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp_percent,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp_percent,84))
                    elif key == 'angular_res':
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
                    elif key == 'XYZ':
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(add_50th_error(bias_tmp, laplace = False))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
                    else:
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
        
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
    return biases

    
def CalculateStatistics(data,keys, key_bins,include_retro = False):
    biases = {'dynedge': ExtractStatistics(data, keys, key_bins, is_retro = False)}
    if include_retro:
        biases['retro'] = ExtractStatistics(data, keys, key_bins, is_retro = True)
    return biases
    

def CalculateRelativeImprovementError(relimp, w1, w1_sigma, w2, w2_sigma):
    w1 = np.array(w1)
    w2 = np.array(w2)
    w1_sigma = np.array(w1_sigma)
    w2_sigma = np.array(w2_sigma)
    #sigma = np.sqrt((np.array(w1_sigma)/np.array(w1))**2 + (np.array(w2_sigma)/np.array(w2))**2)
    sigma = np.sqrt(((1/w2)*w1_sigma)**2  + ((w1/w2**2)*w2_sigma)**2)
    return sigma

def MakeSummaryWidthPlot(key_limits, biases, include_retro, track_cascade = False):
    key_limits = key_limits['width']
    if track_cascade == False:
        for key in biases['dynedge'].keys():
            fig = plt.figure()
            plt.hist(biases['dynedge'][key]['all_pid']['1.0']['predictions'], histtype = 'step', label = 'dynedge')
            plt.hist(biases['retro'][key]['all_pid']['1.0']['predictions'], histtype = 'step', label = 'retro')
            plt.legend()
            fig.savefig('test_hist.png')
            fig = plt.figure()
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
            pid_count = 0
            pid = 'all_pid'
            interaction_type = str(1.0)
            plot_data = biases['dynedge'][key][pid][interaction_type]
            if include_retro:
                plot_data_retro = biases['retro'][key][pid][interaction_type]
            if len(plot_data['mean']) != 0:
                ax3 = ax1.twinx()
                ax3.bar(x = (plot_data['mean']), height = plot_data['count'], 
                        alpha = 0.3, 
                        color = 'grey',
                        align = 'center',
                        width = 0.25)
                ax1.errorbar(plot_data['mean'],plot_data['width'],plot_data['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
                if include_retro:
                    ax1.errorbar(plot_data_retro['mean'],plot_data_retro['width'],plot_data_retro['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, label = 'RetroReco')
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(plot_data['mean'], np.repeat(0, len(plot_data['mean'])), color = 'black', lw = 2)
                #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
                if include_retro:
                    ax2.errorbar(plot_data['mean'],1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']),CalculateRelativeImprovementError(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']), plot_data['width'], plot_data['width_error'], plot_data_retro['width'], plot_data_retro['width_error']),marker='o', capsize = 10,markeredgecolor='black')

                #plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis='x', labelsize=10)
                ax1.tick_params(axis='y', labelsize=10)
                ax1.set_xlim(key_limits[key]['x'])
                ax2.set_xlim(key_limits[key]['x'])
                ax2.set_ylim([-0.1,0.55])
                ax1.legend()
                if key == 'energy':
                    unit_tag = '[%]'
                else:
                    unit_tag = '[deg.]'
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key, unit_tag), size = 15)
                ax2.set_xlabel('$Energy_{log10}$ [GeV]', size = 15)
                ax2.set_ylabel('Rel. Impro.', size = 15)  
                
                fig.suptitle('%s CC'%key, size = 20)
                fig.savefig('performance_%s.png'%key)
    else:
        for key in biases['dynedge'].keys():
            fig = plt.figure()
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
            pid_count = 0
            pid = 'track'
            interaction_type = str(1.0)
            plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
            plot_data_cascade = biases['dynedge'][key]['cascade']
            if include_retro:
                plot_data_retro_track = biases['retro'][key][str(14.0)][str(1.0)]
                plot_data_retro_cascade = biases['retro'][key]['cascade']
            if len(plot_data_track['mean']) != 0:
                ax3 = ax1.twinx()
                #ax3.bar(x = plot_data_track['mean'], height = plot_data_track['count'], 
                #        alpha = 0.3, 
                #        color = 'grey',
                #        align = 'center',
                #        width = 0.25)
                ax1.errorbar(plot_data_track['mean'],plot_data_track['width'],plot_data_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'blue', label = 'GCN-all Track')
                ax1.errorbar(plot_data_cascade['mean'],plot_data_cascade['width'],plot_data_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkblue', label = 'GCN-all Cascade')
                if include_retro:
                    ax1.errorbar(plot_data_retro_track['mean'],plot_data_retro_track['width'],plot_data_retro_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'orange', label = 'RetroReco Track')
                    ax1.errorbar(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],plot_data_retro_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkorange' , label = 'RetroReco Cascade')
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'black', lw = 2)
                #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
                if include_retro:
                    ax2.errorbar(plot_data_track['mean'],1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'limegreen', label = 'track',linestyle='dotted')
                    ax2.errorbar(plot_data_cascade['mean'],1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'springgreen', label = 'cascade',linestyle='solid')
                    ax2.legend()
                #plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis='x', labelsize=10)
                ax1.tick_params(axis='y', labelsize=10)
                ax1.set_xlim(key_limits[key]['x'])
                ax2.set_xlim(key_limits[key]['x'])
                ax2.set_ylim([-0.40,0.40])
                ax1.legend()
                if key == 'energy':
                    unit_tag = '[%]'
                else:
                    unit_tag = '[deg.]'
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key, unit_tag), size = 15)
                ax2.set_xlabel('$Energy_{log10}$ [GeV]', size = 15)
                ax2.set_ylabel('Rel. Impro.', size = 15)  
                
                fig.suptitle('%s Performance'%key, size = 20)
                fig.savefig('performance_track_cascade_outline_presentation_%s.png'%key)

    return fig

def transform_energy(data):
    data['energy'] = np.log10(data['energy'])
    data['energy_pred'] = np.log10(data['energy_pred'])
    data['energy_retro'] = np.log10(data['energy_retro'])
    return data

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data

def make_resolution_plots(targets, plot_config, include_retro, track_cascade = False):
    data = pd.read_csv(plot_config['data'])
    data = transform_energy(data)
    data = remove_muons(data)

    width = 3.176
    height = 3.176

    capsize = 5
    markersize = 4
    key_limits = plot_config['width']
    key_bins = plot_config['key_bins']
    if exists('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle'):
        with open('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle', 'rb') as handle:
            biases = pickle.load(handle)
    else:
        biases = CalculateStatistics(data,targets, key_bins,include_retro = True)
        with open('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle', 'wb') as handle:
            pickle.dump(biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for key in biases['dynedge'].keys():
        fig = plt.figure(constrained_layout = True)
        fig.set_size_inches(width, height)
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
        ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
        pid_count = 0
        pid = 'track'
        mode = 'area'
        interaction_type = str(1.0)
        plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
        plot_data_cascade = biases['dynedge'][key]['cascade']
        plot_data_track['width'] = np.array(plot_data_track['width'])
        plot_data_track['width_error'] = np.array(plot_data_track['width_error'])
        plot_data_cascade['width'] = np.array(plot_data_cascade['width'])
        plot_data_cascade['width_error'] = np.array(plot_data_cascade['width_error'])
        if include_retro:
            plot_data_retro_track = biases['retro'][key][str(14.0)][str(1.0)]
            plot_data_retro_cascade = biases['retro'][key]['cascade']
            plot_data_retro_track['width'] = np.array(plot_data_retro_track['width'])
            plot_data_retro_track['width_error'] = np.array(plot_data_retro_track['width_error'])
            plot_data_retro_cascade['width'] = np.array(plot_data_retro_cascade['width'])
            plot_data_retro_cascade['width_error'] = np.array(plot_data_retro_cascade['width_error'])
        if len(plot_data_track['mean']) != 0:
            ax3 = ax1.twinx()
            #ax3.bar(x = plot_data_track['mean'], height = plot_data_track['count'], 
            #        alpha = 0.3, 
            #        color = 'grey',
            #        align = 'center',
            #        width = 0.25)
            if mode == 'errorbar':
                ax1.errorbar(plot_data_track['mean'],plot_data_track['width'],plot_data_track['width_error'],linestyle='dotted',fmt = 'o',capsize = capsize, markersize=markersize, color = 'tab:blue', alpha = 1 ,label = 'dynedge Track')
                ax1.errorbar(plot_data_cascade['mean'],plot_data_cascade['width'],plot_data_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = capsize, markersize=markersize, color = 'tab:blue', alpha = 0.5,  label = 'dynedge Cascade')
            if mode == 'area':
                ax1.plot(plot_data_track['mean'],plot_data_track['width'],linestyle='solid', lw = 0.5, color = 'black', alpha = 1)
                #ax1.plot( plot_data_track['mean'], plot_data_track['width'] - plot_data_track['width_error'],linestyle='solid', color = 'tab:blue', alpha = 1, lw = 0.25,label = 'dynedge Track')
                #ax1.plot( plot_data_track['mean'], plot_data_track['width'] + plot_data_track['width_error'],linestyle='solid', color = 'tab:blue', alpha = 1, lw = 0.25)
                ax1.fill_between(plot_data_track['mean'],plot_data_track['width'] - plot_data_track['width_error'], plot_data_track['width'] + plot_data_track['width_error'],color = 'tab:blue', alpha = 0.8 ,label = 'dynedge Track')
                
                ax1.plot(plot_data_cascade['mean'],plot_data_cascade['width'],linestyle='dashed', color = 'tab:blue', lw = 0.5, alpha = 1)
                #ax1.plot(plot_data_cascade['mean'], plot_data_cascade['width']- plot_data_cascade['width_error'], linestyle='solid',color = 'tab:blue', alpha = 0.5, lw = 1)
                #ax1.plot(plot_data_cascade['mean'], plot_data_cascade['width']+ plot_data_cascade['width_error'],linestyle='solid', color = 'tab:blue', alpha = 0.5, lw = 1)
                ax1.fill_between(plot_data_cascade['mean'], plot_data_cascade['width']- plot_data_cascade['width_error'], plot_data_cascade['width']+ plot_data_cascade['width_error'], color = 'tab:blue', alpha = 0.3, label = 'dynedge Cascade' )
            if include_retro:
                if mode == 'errorbar':
                    ax1.errorbar(plot_data_retro_track['mean'],plot_data_retro_track['width'],plot_data_retro_track['width_error'],linestyle='dotted',fmt = 'o',capsize = capsize, markersize=markersize, color = 'tab:orange', alpha = 1 ,label = 'Retro Track')
                    ax1.errorbar(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],plot_data_retro_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = capsize, markersize=markersize, color = 'tab:orange' , alpha = 1 ,label = 'Retro Cascade')
                if mode == 'area':
                    ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'],linestyle='solid', color = 'black', lw = 0.5, alpha = 1)
                    #ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'] - plot_data_retro_track['width_error'],linestyle='dotted', color = 'tab:orange', alpha = 1)
                    #ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'] + plot_data_retro_track['width_error'],linestyle='dotted', color = 'tab:orange', alpha = 1)
                    ax1.fill_between(plot_data_retro_track['mean'],plot_data_retro_track['width'] - plot_data_retro_track['width_error'],plot_data_retro_track['width'] + plot_data_retro_track['width_error'], color = 'tab:orange', alpha = 0.8,label = 'Retro Track')
                    
                    ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],linestyle='dashed', color = 'tab:orange', lw = 0.5 , alpha = 1 )
                    #ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width']-plot_data_retro_cascade['width_error'],linestyle='solid', color = 'tab:orange' , alpha = 0.5)
                    #ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width']+plot_data_retro_cascade['width_error'],linestyle='solid', color = 'tab:orange' , alpha = 0.5)
                    ax1.fill_between(plot_data_retro_cascade['mean'], plot_data_retro_cascade['width']-plot_data_retro_cascade['width_error'], plot_data_retro_cascade['width'] + plot_data_retro_cascade['width_error'], color = 'tab:orange' , alpha = 0.3, label = 'Retro Cascade')
            labels = [item.get_text() for item in ax1.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax1.set_xticklabels(empty_string_labels)
            #ax1.grid()
            ax2.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'black', lw = 1)
            #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
            if include_retro:
                if mode == 'errorbar':
                    ax2.errorbar(plot_data_track['mean'],(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100,CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error'])*100,marker='o', capsize = capsize,markeredgecolor='black',markersize=markersize, color = 'tab:blue', alpha = 1, label = 'track',linestyle='dotted')
                    ax2.errorbar(plot_data_cascade['mean'],(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100,CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error'])*100,marker='o', capsize = capsize,markeredgecolor='black', markersize=markersize, color = 'tab:orange', alpha = 0.5, label = 'cascade',linestyle='solid')
                    ax2.legend(fontsize = 6)
                if mode == 'area':
                    ax2.plot(plot_data_track['mean'],(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100, color = 'black', lw = 0.5, alpha = 1,linestyle='solid')
                    ax2.fill_between(plot_data_track['mean'], (1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100 - CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error'])*100, (1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100 + CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error'])*100, label = 'Track', color = 'tab:olive')
                    
                    ax2.plot(plot_data_cascade['mean'],(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 , color = 'black', alpha = 0.5, lw = 0.5,linestyle='solid')
                    ax2.fill_between(plot_data_cascade['mean'], (1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 -  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error'])*100, (1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 +  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error'])*100,  label = 'Cascade', color = 'tab:green')
                    ax2.legend(frameon=False, fontsize = 6)
                
            #plt.title('$\\nu_{v,u,e}$', size = 20)
            ax1.tick_params(axis='x', labelsize=6)
            ax1.tick_params(axis='y', labelsize=6)
            ax2.tick_params(axis='x', labelsize=6)
            ax2.tick_params(axis='y', labelsize=6)
            ax1.set_xlim(key_limits[key]['x'])
            ax2.set_xlim(key_limits[key]['x'])
            ax2.set_ylim([-40,40])

            ylbl = ax1.yaxis.get_label()
            ylbl_target = ax2.yaxis.get_label()
            ax1.yaxis.set_label_coords(-0.1,ylbl.get_position()[1])
            ax2.yaxis.set_label_coords(-0.1,ylbl_target.get_position()[1])
            if mode == 'area':
                leg = ax1.legend(frameon=False, fontsize = 6)
                for line in leg.get_lines():
                    line.set_linewidth(4.0)
            else:
                ax1.legend(fontsize = 6)
            if key == 'energy':
                unit_tag = '(%)'
            else:
                unit_tag = '(deg.)'
            if key == 'angular_res':
                key = 'direction'
                ax2.set_ylim([-60,30])
            if key == 'zenith':
                ax2.set_ylim([-50,40])
            if key == 'XYZ':
                key = 'vertex'
                unit_tag = '(m)'

            
            plt.tick_params(right=False,labelright=False)
            ax1.set_ylabel('%s Resolution %s'%(key.capitalize(), unit_tag), size = 8)
            ax2.set_xlabel('Energy  (log10 GeV)', size = 8)
            ax2.set_ylabel('Rel. Impro. (%)', size = 8)  


            fig.suptitle('%s Resolution'%key.capitalize(), size = 12)
            fig.savefig('performance_track_cascade_outline_presentation_%s.pdf'%key,bbox_inches="tight")

    return fig

def get_axis(key, fig, gs):
    if key == 'energy':
        ax1 = fig.add_subplot(gs[0:4, 0:6])
        ax2 = fig.add_subplot(gs[4:6, 0:6])
    if key == 'zenith':
        ax1 = fig.add_subplot(gs[6:10, 0:6])
        ax2 = fig.add_subplot(gs[10:12, 0:6])
    if key == 'angular_res':
        ax1 = fig.add_subplot(gs[0:4, 6:12])
        ax2 = fig.add_subplot(gs[4:6, 6:12])
    if key == 'XYZ':
        ax1 = fig.add_subplot(gs[6:10, 6:12])
        ax2 = fig.add_subplot(gs[10:12, 6:12])
    return ax1, ax2


def make_combined_resolution_plot(targets, plot_config, include_retro, track_cascade = False):
    data = pd.read_csv(plot_config['data'])
    data = transform_energy(data)
    data = remove_muons(data)

    width = 2*3.176
    height = 2*3.176

    key_limits = plot_config['width']
    key_bins = plot_config['key_bins']
    if exists('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle'):
        with open('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle', 'rb') as handle:
            biases = pickle.load(handle)
    else:
        biases = CalculateStatistics(data,targets, key_bins,include_retro = True)
        with open('/remote/ceph/user/o/oersoe/paper_data/paper/paper_data/plots/performance_statistics.pickle', 'wb') as handle:
            pickle.dump(biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fig = plt.figure(constrained_layout = True)
    fig.set_size_inches(width, height)
    gs = fig.add_gridspec(6*2, 6*2)
    blk_like = 'white'
    for key in targets:
        print(key)
        ax1, ax2 = get_axis(key, fig, gs)
        pid_count = 0
        pid = 'track'
        mode = 'area'
        interaction_type = str(1.0)
        plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
        plot_data_cascade = biases['dynedge'][key]['cascade']
        if include_retro:
            plot_data_retro_track = biases['retro'][key][str(14.0)][str(1.0)]
            plot_data_retro_cascade = biases['retro'][key]['cascade']

        plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
        plot_data_cascade = biases['dynedge'][key]['cascade']
        plot_data_track['width'] = np.array(plot_data_track['width'])
        plot_data_track['width_error'] = np.array(plot_data_track['width_error'])
        plot_data_cascade['width'] = np.array(plot_data_cascade['width'])
        plot_data_cascade['width_error'] = np.array(plot_data_cascade['width_error'])
        plot_data_retro_track['width'] = np.array(plot_data_retro_track['width'])
        plot_data_retro_track['width_error'] = np.array(plot_data_retro_track['width_error'])
        plot_data_retro_cascade['width'] = np.array(plot_data_retro_cascade['width'])
        plot_data_retro_cascade['width_error'] = np.array(plot_data_retro_cascade['width_error'])
        if len(plot_data_track['mean']) != 0:
            if mode == 'area':
                ax1.plot(plot_data_track['mean'],plot_data_track['width'],linestyle='solid', lw = 0.5, c='tab:blue') # , alpha = 1
                #ax1.plot( plot_data_track['mean'], plot_data_track['width'] - plot_data_track['width_error'],linestyle='solid', color = 'tab:blue', alpha = 1, lw = 0.25,label = 'dynedge Track')
                #ax1.plot( plot_data_track['mean'], plot_data_track['width'] + plot_data_track['width_error'],linestyle='solid', color = 'tab:blue', alpha = 1, lw = 0.25)
                l1 =  ax1.fill_between(plot_data_track['mean'],plot_data_track['width'] - plot_data_track['width_error'], plot_data_track['width'] + plot_data_track['width_error'],color = 'tab:blue', alpha = 0.8 ,label = 'dynedge Track')
                
                ax1.plot(plot_data_cascade['mean'],plot_data_cascade['width'],linestyle='dashed', color = 'tab:blue', lw = 0.5, alpha = 1)
                #ax1.plot(plot_data_cascade['mean'], plot_data_cascade['width']- plot_data_cascade['width_error'], linestyle='solid',color = 'tab:blue', alpha = 0.5, lw = 1)
                #ax1.plot(plot_data_cascade['mean'], plot_data_cascade['width']+ plot_data_cascade['width_error'],linestyle='solid', color = 'tab:blue', alpha = 0.5, lw = 1)
                l2 = ax1.fill_between(plot_data_cascade['mean'], plot_data_cascade['width']- plot_data_cascade['width_error'], plot_data_cascade['width']+ plot_data_cascade['width_error'], color = 'tab:blue', alpha = 0.3, label = 'dynedge Cascade' )
            if include_retro:
                if mode == 'area':
                    ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'],linestyle='solid', lw = 0.5, c='tab:orange') # , color = blk_like, , alpha = 1, c='tab:orange'
                    #ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'] - plot_data_retro_track['width_error'],linestyle='dotted', color = 'tab:orange', alpha = 1)
                    #ax1.plot(plot_data_retro_track['mean'],plot_data_retro_track['width'] + plot_data_retro_track['width_error'],linestyle='dotted', color = 'tab:orange', alpha = 1)
                    l3 = ax1.fill_between(plot_data_retro_track['mean'],plot_data_retro_track['width'] - plot_data_retro_track['width_error'],plot_data_retro_track['width'] + plot_data_retro_track['width_error'], color = 'tab:orange', alpha = 0.8,label = 'Retro Track')
                    
                    ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],linestyle='dashed', color = 'tab:orange', lw = 0.5 , alpha = 1 )
                    #ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width']-plot_data_retro_cascade['width_error'],linestyle='solid', color = 'tab:orange' , alpha = 0.5)
                    #ax1.plot(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width']+plot_data_retro_cascade['width_error'],linestyle='solid', color = 'tab:orange' , alpha = 0.5)
                    l4 = ax1.fill_between(plot_data_retro_cascade['mean'], plot_data_retro_cascade['width']-plot_data_retro_cascade['width_error'], plot_data_retro_cascade['width'] + plot_data_retro_cascade['width_error'], color = 'tab:orange' , alpha = 0.3, label = 'Retro Cascade')
            labels = [item.get_text() for item in ax1.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax1.set_xticklabels(empty_string_labels)
            #ax1.grid()
            ax2.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'tab:orange', lw = 1, alpha=0.8, zorder=0)
            #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
            if include_retro:
                if mode == 'area':
                    ax2.plot(plot_data_track['mean'],(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100, lw = 0.5, alpha = 1,linestyle='solid', c='tab:blue') # , color = blk_like
                    l5 = ax2.fill_between(plot_data_track['mean'], (1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100 - CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error'])*100, (1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']))*100 + CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error'])*100, color = 'tab:blue', alpha=0.7) # , label = 'Track'
                    
                    ax2.plot(plot_data_cascade['mean'],(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 , lw = 0.5, c='tab:blue', linestyle='dashed') # , color = blk_like, alpha = 0.5,linestyle='solid'
                    l6 = ax2.fill_between(plot_data_cascade['mean'], (1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 -  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error'])*100, (1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']))*100 +  CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error'])*100, color = 'tab:blue', alpha=0.3) # ,  label = 'Cascade'
                    #ax2.legend(frameon=False, fontsize = 6)
                
            
            
            ax1.tick_params(axis='x', labelsize=6)
            ax1.tick_params(axis='y', labelsize=6)
            ax2.tick_params(axis='x', labelsize=6)
            ax2.tick_params(axis='y', labelsize=6)
            ax1.set_xlim(key_limits[key]['x'])
            ax2.set_xlim(key_limits[key]['x'])
            ax2.set_ylim([-40,40])

            if key == 'energy':
                unit_tag = '(%)'
                ax1.legend([l1,l2,l3,l4], ['dynedge $\\mathcal{T}$','dynedge $\\mathcal{C}$', 'Retro $\\mathcal{T}$','Retro $\\mathcal{C}$'], ncol = 1, fontsize = 8, frameon = False)
                # ax1.legend([l1,l2,l3,l4,l5,l6], ['dynedge $\\mathcal{T}$','dynedge $\\mathcal{C}$', 'Retro $\\mathcal{T}$','Retro $\\mathcal{C}$', '$\\mathcal{T}$', '$\\mathcal{C}$'], ncol = 1, fontsize = 8, frameon = False)
            else:
                unit_tag = '(deg.)'
            if key == 'angular_res':
                key = 'direction'
                #plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key.capitalize(), unit_tag), size = 8)
                ax2.set_ylabel('Rel. Impro. (%)', size = 8)
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
                ylbl = ax1.yaxis.get_label()
                ylbl_target = ax2.yaxis.get_label()
                ax1.yaxis.set_label_coords(+1.12,ylbl.get_position()[1])
                ax2.yaxis.set_label_coords(+1.12,ylbl_target.get_position()[1])
                ax2.set_ylim([-60,30])
            if key == 'XYZ':
                key = 'vertex'
                unit_tag = '(m)'
                #plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key.capitalize(), unit_tag), size = 8)
                ax2.set_ylabel('Rel. Impro. (%)', size = 8)
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
                ylbl = ax1.yaxis.get_label()
                ylbl_target = ax2.yaxis.get_label()
                ax1.yaxis.set_label_coords(+1.12,ylbl.get_position()[1])
                ax2.yaxis.set_label_coords(+1.12,ylbl_target.get_position()[1])
            if key not in ['vertex', 'direction']:
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key.capitalize(), unit_tag), size = 8)
                ax2.set_ylabel('Rel. Impro. (%)', size = 8)
                ylbl = ax1.yaxis.get_label()
                ylbl_target = ax2.yaxis.get_label()
                ax1.yaxis.set_label_coords(-0.1,ylbl.get_position()[1])
                ax2.yaxis.set_label_coords(-0.1,ylbl_target.get_position()[1])
            if key == 'energy' or key == 'direction':
                labels = [item.get_text() for item in ax2.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax2.set_xticklabels(empty_string_labels)  
            else:
                ax2.set_xlabel('Energy' + ' (log10 GeV)', size = 10)

    fig.suptitle('Resolution Performance', size = 12)
    fig.savefig('performance_track_cascade_combined.pdf',bbox_inches="tight")

    return fig


width_limits = {'energy':{'x':[0,3], 'y':[-0.5,1.5]},
                'zenith': {'x':[0,3], 'y':[-100,100]},
                'azimuth': {'x':[0,3], 'y':[-100,100]},
                'XYZ': {'x':[0,3], 'y':[-100,100]},
                'angular_res': {'x':[0,3], 'y':[-100,100]}}

key_bins = { 'energy': np.arange(0, 3.25, 0.25),
                'zenith': np.arange(0, 180, 10),
                'azimuth': np.arange(0, 2*180, 20) }


plot_config = {'data': '/remote/ceph/user/o/oersoe/paper_data/data/0000/reconstruction.csv',
                'width': width_limits,
                'key_bins': key_bins}
#['zenith', 'energy','azimuth' ,'angular_res', 'XYZ']
#make_resolution_plots(targets = ['zenith', 'energy','azimuth' ,'angular_res', 'XYZ'], plot_config = plot_config, include_retro= True, track_cascade = True)
make_combined_resolution_plot(targets =['zenith', 'energy','angular_res', 'XYZ'], plot_config = plot_config, include_retro= True, track_cascade = True)
