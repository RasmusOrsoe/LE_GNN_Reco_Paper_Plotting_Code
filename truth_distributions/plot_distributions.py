import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import matplotlib as mpl
from os.path import exists
import pickle
import os
from copy import deepcopy
mpl.use('pdf')
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_test_sets(path):
    data = {}
    files = os.listdir(path)
    for file in files:
        data[file.replace('.csv', '')] = pd.read_csv(path + '/' + file)
    return data

def get_test_sets_v2(path):
    data = {}
    files = os.listdir(path)
    for file in files:
        if 'reco' in file:
            data[file.replace('.csv', '')] = pd.read_csv(path + '/' + file)
    return data

def add_post_fix(target, is_retro):
    if is_retro != None:
        if is_retro:
            target = target + '_retro'
        else:
            target = target + '_pred'
    return target

def transform_data_and_get_bins(df, target, is_retro = None):
    target = add_post_fix(target, is_retro)
    try:
        df.shape[1]
    except:
        df = pd.DataFrame(data = df, columns= [target])
    
    if 'zenith' in target or 'azimuth' in target:
        df[target] = df[target]*(360/(2*np.pi))
        if 'zenith' in target:
            bins = np.arange(0,180,5)
        else:
            bins = np.arange(0,360,10)
    elif 'energy' in target:
        df[target] = np.log10(df[target])
        bins = np.arange(0,3, 0.1)
    elif 'position_x' in target or 'position_y' in target:
        bins = np.arange(-300,300, 20)
    elif 'position_z' in target:
        bins = np.arange(-800,0, 40)
    return df, bins

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data

def check_and_add_truth(data, database):
    try:
        data['pid']
        return data
    except:
        data = data.sort_values('event_no').reset_index(drop = True)
        with sqlite3.connect(database) as con:
            query = 'SELECT event_no, pid FROM truth WHERE event_no in %s'%str(tuple(data['event_no']))
            truth = pd.read_sql(query, con).sort_values('event_no').reset_index(drop = True)
        data['pid'] = truth['pid']
        return data
        
def get_training_sample(test_set, training_selections, database, selection):
    if selection == 'track_cascade':
        training_selection = training_selections['track']
    elif selection == 'signal':
        training_selection = training_selections['neutrino']
    else:
        training_selection = training_selections['reco']
    with sqlite3.connect(database) as con:
        query = 'SELECT * FROM truth WHERE event_no in %s and event_no not in %s'%(str(tuple(training_selection['event_no'])), str(tuple(test_set['event_no'])))
        training_set = pd.read_sql(query,con)
    return training_set

def prepare_data(path, database):
    if exists('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions.pickle'):
        with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions.pickle', 'rb') as file:
            plot_data = pickle.load(file)
    else:
        test_sets = get_test_sets(path)
        training_selections = {'track': pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/selections/paper_selections/track/train_selection.csv'),
                                'neutrino': pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/selections/paper_selections/signal/train_selection.csv'),
                                'reco': pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/selections/paper_selections/regression/train_selection.csv') }
        targets = ['energy', 'zenith', 'azimuth', 'position_x', 'position_y', 'position_z']
        plot_data = {}
        for selection in test_sets.keys():
            plot_data[selection] = {}
            for target in targets:
                data = test_sets[selection]
                data = check_and_add_truth(data, database)
                training_set = get_training_sample(data, training_selections, database, selection)
                plot_data[selection][target] = {'test': data, 'train': training_set}
        with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions.pickle', 'wb') as file:
            pickle.dump(plot_data, file)
    return plot_data


def prepare_data_v2(path, database):
    #if exists('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions_v2.pickle'):
    #    with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions_v2.pickle', 'rb') as file:
    #        plot_data = pickle.load(file)
    #else:
    #   test_sets = get_test_sets_v2(path)
    #    training_selections = {'reco': pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/regression_results/selections/paper_selections/regression/train_selection.csv') }
    #    targets = ['energy', 'zenith', 'azimuth', 'position_x', 'position_y', 'position_z']
    #    plot_data = {}
    #    recos = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/paper_data/data/0000/reconstruction.csv')
    #    for selection in test_sets.keys():
    #        plot_data[selection] = {}
    #        for target in targets:
    #            data = test_sets[selection]
    #            data = check_and_add_truth(data, database)
    #            training_set = get_training_sample(data, training_selections, database, selection)
    #            plot_data[selection][target] = {'test': data, 'train': training_set}
    #            plot_data[selection][target + '_reco'] = {'dynedge':recos[target + '_pred'], 'retro':recos[target + '_retro']}
    #    with open('/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/truth_distributions_v2.pickle', 'wb') as file:
    #        pickle.dump(plot_data, file)
    return pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/paper_data/data/0000/reconstruction.csv')
            
def plot_distributions(path, database):
    width = 2*3*2.388#3.176
    height = 1.5*3.176#2*2.388
    plot_data = prepare_data(path, database)
    targets = ['energy', 'zenith', 'azimuth', 'position_x', 'position_y', 'position_z']
    fig, ax = plt.subplots(3,6,constrained_layout = True)
    fig.set_size_inches(width, height)
    density = True
    row_idx = 0
    for selection in plot_data.keys():
        column_idx = 0
        if selection == 'signal':
            label = '$\\nu_\\alpha / \\mu$'
        if selection == 'track_cascade':
            label = '$\\mathcal{T}/\\mathcal{C}$'
        if selection == 'reconstruction':
            label = 'Reconstruction'
        for target in targets:
            if target == 'energy':
                title = 'E'
                xlabel = '$log_{10}$' + ' (GeV)'
            if target == 'zenith':
                title = '$\\theta$'
                xlabel = 'Degrees'
            if target == 'azimuth':
                title = '$\\phi$'
                xlabel = 'Degrees'
            if target == 'position_x':
                title = '$V_x$'
                xlabel = '(M)'
            if target == 'position_y':
                title = '$V_y$'
                xlabel = '(M)'
            if target == 'position_z':
                title = '$V_z$'
                xlabel = '(M)'
            data = deepcopy(plot_data[selection][target]['test'])
            training_set = deepcopy(plot_data[selection][target]['train'])
            if selection != 'signal':
                data = remove_muons(data)
                training_set = remove_muons(training_set)
                data, bins = transform_data_and_get_bins(data, target)
                training_set, bins = transform_data_and_get_bins(training_set, target)
                ax[row_idx, column_idx].hist(data[target], bins =  bins, histtype = 'step', label = 'Test Set', color = 'tab:orange', density =  density)
                ax[row_idx, column_idx].hist(training_set[target], bins =  bins, histtype = 'step', label = 'Train Set', color = 'tab:blue', density =  density, lw = 0.5)
                #ax[row_idx, column_idx].set_ylim(0,1)
            else:
                data, bins = transform_data_and_get_bins(data, target)
                training_set, bins = transform_data_and_get_bins(training_set, target)
                ax[row_idx, column_idx].hist(data[target], bins =  bins, histtype = 'step', label = 'Test Set', color = 'tab:orange', density =  density)
                ax[row_idx, column_idx].hist(training_set[target], bins =  bins, histtype = 'step', label = 'Train Set', color = 'tab:blue', alpha = 1, density =  density, lw = 0.15)

            
            if row_idx == 0:
                ax[row_idx, column_idx].set_title(title)
            if column_idx == 0:
                ax[row_idx, column_idx].set_ylabel(label)
            #if column_idx != 0:
            #    labels = [item.get_text() for item in  ax[row_idx, column_idx].get_yticklabels()]
            #    empty_string_labels = ['']*len(labels)
            #    ax[row_idx, column_idx].set_yticklabels(empty_string_labels)
            if row_idx != 2:
                labels = [item.get_text() for item in  ax[row_idx, column_idx].get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax[row_idx, column_idx].set_xticklabels(empty_string_labels)
            else:
                ax[row_idx, column_idx].set_xlabel(xlabel)
            labels = [item.get_text() for item in  ax[row_idx, column_idx].get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax[row_idx, column_idx].set_yticklabels(empty_string_labels)
            ax[row_idx, column_idx].tick_params(left = False)
            #ax[row_idx, column_idx].set_ylim(ax[row_idx, 0].get_ylim() )
            column_idx +=1
        row_idx +=1
    fig.savefig('plots/truth_distributions.pdf',bbox_inches="tight")

def catch_cyclicity(residual):
    residual[residual>= 180] = 360 - residual[residual>= 180]
    residual[residual<= -180] = -(residual[residual<= -180] + 360)
    return residual

def get_residual_bins(target):
    n_bins = 30
    if target == 'energy':
        return np.linspace(-1.5,1.5,n_bins)
    if target == 'zenith':
        return np.linspace(-100,100,n_bins)
    if target == 'azimuth':
        return np.linspace(-180,180,n_bins)
    if 'position' in target:
        return np.linspace(-50,50,n_bins)

    

def plot_distributions_v2(path, database):
    width = 2*3*2.388#3.176
    height = width/3#3.176 #2*2.388
    plot_data = prepare_data_v2(path, database)
    targets = ['energy', 'zenith', 'azimuth', 'position_x', 'position_y', 'position_z']
    fig, ax = plt.subplots(2,6,constrained_layout = True)
    fig.set_size_inches(width, height)
    density = True
    column_idx = 0
    for target in targets:
        if target == 'energy':
            title = 'E'
            xlabel = title + ' $(log_{10}$' + ' GeV)'
        if target == 'zenith':
            title = '$\\theta$'
            xlabel = title + '  $(^{\circ})$'
        if target == 'azimuth':
            title = '$\\phi$'
            xlabel = title + '  $(^{\circ})$'
        if target == 'position_x':
            title = '$V_x$'
            xlabel = title + ' (m)'
        if target == 'position_y':
            title = '$V_y$'
            xlabel = title + ' (m)'
        if target == 'position_z':
            title = '$V_z$'
            xlabel = title + ' (m)'
        truth = deepcopy(plot_data[target])
        retro = deepcopy(plot_data[target + '_retro'])
        dynedge = deepcopy(plot_data[target + '_pred'])
        
        truth, bins = transform_data_and_get_bins(truth, target)
        retro, bins = transform_data_and_get_bins(retro, target, is_retro = True)
        dynedge, bins = transform_data_and_get_bins(dynedge, target, is_retro = False)
        ax[0,column_idx].hist(truth[target], bins =  bins, label = 'Truth', color = 'black', density =  density, alpha = 0.3)            
        ax[0,column_idx].hist(dynedge[target + '_pred'], bins =  bins, histtype = 'step', label = 'dynedge', color = 'tab:blue', density =  density)
        ax[0,column_idx].hist(retro[target + '_retro'], bins =  bins, histtype = 'step', label = 'Retro', color = 'tab:orange', density =  density)

        if target in ['zenith', 'azimuth', 'position_x', 'position_z','position_z']:
            residual_dynedge = dynedge[target + '_pred'] - truth[target]
            residual_retro = retro[target + '_retro'] - truth[target]
            if target == 'azimuth':
                residual_dynedge = catch_cyclicity(residual_dynedge)
                residual_retro = catch_cyclicity(residual_retro)
        elif target  == 'energy':
            #residual_dynedge = ((10**truth[target] - 10**dynedge[target + '_pred'])/(10**truth[target]))*100
            #residual_retro = ((10**truth[target] - 10**retro[target + '_retro']) /(10**truth[target]))*100
            residual_dynedge = truth[target] - dynedge[target + '_pred']
            residual_retro = truth[target] - retro[target + '_retro']

        residual_bins = get_residual_bins(target)
        n, _, _ = ax[1,column_idx].hist(residual_dynedge, bins =  residual_bins, histtype = 'step', label = None, color = 'tab:blue', density =  density)
        ax[1,column_idx].hist(residual_retro, bins = residual_bins, histtype = 'step', label = None, color = 'tab:orange', density =  density)
        ax[1,column_idx].plot(np.repeat(0,2), [0,np.max(n) + (np.max(n)/100)*10], color = 'black', lw = 0.5)

        ax[1,column_idx].set_xlabel(xlabel, fontsize = 14)
        if column_idx == 0:
            ax[1,column_idx].set_ylabel('Residual', fontsize = 14)
            ax[0,column_idx].set_ylabel('Disitribution', fontsize = 14)
            legend =fig.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=3, fontsize = 14, frameon=False)
        labels = [item.get_text() for item in  ax[0,column_idx].get_yticklabels()]
        empty_string_labels = ['']*len(labels)
        ax[0,column_idx].set_yticklabels(empty_string_labels)
        ax[0,column_idx].tick_params(left = False)
        labels = [item.get_text() for item in  ax[1,column_idx].get_yticklabels()]
        empty_string_labels = ['']*len(labels)
        ax[1,column_idx].set_yticklabels(empty_string_labels)
        ax[1,column_idx].tick_params(left = False)
        #ax[row_idx, column_idx].set_ylim(ax[row_idx, 0].get_ylim() )
        column_idx +=1
        
    fig.savefig('plots/truth_distributions.pdf',bbox_inches="tight")
     
path = '/mnt/scratch/rasmus_orsoe/paper_data_pass2/data_with_overlap_labels/0000'
database = '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db'

plot_distributions_v2(path, database)