import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pickle
import os
import matplotlib as mpl
mpl.use('pdf')
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 

def add_truth(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select event_no, energy, interaction_type, pid, position_x, position_y, position_z from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    
    truth['track'] = 0
    truth.loc[(abs(truth['pid']) == 14) & (truth['interaction_type'] == 1), 'track'] = 1
    add_these = []
    for key in truth.columns:
        if key not in data.columns:
            add_these.append(key)
    for key in add_these:
       data[key] = truth[key]
    return data
    
def get_interaction_type(row):
    if row["interaction_type"] == 1:  # CC
        particle_type = "nu_" + {12: 'e', 14: 'mu', 16: 'tau'}[abs(row['pid'])]
        return f"{particle_type} CC"
    else:
        return "NC"
def resolution_fn(r,target):
        if target in ['energy', 'zenith', 'azimuth']:
            if len(r) > 1:
                return (np.percentile(r, 84) - np.percentile(r, 16)) / 2.
            else:
                return np.nan
        else:
            if len(r) > 1:
                return (np.percentile(r, 50))
            else:
                return np.nan

def add_energylog10(df):
    df['energy_log10'] = np.log10(df['energy'])
    return df

def get_error(residual,target):
    rng = np.random.default_rng(42)
    w = []
    for i in range(150):
        new_sample = rng.choice(residual, size = len(residual), replace = True)
        w.append(resolution_fn(new_sample,target))
    return np.std(w)

def get_roc_and_auc(data, target):
    fpr, tpr, _ = roc_curve(data[target], data[target+'_pred'])
    auc_score = auc(fpr,tpr)
    return fpr,tpr,auc_score  

def EuclideanDistance(data, target):
    return np.sqrt((data['position_x'] - data['position_x_pred'])**2 + (data['position_y'] - data['position_y_pred'])**2 +(data['position_z'] - data['position_z_pred'])**2)


def convert_to_unit_vectors(data, post_fix):
    
    data['x'] = np.cos(data['azimuth'])*np.sin(data['zenith'])
    data['y'] = np.sin(data['azimuth'])*np.sin(data['zenith'])
    data['z'] = np.cos(data['zenith'])

    data['x' + post_fix] = np.cos(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['y' + post_fix] = np.sin(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['z' + post_fix] = np.cos(data['zenith' + post_fix])
    return data

def calculate_angular_difference(data):
    post_fix = '_pred'
    data = convert_to_unit_vectors(data, post_fix)
    dotprod = (data['x']*data['x' + post_fix].values + data['y']*data['y'+ post_fix].values + data['z']*data['z'+ post_fix].values)
    norm_data = np.sqrt(data['x'+ post_fix]**2 + data['y'+ post_fix]**2 + data['z'+ post_fix]**2).values
    norm_truth = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2).values

    cos_angle = dotprod/(norm_data*norm_truth)

    return np.arccos(cos_angle).values*(360/(2*np.pi))

def calculate_width(data_sliced, target):
    track =data_sliced.loc[data_sliced['track'] == 1,:].reset_index(drop = True)
    cascade =data_sliced.loc[data_sliced['track'] == 0,:].reset_index(drop = True)
    if target == 'energy':
        residual_track = ((track[target + '_pred'] - track[target])/track[target])*100
        residual_cascade = ((cascade[target + '_pred'] - cascade[target])/cascade[target])*100
        residual =((data_sliced[target + '_pred'] - data_sliced[target])/data_sliced[target])*100
    elif target == 'zenith':
        residual_track = (track[target + '_pred'] - track[target])*(360/(2*np.pi))
        residual_cascade = (cascade[target + '_pred'] - cascade[target])*(360/(2*np.pi))
        residual = (data_sliced[target + '_pred'] - data_sliced[target])*(360/(2*np.pi))
    elif target == 'XYZ':
        residual_track =  EuclideanDistance(track, target)
        residual_cascade =  EuclideanDistance(cascade, target)
        residual = EuclideanDistance(data_sliced, target)
    elif target == 'angular_res':
        residual_track =   calculate_angular_difference(track)
        residual_cascade =  calculate_angular_difference(cascade)
        residual = calculate_angular_difference(data_sliced)
    else:
        residual_track = (track[target + '_pred'] - track[target])
        residual_cascade = (cascade[target + '_pred'] - cascade[target])
        residual = (data_sliced[target + '_pred'] - data_sliced[target])

    return resolution_fn(residual_track,target), resolution_fn(residual_cascade,target), resolution_fn(residual,target) , get_error(residual_track,target), get_error(residual_cascade,target),get_error(residual,target)

def get_width(df, target):
    track_widths = []
    cascade_widths = []
    track_errors = []
    cascade_errors = []
    energy = [] 
    bins = np.arange(0,3.1,0.1)
    if target in ['zenith', 'energy', 'XYZ', 'angular_res']:
        track_width, cascade_width, width, track_error, cascade_error, width_error = calculate_width(df, target)
        track_plot_data = pd.DataFrame({'width': [track_width], 'width_error': [track_error]})
        cascade_plot_data = pd.DataFrame({'width': [cascade_width], 'width_error': [cascade_error]})
        all_signatures = pd.DataFrame({'width': [width], 'width_error': [width_error]})
        return track_plot_data, cascade_plot_data,  all_signatures
    else:
        print('target not supported: %s'%target)

def get_auc_error(df, target):
    rng = np.random.default_rng(42)
    columns = df.columns
    aucs= []
    for i in range(150):
        new_sample = pd.DataFrame(data = rng.choice(df, size = len(df), replace = True), columns = columns)
        _, _, auc = get_roc_and_auc(new_sample,  target)
        aucs.append(auc)
    return np.std(aucs)



def add_nominal(results,target):
    database = '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db'
    if target != 'angular_res':
        nom_path = '/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_%s/results.csv'%target
        df = pd.read_csv(nom_path)
    else:
        nom_path_az = '/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_azimuth/results.csv'
        nom_path_zen = '/home/iwsatlas1/oersoe/phd/paper/regression_results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_paper_valid_set_zenith/results.csv'
        df_az = pd.read_csv(nom_path_az).sort_values('event_no').reset_index(drop =True)
        df_zen = pd.read_csv(nom_path_zen).sort_values('event_no').reset_index(drop =True)
        df_az['zenith'] = df_zen['zenith']
        df_az['zenith_pred'] = df_zen['zenith_pred']
        df = df_az
    df = add_truth(df, database)
    df = add_energylog10(df)
    if target not in ['track', 'neutrino']:
        plot_data_track, plot_data_cascade, plot_data_all = get_width(df, target)
        results['nominal'] = {'track': plot_data_track, 'cascade': plot_data_cascade, 'all_pid' : plot_data_all}
    else:
        _, _, auc = get_roc_and_auc(df,  target)
        auc_error = get_auc_error(df, target)
        results['nominal'] = {'auc': auc, 'auc_error': auc_error}
    return results

def pickle_exists(target, variable, save_path):
    files = os.listdir(save_path)
    if '%s_%s.pickle'%(target, variable) in files:
        print('Found file')
        return True
    else:
        return False

def save_pickle(results, pickle_path, target, variable):
    with open(pickle_path + '/%s_%s.pickle'%(target, variable), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_pickle(pickle_path, target, variable):
    with open(pickle_path + '/%s_%s.pickle'%(target, variable), 'rb') as handle:
        result = pickle.load(handle)
    return result

def get_data(target, variable):
    database = '/mnt/scratch/rasmus_orsoe/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db'
    path = '/home/iwsatlas1/oersoe/phd/paper/input_pertubation/dev_lvl7_robustness_muon_neutrino_0000'
    pickle_path = '/home/iwsatlas1/oersoe/phd/paper/input_pertubation/plot_data'
    has_pickle = pickle_exists(target, variable, pickle_path)
    print(has_pickle)
    if has_pickle == False:
        results = {}
        folders = os.listdir(path)
        target_folders = []
        if target == 'angular_res':
            for folder in folders:
                if 'azimuth' in folder and variable in folder:
                    target_folders.append(folder)
            print(target_folders)
        else:
            for folder in folders:
                if target in folder and variable in folder:
                    target_folders.append(folder)
        target_folders.sort() ## Ensures same sorting
        for folder in target_folders:
            predictions_path = path + '/' + folder + '/results.csv'
            if target == 'angular_res':
                predictions_path_az = path + '/' + folder + '/results.csv'
                predictions_path_zen = path + '/' + folder.replace('azimuth', 'zenith') + '/results.csv'
                df_az = pd.read_csv(predictions_path_az).sort_values('event_no').reset_index(drop = True)
                df_zen = pd.read_csv(predictions_path_zen).sort_values('event_no').reset_index(drop = True)
                df_az['zenith'] = df_zen['zenith']
                df_az['zenith_pred'] = df_zen['zenith_pred']
                df = df_az
            else:
                df = pd.read_csv(predictions_path).sort_values('event_no').reset_index(drop = True)
            df = add_truth(df, database)
            df = add_energylog10(df)
            if target not in ['track', 'neutrino']:
                plot_data_track, plot_data_cascade, plot_data_all = get_width(df, target)
                results[folder.split('_')[-1]] = {'track': plot_data_track, 'cascade': plot_data_cascade, 'all_pid': plot_data_all}
            else:
                _, _, auc = get_roc_and_auc(df,  target)
                auc_error = get_auc_error(df, target)
                results[folder.split('_')[-1]] = {'auc': auc, 'auc_error': auc_error}
        results = add_nominal(results, target)
        save_pickle(results, pickle_path, target, variable)
    else:
        results = load_pickle(pickle_path, target, variable)   
    
    return results   



def sort_experiments(results,variable):
    #results.pop('nominal')
    print(results.keys())
    keys = list(results.keys())
    keys.remove('nominal')
    if variable == 'dom_time':
        washed_keys = []
        for key in keys:
            washed_keys.append(int(key.split('ns')[0]))
        for i in range(len(washed_keys)):
            washed_keys[i] = str(washed_keys[i])
        for i in range(len(keys)):
            results[washed_keys[i]] = results[keys[i]]
            #results.pop(keys[i])
        drop_these = ['100', '50', '10']
        for i in drop_these:
            washed_keys.remove(i)
            results.pop(i)
        
        for i in range(len(washed_keys)):
            washed_keys[i] = int(washed_keys[i])
        washed_keys.sort()
        for i in range(len(washed_keys)):
            washed_keys[i] = str(washed_keys[i])


    if variable == 'charge':
        washed_keys = []
        keys.remove('1ns')
        keys.remove('0.5ns')
        if '[0.25]' in keys:
            keys.remove('0.25ns')

        for key in keys:
            #if key not in ['1.0q','5.0q','10.0q']:
            if '['  in key or ']' in key:
                k = key.replace('[', '')
                k = k.replace(']', '')
                k = k 
                washed_keys.append(k)
            else:
                washed_keys.append(key.replace('ns', ''))
        for i in range(len(keys)):
            results[washed_keys[i]] = results[keys[i]]
            results.pop(keys[i])
        drop_these = ['10', '5', '0.5', '1.0']
        for i in drop_these:
            print(i)
            washed_keys.remove(i)
            results.pop(i)
        for i in range(len(washed_keys)):
            washed_keys[i] = float(washed_keys[i])
        print('--------------------')
        print(washed_keys)
        print(results.keys())
        print('--------------------')
        washed_keys.sort()
        for i in range(len(washed_keys)):
            washed_keys[i] = str(washed_keys[i])
        
        
        print(washed_keys)

    if variable == 'dom_x':
        washed_keys = []
        for key in keys:
            k = [0, 0, 0.25]
            k = key.replace("[", '(' )
            k = k.replace("]", ')' )
            k = k.replace('(0, 0,', '(')
            k = k.replace(', 0)', ')')
            if ',' not in k:
                k = k.replace('(', '')
                k = k.replace(')', '')
            #if k in ['(0.25)', '(1)', '(1,1)','(3,3)']:
            washed_keys.append(k)
        for i in range(len(washed_keys)):
            washed_keys[i] = str(washed_keys[i])
        for i in range(len(keys)):
            results[washed_keys[i]] = results[keys[i]]
            results.pop(keys[i])
        drop_these = [' 0', ' 5', '(5, 5)']
        for i in drop_these:
            washed_keys.remove(i)
            results.pop(i)
        
        #for key in results.keys():
        #    if key not in washed_keys:
        #        results.pop(key)

    return results,washed_keys

def relative_improvement_error(w1,w2, w1_sigma,w2_sigma):
    w1 = np.array(w1)
    w1_sigma = np.array(w1_sigma)
    sigma = np.sqrt(((1/w2)*w1_sigma)**2  + ((w1/w2**2)*w2_sigma)**2)
    return sigma

def make_bar_plot(targets, variables):
    width = 3.176*1.3
    height = 2.388
    fontsize = 6
    errorbar_lw = 0.5
    fig, ax = plt.subplots(1,1, figsize = (width,height), sharex=  True)
    bar_width = 0.15*3
    fmt = '.'
    capsize = 0
    s = 0 #markersize
    edge_width = 0.5
    
    c = len(targets)
    x_offset = np.linspace(-(c-1)/2,(c-1)/2, c)
    x = np.arange(0,(5+20)*4,4)
    x = np.delete(x,[2,6,9])

    colors = ['tab:blue','tab:orange', 'tab:olive', 'tab:purple', 'tab:green', 'tab:red']
    print(x_offset)
    xlabels = []
    is_first = True
    for target in targets:
        print(target)
        c = c - 1
        x_count = 0
        label_set = False
        if target == 'XYZ':
            label = '$V_{xyz}$'
        elif target == 'track':
            label = '$ \\mathcal{T}/\\mathcal{C}$' +' (AUC)'
        elif target == 'neutrino':
            label = '$\\nu / \\mu$' + ' (AUC)'
        elif target == 'angular_res':
            label = 'Direction'
        else:
            label = target.capitalize()
        # if target == 'angular_res':
        #     label = '$\\vec{r}$'
        # if target == 'energy':
        #     label =  'E'
        # if target == 'zenith':
        #     label = '$\\theta$'
        # if target == 'azimuth':
        #     label = '$\\phi$'
        for variable in variables:
            results = get_data(target, variable)
            results,keys = sort_experiments(results,variable)
            if target == 'neutrino':
                print('---------')
                print(keys)
                print(results.keys())
                print('---------')
            for i in range(len(keys)):
                key = keys[i]
                if label_set == False:
                    if target not in ['track', 'neutrino']:
                        ax.bar(x[x_count] - x_offset[c]*bar_width, (1- results[key]['all_pid']['width']/results['nominal']['all_pid']['width'])*100,bar_width,  label = label, color = colors[c], alpha = 1)
                        ax.errorbar(x[x_count] - x_offset[c]*bar_width, (1- results[key]['all_pid']['width']/results['nominal']['all_pid']['width'])*100, relative_improvement_error(results[key]['all_pid']['width'],results['nominal']['all_pid']['width'],results[key]['all_pid']['width_error'],results['nominal']['all_pid']['width_error'])*100,fmt = fmt,capsize = capsize, markeredgewidth = edge_width, color = colors[c], alpha = 1, markersize = s, lw = errorbar_lw, ecolor = 'black')
                    else:
                        ax.bar(x[x_count] - x_offset[c]*bar_width, (-1 +results[key]['auc']/results['nominal']['auc'])*100,bar_width,  label = label, color = colors[c], alpha = 1)
                        ax.errorbar(x[x_count] - x_offset[c]*bar_width, (-1+ results[key]['auc']/results['nominal']['auc'])*100, relative_improvement_error(results[key]['auc'],results['nominal']['auc'],results[key]['auc_error'],results['nominal']['auc_error'])*100,fmt = fmt,capsize = capsize, color = colors[c], alpha = 1, markersize = s, lw = errorbar_lw, markeredgewidth = edge_width, ecolor = 'black')
                else:
                    if target not in ['track', 'neutrino']:
                        ax.bar(x[x_count] - x_offset[c]*bar_width, (1- results[key]['all_pid']['width']/results['nominal']['all_pid']['width'])*100,bar_width, color = colors[c], alpha = 1)
                        ax.errorbar(x[x_count] - x_offset[c]*bar_width, (1- results[key]['all_pid']['width']/results['nominal']['all_pid']['width'])*100, relative_improvement_error(results[key]['all_pid']['width'],results['nominal']['all_pid']['width'],results[key]['all_pid']['width_error'],results['nominal']['all_pid']['width_error'])*100,fmt = fmt,capsize = capsize, color = colors[c], alpha = 1, markeredgewidth = edge_width,markersize = s, lw = errorbar_lw, ecolor = 'black')
 
                    else:
                        ax.bar(x[x_count] - x_offset[c]*bar_width, (-1 +results[key]['auc']/results['nominal']['auc'])*100,bar_width, color = colors[c], alpha = 1)
                        ax.errorbar(x[x_count] - x_offset[c]*bar_width, (-1+ results[key]['auc']/results['nominal']['auc'])*100, relative_improvement_error(results[key]['auc'],results['nominal']['auc'],results[key]['auc_error'],results['nominal']['auc_error'])*100,fmt = fmt,capsize = capsize, color = colors[c], alpha = 1,markersize = s,lw = errorbar_lw, markeredgewidth = edge_width, ecolor = 'black')
                
                x_count += 1
                label_set = True
            if is_first:
                xlabels.extend(keys)
        is_first = False
    ax.set_xticks(x[0:(len(xlabels))])
    ax.set_xticklabels(xlabels, rotation = 90, fontsize = 6)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid()
    ax.yaxis.grid(False)
    plt.ylabel('Variation (%)', fontsize = 6)
    plt.ylim(-3.2,1)
    trans = ax.get_xaxis_transform()
    # lengthB = 20.58
    ax.annotate(text = r'$\sigma_{DOM_{Time}}$ (ns)', xy = (1.5,-0.17), xytext = (1.5,-0.25), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 3 , lengthB=0.8', connectionstyle="bar, angle = 0, fraction = 0",color="k", lw=0.5), size = fontsize)

    ax.annotate(text = r'$\sigma_{DOM_{z}}$ (m)', xy = (15,-0.17), xytext = (15,-0.25), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 4.5 , lengthB=0.8', connectionstyle="bar, angle = 0, fraction = 0",color="k", lw=0.5), size = fontsize)

    ax.annotate(text = r'$\sigma_{DOM_{xy}}$ (m)', xy = (30,-0.17), xytext = (30,-0.25), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 3.6 , lengthB=0.8', connectionstyle="bar, angle = 0, fraction = 0",color="k", lw=0.5), size = fontsize)

    ax.annotate(text = r'$\sigma_{DOM_{charge}}$ (p.e)', xy = (42.5,-0.17), xytext = (42.5,-0.25), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 3.2 , lengthB=0.8', connectionstyle="bar, angle = 0, fraction = 0",color="k", lw=0.5), size = fontsize)

    plt.plot(x[0:(len(xlabels))], np.repeat(0, len(x[0:(len(xlabels))])), lw = 0.25, color = 'black')
    legend = ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize = 4.5, frameon=False)

    fig.savefig('/home/iwsatlas1/oersoe/github/gnn_paper_plot_code-1_BACKUP/input_pertubation/plots/bar_resolution_pertubation.pdf',bbox_inches="tight")

variables = ['dom_time', 'dom_x', 'charge']
make_bar_plot( ['zenith', 'energy', 'angular_res', 'track', 'XYZ', 'neutrino'], variables)
#for target in ['zenith', 'energy', 'track']:
#    if target != 'track':
#        make_bar_plot(target, variables)
    #else:
    #    plot_roc(target)