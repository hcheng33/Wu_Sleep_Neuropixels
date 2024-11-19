import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

def load_files(spike_times_file, spike_clusters_file, cluster_group_file):
    spike_times = np.load(spike_times_file)
    spike_clusters = np.load(spike_clusters_file)

    clust_group = pd.read_csv(cluster_group_file, sep='\t')
    #clust_label = clust_group.index[clust_group["label"] == "good"].tolist()
    clust_label = np.unique(spike_clusters)

    return spike_times, spike_clusters, clust_label

def spike_times_by_cluster(spike_times, spike_clusters, clust_label):
    spike_times_all = []
    #clust_ind = clust_label.index[clust_label["label"] == "good"].tolist()

    for i in clust_label:
        spike_times_clust = spike_times[spike_clusters == i]

        #if len(spike_times_clust) >= 30:
        spike_times_all.append(spike_times_clust)

    return spike_times_all

def firing_rate_calc(spike_times, spike_clusters, clust_label, bin_size, cluster):

    if cluster == "all":
        clust_tot = len(clust_label)
        clust_ind = clust_label
    else:
        clust_ind = cluster
        
    t_end = np.max(spike_times)
    t_bins = np.arange(0,t_end,bin_size)
    
    clust_num = len(clust_ind)
    fr = np.zeros((len(clust_ind), len(t_bins)-1))
    for i in range(len(clust_ind)):
        spikes_t_ind = spike_times[np.where(spike_clusters == clust_ind[i])[0]]
        spikes_count, edges = np.histogram(spikes_t_ind, t_bins)

        fr[i,:] = spikes_count

    return fr, t_bins[:-1], clust_num, clust_tot

def firing_rate_smooth(fr, sigma):
    fr_smooth = np.zeros(fr.shape)
    for i in range(len(fr_smooth)):
        fr_smooth[i,:] = gaussian_filter1d(fr[i,:], sigma)

    return fr_smooth