import numpy as np
import pandas as pd

def firing_rate_metics(spike_times, spike_clusters, clust_label):
    #spike_times = np.load(spike_times_file)
    #spike_clusters = np.load(spike_clusters_file)

    clust_ind = np.unique(spike_clusters)
    #clust_ind = clust_label
    cluster_spike_count = np.zeros(len(clust_ind))

    for i in range(len(clust_ind)):
        count = len(np.where(spike_clusters == clust_ind[i])[0])
        cluster_spike_count[i] = count

    firing_rate = cluster_spike_count/(np.max(spike_times)/30000)

    df = pd.DataFrame({'cluster_id':clust_ind, 'firing_rate':firing_rate})
    
    return df

def bursting_metrics(spike_times_all, freq, sample_rate):

    burst_count_all = []
    burst_ratio_all = []

    for i in range(len(spike_times_all)): # loop for clusters
        clust_t_delta = np.diff(spike_times_all[i])

        burst_count =sum(clust_t_delta <= ((1/freq)*sample_rate))

        if len(spike_times_all[i]) == 0:
            burst_ratio = 0 
        elif len(spike_times_all[i] >= 1):
            burst_ratio = burst_count/ (len(spike_times_all[i]))

        burst_count_all.append(burst_count)
        burst_ratio_all.append(burst_ratio)

    return burst_count_all, burst_ratio_all