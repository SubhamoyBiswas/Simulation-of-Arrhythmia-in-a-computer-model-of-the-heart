"""Please keep the input files for the times of activation of nodes in all cases under a single folder."""

"""Inporting the necessary modules"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os
import collections
from mpl_toolkits import mplot3d
import csv
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
"""=========================================================================================================================================================================="""

"""Input portion begins"""
node_paired_unpaired={}
clusters_case={}
no_info_nodes={}
list_data=[2, 3, 4, 6, 7, 8, 9, 10, 11]
threshold_for_neighbours_space=input("Enter threshold for nearest neighbours in space coordinates: ")
"""Create a folder where all outputs will be stored. If you want to reuse the code, first clear all contents of this folder."""
output_path=input("Enter output folder path: ")
in_positions=input("Enter data path for node positions: ")
in_triangles=input("Enter data path for close traingles: ")
in_time=input("Enter common data path for times of activation of nodes for all cases: ")
output_path.replace('"', '')
in_positions.replace('"', '')
in_triangles.replace('"', '')
in_time.replace('"', '')
for i in [2, 3, 4, 6, 7, 8, 9, 10, 11]:
    os.mkdir(f'{output_path}/case {str(i)}')

f=open(in_positions, 'r')
fl=f.readline()
node_pos=[]
while fl!='':
    fl.replace('\n', '')
    list_f=fl.split(" ")
    list_f[len(list_f)-1]=list_f[len(list_f)-1].replace('\n', '')
    list_f=[float(i) for i in list_f]
    list_f[0]=int(list_f[0])
    node_pos.append(list_f)
    fl=f.readline()
f.close()
with open(f'{output_path}/LOCATION.csv', 'w+') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(['Node no.', 'X', 'Y', 'Z'])
    for i in node_pos:
        writer.writerow(i)
csvFile.close()
node_positions=np.array(node_pos)
print("Node positions data retrived...")

tri_list=[]
df1=open(in_triangles, 'r')
dfl=df1.readline()
node_pos=[]
while dfl!='':
    dfl.replace('\n', '')
    list_f=dfl.split(" ")
    list_f[len(list_f)-1]=list_f[len(list_f)-1].replace('\n', '')
    list_f=[float(i) for i in list_f]
    list_f[0]=int(list_f[0])
    tri_list.append(list_f)
    dfl=df1.readline()
df1.close()
with open(f'{output_path}/TRIANGLES.csv', 'w+') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(['Triangle no.', 'Vertex 1', 'Vertex 2', 'Vertex 3'])
    for i in tri_list:
        writer.writerow(i)
csvFile.close()
print("Data for triangular mesh of three closest nodes retrieved...")

"""Input portion ends"""
"""==========================================================================================================================================================================="""

"""Output and function portion begins"""

def retrieval(filename):
    file_df=np.fromfile(filename, np.int32).reshape((-1,2))
    file_nodes=file_df[:,0]
    file_activ_time=file_df[:,1]/1000
    return file_nodes, file_activ_time


def distance_matrix(ln_array, nonln_array, lnt, nonlnt, node, case, ln_activ_time, ln_nodes, nonln_activ_time, nonln_nodes):
    """The function distance_matrix takes the node no. and its list of linear and non-linear activations and find all possible absolute difference between the instants."""
    ln_matrix, nonln_matrix=[], []
    ln_array=list(ln_array.reshape((1, lnt)))
    nonln_array=list(nonln_array.reshape((1, nonlnt)))
    for i in range(nonlnt):
        ln_matrix.append(ln_array)
    for i in range(lnt):
        nonln_matrix.append(nonln_array)
    ln_matrix=np.transpose(np.array(ln_matrix).reshape((nonlnt, lnt)))
    distance_mat=np.subtract(ln_matrix, np.array(nonln_matrix).reshape((lnt, nonlnt)))
    distance_matrix1=np.abs(distance_mat)
    if distance_matrix1.shape[0]==0 and distance_matrix1.shape[1]==0:
        no_info_nodes[case].append(node)
    else:
        paired_unpaired(distance_matrix1, node, case, ln_activ_time, ln_nodes, nonln_activ_time, nonln_nodes)


def paired_unpaired(matrix_set, node, case, ln_activ_time, ln_nodes, nonln_activ_time, nonln_nodes):
    """Using the results of distance_matrix, paired_unpaired finds the mutual nearest neighbour pairs among the activation times of the node."""
    node_paired_unpaired[case][node]={'paired':[], 'unpaired linear':[], 'unpaired non-linear':[]}
    paired_linear_activ_time_instants, paired_nonlinear_activ_time_instants=[], []
    paired_indices_ln, paired_indices_nonln=[], []
    for i in range(len(matrix_set)):
        p=list(matrix_set[i])
        q=p.index(min(p))
        if min(list(matrix_set[:, q]))==p[q]:
            node_paired_unpaired[case][node]['paired'].append([ln_activ_time[ln_nodes==node][i], nonln_activ_time[nonln_nodes==node][q]])
            paired_indices_ln.append(i)
            paired_indices_nonln.append(q) 
    for i in range(len(matrix_set)):
        if i in paired_indices_ln:
            continue
        else:
            node_paired_unpaired[case][node]['unpaired linear'].append(ln_activ_time[ln_nodes==node][i])   
    for i in range(len(matrix_set[0])):
        if i in paired_indices_nonln:
            continue
        else:
            node_paired_unpaired[case][node]['unpaired non-linear'].append(nonln_activ_time[nonln_nodes==node][i])


def moving_average(t_val, x_val, interval, n):
    """The function moving_average is used to perform moving average operation using a given window length."""
    dt = (interval[1] - interval[0]) / n
    tm = interval[0] + dt/2 + np.arange(n) * dt
    xm = np.zeros_like(tm)
    for i in range(n):
        xm[i] = np.mean(x_val[(t_val >= tm[i]-dt/2) & (t_val < tm[i]+dt/2)])
    return tm, xm


def spacing_graph(t_val, x_val, del_times):
    """The function spacing_graph finds the moving averaged RMSE values vs time over all nodes in the case"""
    ret_cl={}
    for i in range(3501):
        t_ret, x_ret=moving_average(t_val, x_val, [i, i+1499], 100)
        for j in range(50):
            if math.isnan(float(x_ret[j])):
                del_times.append(t_ret[j])
            else:
                ret_cl[t_ret[j]]=x_ret[j]
    ret_cl=collections.OrderedDict(sorted(ret_cl.items()))
    return ret_cl


def rmse_vs_time_plotter(master_rmse_avg, case):
    """The function rmse_vs_time_plotter plots the RMSE vs time plot"""
    plt.figure(figsize=(14, 10), dpi=400)
    plt.title(f'For case {str(case)} dataset')
    plt.xlabel('Time -->')
    plt.ylabel('RMSE -->')
    plt.plot(list(master_rmse_avg.keys()), list(master_rmse_avg.values()), 'r')
    plt.savefig(f"{output_path}/case {str(case)}/RMSE MOV AVG PLOT.png")
    

def KDE_all_linear(node_paired_unpaired, case, ln_activ_time):
    """KDE_all_linear finds the KDE of linear unpaired instants over all nodes in a case"""
    predict_ln=[]
    for i in list(node_paired_unpaired[case].keys()):
        if i in no_info_nodes:
            continue
        else:
            for j in node_paired_unpaired[case][i]['unpaired linear']:
                if j!='Unpaired linear':
                    predict_ln.append(j)                
    kde=KernelDensity(bandwidth=15, kernel='gaussian')
    predict_ln=np.array(predict_ln).reshape(-1, 1)
    kde.fit(predict_ln)
    logprob=kde.score_samples(np.linspace(min(ln_activ_time), max(ln_activ_time), 10000).reshape(-1, 1))
    plt.figure(figsize=(14, 10), dpi=400)
    plt.xlabel('Unpaired linear instant -->')
    plt.ylabel('KDE -->')
    plt.fill_between(np.linspace(min(ln_activ_time), max(ln_activ_time), 10000)[:9600], np.exp(logprob)[:9600], alpha=2)
    plt.savefig(f'{output_path}/case {str(case)}/KDE for unpaired linear instants over all nodes.png')
    
    
def KDE_all_nonlinear(node_paired_unpaired, case, nonln_activ_time):
    """KDE_all_nonlinear finds the KDE of non-linear unpaired instants over all nodes in a case"""
    predict_nonln=[]
    for i in list(node_paired_unpaired[case].keys()):
        if i in no_info_nodes:
            continue
        else:
            for j in node_paired_unpaired[case][i]['unpaired non-linear']:
                if j!='Unpaired non-linear':
                    predict_nonln.append(j)                
    kde=KernelDensity(bandwidth=15, kernel='gaussian')
    predict_nonln=np.array(predict_nonln).reshape(-1, 1)
    kde.fit(predict_nonln)
    logprob=kde.score_samples(np.linspace(min(nonln_activ_time), max(nonln_activ_time), 10000).reshape(-1, 1))
    plt.figure(figsize=(14, 10), dpi=400)
    plt.xlabel('Unpaired non-linear instant -->')
    plt.ylabel('KDE -->')
    plt.fill_between(np.linspace(min(nonln_activ_time), max(nonln_activ_time), 10000)[:9600], np.exp(logprob)[:9600], alpha=2)
    plt.savefig(f'{output_path}/case {str(case)}/KDE for unpaired non-linear instants over all nodes.png')
    
    
def recheck_node_pair_unpair(ndpunp, case, nd):
    """The function recheck_node_pair_unpair refreshes the noda data matrix to remove any sort of discrepancies if any."""
    a=[]
    for i in ndpunp['unpaired linear']:
        if i not in ndpunp['unpaired non-linear']:
            a.append(i)
    node_paired_unpaired[case][nd]['unpaired linear']=a
    
    
def location_unpaired_nodes(dict_loc, case):
    """The function location_unpaired_nodes finds the locations of those nodes which got unpaired at least once."""
    unpaired_node_loc2={}
    for i in list(dict_loc.keys()):
        i1=int(i)-1
        if i1 in no_info_nodes[case]:
            continue
        else:
            if len(node_paired_unpaired[case][i1]['unpaired linear'])!=0 or len(node_paired_unpaired[case][i1]['unpaired non-linear'])!=0:
                unpaired_node_loc2[i1]=dict_loc[i]
    return unpaired_node_loc2


def find_threshold_space(dict_loc, tri_list):
    """The function find_threshold_space determines a default value of threshold for space neighbours if in case user hasn't given any threshold as input"""
    dist_list2=[]
    for i in tri_list:
        a=math.sqrt((dict_loc[i[1]][0]-dict_loc[i[2]][0])**2 + (dict_loc[i[1]][1]-dict_loc[i[2]][1])**2 + (dict_loc[i[1]][2]-dict_loc[i[2]][2])**2)
        b=math.sqrt((dict_loc[i[2]][0]-dict_loc[i[3]][0])**2 + (dict_loc[i[2]][1]-dict_loc[i[3]][1])**2 + (dict_loc[i[2]][2]-dict_loc[i[3]][2])**2)
        c=math.sqrt((dict_loc[i[3]][0]-dict_loc[i[1]][0])**2 + (dict_loc[i[3]][1]-dict_loc[i[1]][1])**2 + (dict_loc[i[3]][2]-dict_loc[i[1]][2])**2)
        dist_list2.append(a)
        dist_list2.append(b)
        dist_list2.append(c)
    d95=np.percentile(np.array(dist_list2), 95)
    dist_list2=np.array(dist_list2).reshape(-1, 1)
    kde=KernelDensity(bandwidth=0.4, kernel='gaussian')
    kde.fit(dist_list2)
    log_prob_dist2=kde.score_samples(np.linspace(min(dist_list2), max(dist_list2), 20000).reshape(-1, 1))
    pdf_prob_dist2=np.exp(log_prob_dist2)
    peaks_dist2, _=find_peaks(pdf_prob_dist2)
    plt.figure(figsize=(14, 10), dpi=400)
    plt.plot(np.linspace(min(dist_list2), max(dist_list2), 20000).reshape(-1, 1), pdf_prob_dist2, 'r')
    plt.show()
    return d95


def neighbour_find_space(threshold, unpaired_node_loc2):
    """The function neighbour_find_space finds the neighbours of a node using only x, y and z coordinates."""
    neighbours_by_space={}
    for i in list(unpaired_node_loc2.keys()):
        min_dist2, fin_data2={}, []
        for j in list(unpaired_node_loc2.keys()):
            dist2=math.sqrt(math.pow(unpaired_node_loc2[j][0]-unpaired_node_loc2[i][0], 2) + math.pow(unpaired_node_loc2[j][1]-unpaired_node_loc2[i][1], 2) + math.pow(unpaired_node_loc2[j][2]-unpaired_node_loc2[i][2], 2))
            min_dist2[j]=dist2
        for j in list(min_dist2.keys()):
            if min_dist2[j]!=0 and min_dist2[j]<threshold:
                fin_data2.append([j, min_dist2[j]])
        if fin_data2!=[]:
            neighbours_by_space[i]=[]
            f_dat=np.array(fin_data2)
            neighbours_by_space[i]=list(f_dat[:, 0])
    return neighbours_by_space


def find_threshold_time(neighbours_by_space, node_paired_unpaired, case):
    """The function find_threshold_time finds the threshold to be used for time neighbours. It is usually taken as the lower confidence limit taking 2 std. deviations on the right side of mean line."""
    min_shift=[]
    for i in list(neighbours_by_space.keys()):
        p1=node_paired_unpaired[case][i]['unpaired linear']
        p1.extend(node_paired_unpaired[case][i]['unpaired non-linear'])
        for j in neighbours_by_space[i]:
            if j!=i:
                p11=node_paired_unpaired[case][j]['unpaired linear']
                p11.extend(node_paired_unpaired[case][j]['unpaired non-linear'])
                shifts=[]
                for k in p1:
                    for h in p11:
                        shifts.append(math.sqrt((k-h)**2))
                min_shift.append(min(shifts))
    N_shift=len(min_shift)
    mean_shift=sum(min_shift)/N_shift
    sd_sum=0
    for i in min_shift:
        sd_sum = sd_sum + (i-mean_shift)**2
    sd_shift=math.sqrt(sd_sum/N_shift)
    limit_shift=mean_shift + 1.645*sd_shift/math.sqrt(N_shift)
    return limit_shift
    
     
def neighbour_find_space_and_time(time_thres, neighbours_by_space, node_paired_unpaired, case):
    """The function neighbour_find_space_and_time finds the neighbours that are not only close in space but also in time."""
    neighbours_by_time_space={}
    for i in list(neighbours_by_space.keys()):
        neighbours_by_time_space[i]=[]
        p1=node_paired_unpaired[case][i]['unpaired linear']
        p1.extend(node_paired_unpaired[case][i]['unpaired non-linear'])
        for j in neighbours_by_space[i]:
            if j!=i:
                p11=node_paired_unpaired[case][j]['unpaired linear']
                p11.extend(node_paired_unpaired[case][j]['unpaired non-linear'])
                shifts=[]
                for k in p1:
                    for h in p11:
                        shifts.append(math.sqrt((k-h)**2))
                min_shift=min(shifts)
                if min_shift<time_thres:
                    neighbours_by_time_space[i].append(j)
    print('Time done')
    return neighbours_by_time_space


def delete_isolated_nodes(neighbours_by_time_space):
    """The function delete_isolated_nodes deletes all instances of nodes which don't have any neighbours."""
    final_neighbours={}
    for i in list(neighbours_by_time_space.keys()):
        if len(neighbours_by_time_space[i])!=0:
            final_neighbours[i]=neighbours_by_time_space[i]
    return final_neighbours


def clusters_from_neighbours(final_neighbours):
    """The function clusters_from_neighbours finds clusters from the neighbours data."""
    neighbour_groups=[]
    for i in list(final_neighbours.keys()):
        a=[i]
        a.extend(final_neighbours[i])
        neighbour_groups.append(a)    
    neighbour_groups=np.array(neighbour_groups)
    c, groups=1, [list(neighbour_groups[0])]
    while c<len(neighbour_groups):
        len_group, flag=len(groups), 0
        for i in range(len_group):
            inter_l=np.intersect1d(list(neighbour_groups[c]), list(groups[i]))
            if len(list(inter_l))!=0:
                groups[i]=np.union1d(list(groups[i]), list(neighbour_groups[c]))
                flag=1
                break
        if flag==0:
            groups.append(list(neighbour_groups[c]))
        flag=0
        c=c+1
    return groups


def remove_error_clusters(all_clusters, node_paired_unpaired, case):
    """The function remove_error_clusters deletes those clusters whose max. activation time exceeds 4950ms."""
    edit_clusters=[]
    for i in all_clusters:
        flag=0
        for j in i:
            a=node_paired_unpaired[case][j]['unpaired linear']
            a.extend(node_paired_unpaired[case][j]['unpaired non-linear'])
            if a!=[]:
                if max(a)>4950:
                    flag=1
                    break
        if flag==0:
            edit_clusters.append(i)
    return edit_clusters


def main_controller(threshold_space, node_positions, tri_list):
    """The function main_controller is the primary function which regulates the determination of these clusters."""
    for i1 in [2, 3, 4, 6, 7, 8, 9, 10, 11]:
        print("...")
        print("...")
        print("...")
        print(f"CASE {str(i1)} under study..........")
        ns=0
    
        node_paired_unpaired[i1]={}
        no_info_nodes[i1]=[]
        ln_nodes, ln_activ_time=retrieval(f'{in_time}/case{str(i1)}_linear.bin')
        nonln_nodes, nonln_activ_time=retrieval(f'{in_time}/case{str(i1)}_nonlinear.bin')
        N=max(nonln_nodes)+1
        print(f"Activation times for the nodes for case {str(i1)} retrieved...")
        matrix_set={}
        for i in range(N):
            ln_time=ln_activ_time[ln_nodes==i]
            nonln_time=nonln_activ_time[nonln_nodes==i]
            distance_matrix(ln_time, nonln_time, ln_time.size, nonln_time.size, i, i1, ln_activ_time, ln_nodes, nonln_activ_time, nonln_nodes)
        print("Mutual nearest neighbours for linear and non-linear activation times found...")
    
        master_rmse, master_t=[], []
        for i in list(node_paired_unpaired[i1].keys()):
            master_rmse.extend(list(np.power(np.abs(np.array(node_paired_unpaired[i1][i]['paired'])[:, 1]-np.array(node_paired_unpaired[i1][i]['paired'])[:, 0]), 2)))
            master_t.extend(list(np.array(node_paired_unpaired[i1][i]['paired'])[:, 0]))
        del_times=[]
        master_rmse_avg=spacing_graph(np.array(master_t), np.array(master_rmse), del_times)
        rmse_vs_time_plotter(master_rmse_avg, i1)
        print("RMSE vs Time plot determined and stored in output folder...")
    
        KDE_all_linear(node_paired_unpaired, i1, ln_activ_time)
        KDE_all_nonlinear(node_paired_unpaired, i1, nonln_activ_time)        
        
        dict_loc={}
        for i in node_positions:
            dict_loc[float(i[0])]=[i[1], i[2], i[3]]
        
        time_data={}
        for i in list(node_paired_unpaired[i1].keys()):
            if i in no_info_nodes[i1]:
                continue
            else:   
                if len(node_paired_unpaired[i1][i]['unpaired linear'])!=0 or len(node_paired_unpaired[i1][i]['unpaired non-linear'])!=0:
                    unp_act=node_paired_unpaired[i1][i]['unpaired linear']
                    unp_act.extend(node_paired_unpaired[i1][i]['unpaired non-linear'])
                    unp_act=sorted(unp_act)
                    for j in unp_act:
                        time_data[j]=i
                        
        for i in list(node_paired_unpaired[i1].keys()):
            if i in no_info_nodes[i1]:
                continue
            else:
                recheck_node_pair_unpair(node_paired_unpaired[i1][i], i1, i)
                
        print("Node data refreshed...")
            
        unpaired_node_loc2=location_unpaired_nodes(dict_loc, i1)
        
        if threshold_space:
            ns=1
            print(f"Using {str(threshold_space)} as threshold for space...")
        else:
            threshold_space=find_threshold_space(dict_loc, tri_list)
            print("No threshold for space detected. Using default threshold value...")
        
        neighbours_by_space=neighbour_find_space(float(threshold_space), unpaired_node_loc2)
        print("Neighbours in space found...")
        threshold_time=find_threshold_time(neighbours_by_space, node_paired_unpaired, i1)
        print(f"Threshold in time found as {str(threshold_time)} ms...")
        neighbours_by_time_space=neighbour_find_space_and_time(threshold_time, neighbours_by_space, node_paired_unpaired, i1)
        print("Nearest neighbours determined for every node...")
        final_neighbours=delete_isolated_nodes(neighbours_by_time_space)
        print("Isolated nodes discarded...")
        all_clusters=clusters_from_neighbours(final_neighbours)
        print("Clusters found from neighbours data...")
        modified_clusters=remove_error_clusters(all_clusters, node_paired_unpaired, i1)
        print("Clusters with max. activation time greater than 4950 milliseconds removed...")
        
        clusters_case[i1]=modified_clusters
        print(f"Clusters obtained for CASE {str(i1)}.")
        print("=====================================================================================")
        
        for i in list(node_paired_unpaired[i1].keys()):
            if i in no_info_nodes[i1]:
                continue
            else:
                recheck_node_pair_unpair(node_paired_unpaired[i1][i], i1, i)
                
        
def cluster_analysis():
    """The function cluster_analysis gives the KDE of cluster sizes. Using a suitable cutoff, we consider the clusters."""
    size_cl=[]
    for i in list(clusters_case.keys()):
        for j in clusters_case[i]:
            a=len(list(j))
            size_cl.append(a)
    size_cl=np.array(size_cl).reshape(-1, 1)
    kde=KernelDensity(bandwidth=0.4, kernel='gaussian')
    kde.fit(size_cl)
    log_prob_size=kde.score_samples(np.linspace(min(size_cl), max(size_cl), 20000).reshape(-1, 1))
    pdf_prob_size=np.exp(log_prob_size)
    plt.figure(figsize=(14, 10), dpi=400)
    plt.plot(np.linspace(min(size_cl), max(size_cl), 20000).reshape(-1, 1), pdf_prob_size, 'r')
    plt.savefig(f'{output_path}/cluster_size_distribution KDE.png')
                
    
main_controller(threshold_for_neighbours_space, node_positions, tri_list)
cluster_analysis()
        
