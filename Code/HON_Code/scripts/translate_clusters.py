import pprint
import sys
import re
import itertools
import operator
import os
import csv
from matplotlib import pylab as plt
import time
import collections
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
import math
import matplotlib


# output/2005.tree
def GetPortData(fn, field, delim):
    ports = {}
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delim)
        for row in reader:
            ports[row[field]] = row
    return ports

def build_HON_cluster(dic_file,tree_file):

    clu_file = tree_file.replace('tree','clu')

    NewToOld={}
    OldToNew={}
    # 2005.dic
    with open(dic_file) as g:
        for line in g:
            OldPort=line.strip('\n').split()[0]
            NewPort=int(line.split()[1].strip())
            NewToOld[NewPort]=OldPort
            OldToNew[OldPort]=NewPort
    print('Dict built.')
    

    cluster_labels={} # which port cluste 1 corresponds to (lables)
                #get cluster_lables using tree file
    lines = open(tree_file,'r').readlines()
    curr_cluster = ""
    for line in lines:
        if '#' in line:
            continue
        else:
            cluster_id, flow, _, port_num =line.strip('\n').split(' ')
            cluster_id =cluster_id.split(':')[0]
            OldPort = NewToOld[int(port_num)]
            if curr_cluster != cluster_id:
                cluster_labels[int(cluster_id)] = int(OldPort.split('|')[0]) 
                curr_cluster = cluster_id
    print('Tree parsed.')


    #translate map back to old ports
    cluster={}   # a dictionary that tells the cluster number for each port
    with open(clu_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            if '#' not in line:
                NewPort,cluster_num, flow =line.strip('\n').split(' ')
                OldPort=NewToOld[int(NewPort)]
                cluster[OldPort]=cluster_num

    ordered_cluster=collections.OrderedDict(sorted(cluster.items()))
    hon_cluster=defaultdict(list) 
    for key,val in ordered_cluster.items():
        hon_cluster[int(key.split('|')[0])].append(cluster_labels[int(val)])
    

    # a collection of how many times each cluster appears
    all_clusters=[]
    for key,val in hon_cluster.items():
        for item in set(val):
            all_clusters.append(item)
    freq = {key:len(list(group)) for key, group in groupby(np.sort(all_clusters))} # a dictionary which includes 
                                                                                    #top major clusters. 
    print("Number of Clusters",len(set((all_clusters))))

    
    #O_freq_clusters[0]:{0, :{2613}, 864} most frequent cluster is 0, it's correspond to port '2613|' 
     #it's frequency in all ports is 864
    O_freq_clusters=defaultdict(lambda: defaultdict()) 
    sorted_freq= collections.OrderedDict(sorted(freq.items(), key=operator.itemgetter(1),reverse=True)) 
    length=list(range(len(set(sorted(all_clusters)))+1))[1:]


    i=0
    for key,val in sorted_freq.items():
        try:
            O_freq_clusters[length[i]][key]=val
        except:
            pass
        i=i+1

    #flipped_O_freq_clusters[2613]:{0} it tells you each cluster lable rank(labeled by port) based on its frequency
    flipped_O_freq_clusters = defaultdict(dict)
    for key, val in O_freq_clusters.items():
        for subkey, subval in val.items():
            flipped_O_freq_clusters[subkey] = key

    #final_cluster[OldPort]=[Clusters that port belongs to, clustered labels & ordered based on begin major]
    final_cluster=defaultdict(list)
    for key,val in hon_cluster.items():
        for item in val:        
            final_cluster[key].append(flipped_O_freq_clusters[item])
    return final_cluster,O_freq_clusters,all_clusters,hon_cluster

def write_merged_clusters(final_cluster,agg_risks,ports,outpath):
    l=range(1,50)
    c_lis=['c'+str(i) for i in l]
    #','.join(map(str,['ID','NAME','CLUSTERS','LONGITUDE','LATITUDE']+c_lis))+'\n'
    with open(outpath, 'w') as f:
        f.write(','.join(map(str,['ID','NAME','CLUSTERS','RISK','LONGITUDE','LATITUDE']+c_lis))+'\n')
        O_major_ports=sorted(final_cluster, key=lambda k: len(final_cluster[k]), reverse=True)
        for port in O_major_ports:
            clusters=final_cluster[port]
            a=len(set(clusters))
            try:
                #if agg_risks[port]<0.001:
                    
                f.write(','.join(map(str,[  port,ports[str(port)]['NAME'],a,agg_risks[port],
                                        ports[str(port)]['LONGITUDE_DECIMAL'],
                                          ports[str(port)]['LATITUDE_DECIMAL'] ])))

                for cluster in l:   
                    if cluster in clusters:
                        f.write('1,')
                    else:
                        f.write('0,')
                f.write('\n')
            except:
                None
            
    print('finished')

def write_clusters(final_cluster,outpath):
    with open(outpath, 'w') as f:
        [f.write('{0} {1}\n'.format(key, ','.join(map(str,value)))) for key, value in final_cluster.items()]
    print('finished')


def risk_dic(file,delim):
    print(file)
    dic=defaultdict()
    with open(file,'r') as f:
            lines=f.readlines()
            for line in lines:
                node=int(line.split(delim)[0])
                risk=float(line.split(delim)[1].split(' ')[0])
                dic[node]=risk
    return(dic)


def risk_mean(adjlist):#{source:{dest1:r1,dest2:r2}}
                             #{source:mean_risk(r1,r2)}}       
    agg_dic=defaultdict()
    for FromPort,ToPort in adjlist.items():
        agg_dic[FromPort]=np.mean(list(adjlist[FromPort].values()))
    return agg_dic

def normalize(List):
    new_dic=defaultdict()
    MIN=min(List.values())
    MAX=max(List.values())
    for item,value in List.items():
        new_dic[item]=abs((value-MIN)/(MAX-MIN))
    return new_dic


