import pprint
import sys
from collections import defaultdict
import time
import csv
import math
import sys
import time
from datetime import datetime
import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import itertools




##########################################################
#Writing down links for arcgis
#########################################################
### For all items in AdjList, each source has several destination. First, we average over all "given" ones to get the physical adj matrix

### Then for each destination, we aggregate all the scores over all the incoming ports (origins). 
###  Also for each country, we aggregate the risk over all the ports wihthin that country. So we take the number pf ports into account this way.
def write_hon_links(HON_links,agg_risk,p_AdjList,ports):
    b_agg_th=0;b_risk_th=0
    f_agg_th=0;f_risk_th=0
    i=0
    with open(HON_links, 'w') as f:
        f.write(','.join(map(str, ['source','destination','risk','s_name','d_name','s_latt','s_long','d_latt','d_long',
                                  's_country','d_country',]))+'\n')
        
        for source,val in p_AdjList.items():
            o_p_AdjList=sorted(p_AdjList[source].items(), key=lambda x: x[1],reverse=True) 
            
            for dest,dest_risk in o_p_AdjList:
                source=str(source);dest=str(dest)
                if i<3000:    
                    f.write(','.join(map(str, [source,dest,dest_risk,
                                                    ports[source]['NAME'],ports[dest]['NAME'],
                                                     ports[source]['LATITUDE_DECIMAL'],ports[source]['LONGITUDE_DECIMAL'],
                                                     ports[dest]['LATITUDE_DECIMAL'],ports[dest]['LONGITUDE_DECIMAL'],
                                                     ports[source]['COUNTRY_CODE'],ports[dest]['COUNTRY_CODE'],
                                                       ]))+'\n')
                i+=1

                
                    
    print("Links written")
                                                  
#Writing down dictinary and net for clustering
#########################################################


def write_dic_net(AdjList,all_ports,dic,HONet_translated):
    write_error=0
    PortsList=list(sorted(all_ports))
    translated=list(range(len(all_ports)))
    translation=zip(PortsList,translated)
    OldToNew= dict(zip(PortsList,translated))
    with open(dic,'w') as g:
        for (old,new) in translation:
            g.write('{} {}\n'.format(old,new))
    g.close()        
    print("Dictionary written")
    with open(HONet_translated,'w') as h:
        for froms in AdjList:
            for tos in AdjList[froms]:
                try:
                    h.write('{} {} {}\n'.format(OldToNew[froms],OldToNew[tos],AdjList[froms][tos]))
                except:
                    write_error+=1
    h.close()
    print(".net_translated written with ",write_error," errors")
def write_net(AdjList,net_file): #write a network with no translation or maping dict
    write_error=0
    with open(net_file,'w') as h:
        for froms in AdjList:
            for tos in AdjList[froms]:
                try:
                    h.write('{} {} {}\n'.format(froms,tos,AdjList[froms][tos]))
                except:
                    write_error+=1
    h.close()
    print(".net written with ",write_error," errors")


#Writing down agg risk of ports
#########################################################

def write_port_risks(agg_port_risk,agg_risks_file):
    import operator
    with open(agg_risks_file,'w') as p:
        for port in sorted(agg_port_risk.items(), key=operator.itemgetter(1),reverse=True):
            
            p.write('{0},{1} \n'.format(port[0],port[1]))   
            
    p.close()
    print("Port risks written")


##########################################################
#Auxilary functions
#########################################################
def load_port_risks(file,delim):
    lines=open(file).readlines()
    output_dict=defaultdict()
    for line in lines:
        port,risk=line.strip().split(delim)
        output_dict[int(port)]=float(risk)
    return output_dict


def risk_aggregator(adjlist):#{source:{dest1:r1,dest2:r2}}
                             #{source:agg_risk(r1,r2)}}   
    agg_dic=defaultdict()
    for FromPort,ToPort in adjlist.items():
        prod=1
        for dest,risk in ToPort.items():
            prod*=(1-risk)
        agg_dic[FromPort]=1-prod    

    return agg_dic

def risk_mean(adjlist):#{source:{dest1:r1,dest2:r2}}
                             #{source:mean_risk(r1,r2)}}       
    agg_dic=defaultdict()
    for FromPort,ToPort in adjlist.items():
        agg_dic[FromPort]=np.mean(list(adjlist[FromPort].values()))
    return agg_dic

def risk_sum(adjlist):#{source:{dest1:r1,dest2:r2}}
                             #{source:mean_risk(r1,r2)}}       
    agg_dic=defaultdict()
    for FromPort,ToPort in adjlist.items():
        agg_dic[FromPort]=np.sum(list(adjlist[FromPort].values()))
    return agg_dic

def list_aggregator_dic(adjlist):  #{source:{dest1:[r1,r2,r3],dest2:[r1,r2,r3]}}
                               # {source:{dest1:r1_agg,dest2:r2_agg}}   
    agg_dic=defaultdict(dict) 
    for FromPort,ToPort in adjlist.items():        
        for dest,risk in ToPort.items():
            prod=1
            for item in risk:
                prod*=(1-item) 
            agg_dic[FromPort][dest]=1-prod 
    return agg_dic

def list_mean_dic(adjlist):  #{source:{dest1:[r1,r2,r3],dest2:[r1,r2,r3]}}
                           # {source:{dest1:r_mean,dest2:r_mean}}
    mean_agg_dic=defaultdict(dict) 
    for FromPort,ToPort in adjlist.items():        
        for dest,risk in ToPort.items():
            mean_agg_dic[FromPort][dest]=np.mean(adjlist[FromPort][dest])
    return(mean_agg_dic)

def list_sum_dic(adjlist):  #{source:{dest1:[r1,r2,r3],dest2:[r1,r2,r3]}}
                           # {source:{dest1:sum ,dest2:sum}} #sum over all incoming HON coonections for a port
    sum_agg_dic=defaultdict(dict) 
    for FromPort,ToPort in adjlist.items():        
        for dest,risk in ToPort.items():
            sum_agg_dic[FromPort][dest]=np.sum(adjlist[FromPort][dest])
    return(sum_agg_dic)

def list_aggregator(adjlist):  #{source1:[r1,r2,r3],source2:[r1,r2,r3]}
                               # {source1:r1_agg, source2:r2_agg}  
    agg_dic=defaultdict() 
    for FromPort,risk in adjlist.items():        
            prod=1
            for item in risk:
                prod*=(1-item)
            agg_dic[FromPort]=1-prod

    return agg_dic

def list_mean(adjlist):  #{source1:[r1,r2,r3],source2:[r1,r2,r3]}}
                           # {source1:r1_mean, source2:r2_mean}
    mean_agg=defaultdict(dict) 
    for FromPort,risks in adjlist.items():        
            mean_agg[FromPort]=np.mean(risks)
    return(mean_agg)

##########################################################
#Adjacency list from HON---- Get Adjlist of region-Get Shipping net
#########################################################

def get_p_adjlist(HONet): #Input: HONet # average all risk corresponding to a destination over all incoming ports. 
                         #so for each source and dest, we have the corresponding risk.
    AdjList=defaultdict(dict)   
    all_ports=set({})
    # 2005.txt
    with open(HONet) as f:

        for line in f:
            FromPort=line.split(',')[0].strip()
            ToPort=line.split(',')[1].strip()
            if FromPort==ToPort:
                continue
            EdgeWeight=float(line.split(',')[2].strip())
            if EdgeWeight>0:
                all_ports.add(FromPort)
                all_ports.add(ToPort)
                AdjList[FromPort][ToPort]=EdgeWeight

    #Omit the '|' for each port. appending all risks for the phisical ports (source and dest)
    p_AdjList=defaultdict(lambda : defaultdict(list))
    for FromPort,ToPort in AdjList.items():
        p_source=int(FromPort.split('|')[0]) #phisical source
        for port,risk in ToPort.items():
            p_dest=int(port.split('|')[0]) #phisical dest
            p_AdjList[p_source][p_dest].append(risk)
            
    
    return p_AdjList,AdjList,all_ports

##########################################################
#Regional Analysis---- Get Adjlist of region-Get Shipping net
#########################################################

def region_year_info(region_year,region,adjlist,year):
    region_risk=region_adj_list(region,adjlist)
    region_aggrisk=risk_mean(region_risk)
    
    for key,val in region_aggrisk.items():
        region_year[year][key]=val
    return(region_year)

def region_adj_list(region,p_AdjList): #returns the adjacency matrix for region (avg for each region)
    c_AdjList=defaultdict(lambda : defaultdict(list))
    for FromPort,ToPort in p_AdjList.items():
        c_source=ports[str(FromPort)][region]
        for port,risk in ToPort.items():
            c_dest=ports[str(port)][region]
            c_AdjList[c_source][c_dest].append(risk)
    net_region_risk=list_mean_dic(c_AdjList) 
    return net_region_risk

def get_shipping_net(fon_weights):
    fon=open(fon_weights,'r')
    lines=fon.readlines()
    AdjList=defaultdict(dict)   

    for line in lines:
        source,target, weight,ship_type=line.strip().split(',')
        if (source==target):
            continue
        if target not in AdjList[source]:
            AdjList[source][target]=0
        AdjList[source][target]+=1
    return AdjList

def plot_dist(agg_dic,param,title,xlab,ylab,path,bins):
    #%matplotlib inline  
    import matplotlib
    if(param=='Ballast'):
        t='r'
    else:
        t='b'
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.hist(list(agg_dic.values()),color=t,log=True,bins=bins)
    plt.title("Distribution of aggregated HON "+param+" risk") #like, freq=20 and value=100: 50 nodes belong to 100 clusters
    plt.xlabel("Aggregated risk")
    plt.ylabel("Frequency")
    plt.savefig("data/"+y+"figs/Distribution of aggregated HON "+param+" risk",dpi=200)
    plt.show()
    
    
##########################################################
### Risk based on the shipping network
### HON and FON using the output of HON algorithm (traces file required for previous step)
### Paul stuff
#########################################################

def normalize_network(AdjList,max_val):
    normalized_AdjList= defaultdict(lambda: defaultdict())
    for source,incomings in AdjList.items():
        #sum_source=sum(incomings.values())
        for target, val in incomings.items():
            normalized_AdjList[source][target]=val/max_val
    return normalized_AdjList
def get_max_val_in_adjM(adjlist):
    vals=[]
    for source, incomings in adjlist.items():
        vals.extend(list(incomings.values()))
    return max(vals)
def normalize_agg_port_risks(agg_port_risks):
    Max=max(agg_port_risks.values())
    Min=min(agg_port_risks.values())
    print(Max)
    for port,risk in agg_port_risks.items():
        agg_port_risks[port]=(risk-Min)/(Max-Min)
    return agg_port_risks

def build_shipping_net_and_port_risk(hon_alg_output,write,net_file,agg_risk_file):
    p_AdjList,AdjList,all_ports=get_p_adjlist(hon_alg_output)
    print(len(p_AdjList[1640]))
    p_AdjList=list_sum_dic(p_AdjList)
    max_val=get_max_val_in_adjM(p_AdjList)
    print('max_val: ',max_val)
    #max_val=5.718045873213033e-05
    max_val=36.510086825975364
    p_AdjList_norm=normalize_network(p_AdjList,max_val)
    #print(len(p_AdjList_norm[1640]))
    agg_port_risk=risk_aggregator(p_AdjList_norm)
    #agg_port_risk=normalize_agg_port_risks(agg_risk) #Aggregate all risks for a port (over all incoming ports)


    if write:
        #Write adjacency list
        write_net(p_AdjList_norm,net_file)
        #Write aggregated port list
        write_port_risks(agg_port_risk,agg_risk_file)
    return p_AdjList_norm,agg_port_risk,all_ports

def load_adj_net(file,delim,ebunch=None):
    lines=open(file).readlines()
    output_dict=defaultdict(lambda: defaultdict())
    
    for line in lines:
        source,target,risk=line.strip().split(delim)
        output_dict[int(source)][int(target)]=float(risk)
    if ebunch:
        jacc_net=get_jaccard_scores(file,ebunch)
        return output_dict,jacc_net
    else:
        return output_dict
    
def get_jaccard_scores(input_net,ebunch):

    output_scores=defaultdict(lambda: defaultdict())
    net=nx.read_weighted_edgelist(input_net)
    preds=nx.jaccard_coefficient(net, ebunch)
    for u, v, p in preds:
        output_scores[int(u)][int(v)]=p
        output_scores[int(v)][int(u)]=p
    return output_scores

selected={1165:'SY',3110:'AD', 854:'BT',2503:'HN', 2331:'HT',7597:'LB', 4899:'MI',576:'AW',
        2729:'BA' ,2141:'CB',3367:'HS',3108:'NA', 3381:'NO', 7598:'OK',238:'PL', 193:'PM',
        4777:'RC',830:'RT',311:'VN',4538:'GH',7975:'WL',1675:'ZB'}
def write_all_links_paul(year,links_all_file):

    #ebunch=list(itertools.combinations([str(i) for i in selected.keys()], 2))    
    ebunch=list(itertools.product([str(i) for i in selected.keys()],repeat= 2))    
    y=year+'/'
    shipping_FON_freq='data/'+y+'freq_net_'+year+'.txt'
    ballast_FON_noEco='data/'+y+'l_FONet_Ballast_env_noEco_'+str(year)+'_1.net'
    ballast_HON_noEco='data/'+y+'hon_ballast_net_noEco_'+year+'.txt'
    ballast_FON_sameEco='data/'+y+'l_FONet_Ballast_env_sameEco_'+str(year)+'_1.net'
    ballast_HON_sameEco='data/'+y+'hon_ballast_net_sameEco_'+year+'.txt'
    ballast_FON_noEco_noEnv='data/'+y+'l_FONet_Ballast_noEnv_noEco_'+str(year)+'_1.net'
    ballast_HON_noEco_noEnv='data/'+y+'hon_ballast_net_noEco_noEnv_'+year+'.txt'
    
    
    fouling_FON_noEco='data/'+y+'l_FONet_Fouling_env_noEco_'+str(year)+'_1.net'
    fouling_HON_noEco='data/'+y+'hon_fouling_net_noEco_'+year+'.txt'
    fouling_FON_sameEco='data/'+y+'l_FONet_Fouling_env_sameEco_'+str(year)+'_1.net'
    fouling_HON_sameEco='data/'+y+'hon_fouling_net_sameEco_'+year+'.txt'
    fouling_FON_noEco_noEnv='data/'+y+'l_FONet_Fouling_noEnv_noEco_'+str(year)+'_1.net'
    fouling_HON_noEco_noEnv='data/'+y+'hon_fouling_net_noEco_noEnv_'+year+'.txt'
    
    
    shipping_fon_freq,J0=load_adj_net(shipping_FON_freq,' ',ebunch)
    ballast_fon_noEco,J1=load_adj_net(ballast_FON_noEco,' ',ebunch)
    ballast_hon_noEco,J2=load_adj_net(ballast_HON_noEco,' ',ebunch)
    ballast_fon_sameEco,J3=load_adj_net(ballast_FON_sameEco,' ',ebunch)
    ballast_hon_sameEco,J4=load_adj_net(ballast_HON_sameEco,' ',ebunch)
    ballast_hon_noEco_noEnv,J5=load_adj_net(ballast_HON_noEco_noEnv,' ',ebunch)
    ballast_fon_noEco_noEnv,J6=load_adj_net(ballast_FON_noEco_noEnv,' ',ebunch)
    
    fouling_fon_noEco,J7=load_adj_net(fouling_FON_noEco,' ',ebunch)
    fouling_hon_noEco,J8=load_adj_net(fouling_HON_noEco,' ',ebunch)
    fouling_fon_sameEco,J9=load_adj_net(fouling_FON_sameEco,' ',ebunch)
    fouling_hon_sameEco,J10=load_adj_net(fouling_HON_sameEco,' ',ebunch)
    fouling_hon_noEco_noEnv,J11=load_adj_net(fouling_HON_noEco_noEnv,' ',ebunch)
    fouling_fon_noEco_noEnv,J12=load_adj_net(fouling_FON_noEco_noEnv,' ',ebunch)

    
    all_nets=[shipping_fon_freq,
              ballast_fon_noEco,ballast_hon_noEco,ballast_fon_sameEco,ballast_hon_sameEco,
              ballast_fon_noEco_noEnv,ballast_hon_noEco_noEnv,
              fouling_fon_noEco,fouling_hon_noEco,fouling_fon_sameEco,fouling_hon_sameEco,
              fouling_fon_noEco_noEnv,fouling_hon_noEco_noEnv]
    print("Number of models: ",len(all_nets))
    with open(links_all_file, 'w') as f:
        f.write(','.join(['source','target','voyage_freq',
                          'Ballast FON noEco','Ballast HON noEco',
                          'Ballast FON sameEco','Ballast HON sameEco',
                          'Ballast FON noEco_noEnv','Ballast HON noEco_noEnv',
                          'Fouling FON noEco','Fouling HON noEco',
                          'Fouling FON sameEco','Fouling HON sameEco',
                          'Fouling FON noEco_noEnv','Fouling HON noEco_noEnv',
                          'J_voyage_freq',
                          'J_Ballast FON noEco','J_Ballast HON noEco',
                          'J_Ballast FON sameEco','J_Ballast HON sameEco',
                          'J_Ballast FON noEco_noEnv','J_Ballast HON noEco_noEnv',
                          'J_Fouling FON noEco','J_Fouling HON noEco',
                          'J_Fouling FON sameEco','J_Fouling HON sameEco',
                          'J_Fouling FON noEco_noEnv','J_Fouling HON noEco_noEnv'
                        ])+'\n')
        
        max_vals=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        #combs=list(itertools.combinations(selected.keys(), 2))
            
        for s,t in ebunch:    
            s=int(s);t=int(t)
            if s!=t:
                for i,net in enumerate(all_nets):
                    net[s][t]=net.get(s,0).get(t,0)
                    max_vals[i]=max(max_vals[i],net[s][t]) 
        
        
          
        print(max_vals)
        for s,t in ebunch:
            s=int(s);t=int(t)
            if s!=t:
                f.write(','.join(map(str, [selected[s],selected[t],shipping_fon_freq[s][t],
                                                ballast_fon_noEco[s][t]/max_vals[1],
                                                ballast_hon_noEco[s][t]/max_vals[2],
                                                ballast_fon_sameEco[s][t]/max_vals[3],
                                                ballast_hon_sameEco[s][t]/max_vals[4],
                                                ballast_fon_noEco_noEnv[s][t]/max_vals[5],
                                                ballast_hon_noEco_noEnv[s][t]/max_vals[6],
                                                fouling_fon_noEco[s][t]/max_vals[7],
                                                fouling_hon_noEco[s][t]/max_vals[8],
                                                fouling_fon_sameEco[s][t]/max_vals[9],
                                                fouling_hon_sameEco[s][t]/max_vals[10],
                                                fouling_fon_noEco_noEnv[s][t]/max_vals[11],
                                                fouling_hon_noEco_noEnv[s][t]/max_vals[12],
                                                
                                                J0[s][t], J1[s][t], J2[s][t], J3[s][t], J4[s][t], 
                                                J5[s][t], J6[s][t], J7[s][t], J8[s][t], J9[s][t], 
                                                J10[s][t], J11[s][t], J12[s][t]]))+'\n')
        all_nets_updated=[shipping_fon_freq,
              ballast_fon_noEco,ballast_hon_noEco,ballast_fon_sameEco,ballast_hon_sameEco,
              ballast_fon_noEco_noEnv,ballast_hon_noEco_noEnv,
              fouling_fon_noEco,fouling_hon_noEco,fouling_fon_sameEco,fouling_hon_sameEco,
              fouling_fon_noEco_noEnv,fouling_hon_noEco_noEnv]
        return all_nets_updated
                    



def write_hon_links_paul(HON_links,agg_risk,p_hon_AdjList,fon_AdjList,ports):
    i=0
    with open(HON_links, 'w') as f:
        for source,r in agg_risk.items():
            o_p_AdjList=sorted(p_hon_AdjList[source].items(), key=lambda x: x[1],reverse=True) 
            for dest,hon_dest_risk in o_p_AdjList:
                if int(source) in selected.keys() and dest in selected.keys():
                    i+=1                
                    try:
                        fon_dest_risk=fon_AdjList[int(source)][int(dest)]
                    except:
                        fon_dest_risk=0
                    f.write(','.join(map(str, [selected[int(source)],selected[int(dest)],hon_dest_risk,fon_dest_risk]))+'\n')
    print(i," links written") 