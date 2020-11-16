
import operator
import functools
from collections import defaultdict
import csv
from collections import defaultdict, Counter
import math
import numpy as np

###########################################
# Functions
###########################################


Verbose=True
def ReadSequentialData(InputFileName,InputFileDeliminator, MinimumLengthForTraining, LastStepsHoldOutForTesting):
    if Verbose:
        print('Reading raw sequential data')
    RawTrajectories = []
    with open(InputFileName) as f:
        LoopCounter = 0
        for line in f:
            fields = line.strip().split(InputFileDeliminator)
            #ship = fields[0]  #Modified for no ship data
            movements = fields[0:-1]
            prob=fields[-1]
            

            LoopCounter += 1
            #if LoopCounter % 10000 == 0:
             #   VPrint(LoopCounter)
            ## Other preprocessing or metadata processing can be added here
            ## Test for movement length
            MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
            if len(movements) < MinMovementLength:
                continue

            RawTrajectories.append([movements, prob]) #Modified for no ship data
    return RawTrajectories


def BuildTrainingAndTesting(RawTrajectories, LastStepsHoldOutForTesting):
    VPrint('Building training and testing')
    Training = []
    Testing = []
    Prob= []
    for trajectory in RawTrajectories:
        #Modified for no ship data
        #ship, movement,prob = trajectory
        #Training.append([ship, movement[:-LastStepsHoldOutForTesting],prob])
        #Testing.append([ship, movement[-LastStepsHoldOutForTesting],prob])

        movement,prob = trajectory
        Training.append([movement[:-LastStepsHoldOutForTesting],prob])
        Testing.append([movement[-LastStepsHoldOutForTesting],prob])
    return Training, Testing

def DumpRules(Rules, OutputRulesFile):
    VPrint('Dumping rules to file')
    with open(OutputRulesFile, 'w') as f:
        for Source in Rules:
            for Target in Rules[Source]:
                f.write(' '.join([' '.join([str(x) for x in Source]), '=>', Target, str(Rules[Source][Target])]) + '\n')

def SequenceToNode(seq):
    curr = seq[-1]
    node = curr + '|'
    seq = seq[:-1]
    while len(seq) > 0:
        curr = seq[-1]
        node = node + curr + '.'
        seq = seq[:-1]
    if node[-1] == '.':
        return node[:-1]
    else:
        return node

def VPrint(string):
    if Verbose:
        print(string)

     
        
def KLD(a, b):
    divergence = 0
    for target in a:
        try:
            divergence += GetProbability(a, target) * math.exp(GetProbability(a, target)/GetProbability(b, target))
        except OverflowError:
            divergence = float('inf')
    return divergence


def GetProbability(d, key):
    str_key=str(key)
    if not str_key in d:
        return 1e-6
    else:
        return d[str_key]
def DumpNetwork(Network, OutputNetworkFile):
    VPrint('Dumping network to file')
    with open(OutputNetworkFile, 'w') as f:
        for source in Network:
            for target in Network[source]:
                f.write(','.join([SequenceToNode(source), SequenceToNode(target), str(Network[source][target])]) + '\n')  

###########################################
# Auxiliary functions
###########################################

def VPrint(string):
    if Verbose:
        print(string)
        
        
###########################################
# Others 
###########################################        
def dependecy_dic(Network,attr):
    print("Number of edges: ",len(Network))
    n_orders=[]; dependency=defaultdict(dict)
    for source, vals in Network.items():
        for target,weight in vals.items():
            order=max(len(source),len(target))
            dependency[order][(source,target)]=weight
    print("Largest order: ",len(dependency.keys()))


    for key in dependency.keys():
        print("Order Size of",key,": ",len(dependency[key]))
        n_orders.append(len(dependency[key]))
                        
        sorted_key = sorted(dependency[key].items(), key=operator.itemgetter(1),reverse=True)
        for item in sorted_key[0:10]:
            if(item[1]>0.8):
                print([ports[x][attr] for x in item[0][0]], '=>', [ports[x][attr] for x in item[0][1]],item[1])
    return(n_orders)




