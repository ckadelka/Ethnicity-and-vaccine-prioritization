#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2022

@author: ckadelka
"""



#built-in modules
import sys
import random
import os

#added modules
import numpy as np
import networkx as nx
import itertools
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib
import scipy.signal as signal
import main_model_v42 as source
from matplotlib import cm

#see  https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
from scipy import spatial
from functools import reduce

def filter_(pts, pt):
    """
    Get all points in pts that are not Pareto dominated by the point pt
    """
    weakly_worse   = (pts >= pt).all(axis=-1)
    strictly_worse = (pts > pt).any(axis=-1)
    return pts[~(weakly_worse & strictly_worse)]


def get_pareto_undominated_by(pts1, pts2=None):
    """
    Return all points in pts1 that are not Pareto dominated
    by any points in pts2
    """
    if pts2 is None:
        pts2 = pts1
    return reduce(filter_, pts2, pts1)


def get_pareto_frontier(pts):
    """
    Iteratively filter points based on the convex hull heuristic
    """
    pareto_groups = []

    # loop while there are points remaining
    while pts.shape[0]:
        # brute force if there are few points:
        if pts.shape[0] < 10:
            pareto_groups.append(get_pareto_undominated_by(pts))
            break

        # compute vertices of the convex hull
        try:
            hull_vertices = spatial.ConvexHull(pts).vertices
        except:
            return np.vstack(pareto_groups)
        # get corresponding points
        hull_pts = pts[hull_vertices]

        # get points in pts that are not convex hull vertices
        nonhull_mask = np.ones(pts.shape[0], dtype=bool)
        nonhull_mask[hull_vertices] = False
        pts = pts[nonhull_mask]

        # get points in the convex hull that are on the Pareto frontier
        pareto   = get_pareto_undominated_by(hull_pts)
        pareto_groups.append(pareto)

        # filter remaining points to keep those not dominated by
        # Pareto points of the convex hull
        pts = get_pareto_undominated_by(pts, pareto)

    return np.vstack(pareto_groups)

# --------------------------------------------------------------------------------
# previous solutions
# --------------------------------------------------------------------------------

def is_pareto_efficient_dumb(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs>=c, axis=1))
    return is_efficient


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def is_pareto_efficient_2d(costs1,costs2,RETURN_INDICES=False):
    """
    :two costs: two (n_points) arrays
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    assume no ties in costs
    """
    indices_sorted_by_costs1 = np.argsort(costs1)
    costs1=costs1[indices_sorted_by_costs1]
    costs2=costs2[indices_sorted_by_costs1]
    min_costs1 = costs1[0]
    min_costs2 = costs2[0]
    pareto = [[min_costs1,min_costs2]]
    indices = [indices_sorted_by_costs1[0]]
    for i, c in enumerate(costs2[1:]):
        if c < min_costs2:
            min_costs2 = c
            pareto.append([costs1[i+1],c])
            indices.append(indices_sorted_by_costs1[i+1])
    if RETURN_INDICES:
        return pareto,indices
    else:
        return pareto
    
def is_pareto_efficient_3d(costs1,costs2,costs3,RETURN_INDICES=False):
    """
    :three costs: two (n_points) arrays
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    assume no ties in costs
    """
    indices_sorted_by_costs1 = np.argsort(costs1)
    costs1=costs1[indices_sorted_by_costs1]
    costs2=costs2[indices_sorted_by_costs1]
    costs3=costs3[indices_sorted_by_costs1]
    min_costs1 = costs1[0]
    min_costs2 = costs2[0]
    min_costs3 = costs3[0]
    pareto = [[min_costs1,min_costs2,min_costs3]]
    indices = [indices_sorted_by_costs1[0]]
    for i, (c2,c3) in enumerate(zip(costs2[1:],costs3[1:])):
        NEW_OPTIMAL=False
        if c2 < min_costs2:
            min_costs2 = c2
            NEW_OPTIMAL=True
        if c3 < min_costs3:
            min_costs3 = c3
            NEW_OPTIMAL=True
        if NEW_OPTIMAL:
            pareto.append([costs1[i+1],c2,c3])
            indices.append(indices_sorted_by_costs1[i+1])
    if RETURN_INDICES:
        return pareto,indices
    else:
        return pareto
       
def sort_convex_hull_points(points):
    argmin = np.argmin(points[:,0])
    ordered = np.array([points[argmin]])
    lower_points = points[points[:,1]<points[argmin,1],:]
    ordered = np.r_[ordered, lower_points[np.argsort(lower_points[:,0]),:] ]
    higher_points = points[points[:,1]>=points[argmin,1],:]
    ordered = np.r_[ordered, higher_points[np.argsort(higher_points[:,0]),:][::-1,:] ]
    return ordered


def dominates(row, rowCandidate):
    return all(r >= rc for r, rc in zip(row, rowCandidate))


def cull(pts, dominates):
    dominated = []
    cleared = []
    remaining = pts
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated][dominates(candidate, other)].append(other)
        if not any(dominates(other, candidate) for other in new_remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)
        remaining = new_remaining
    return cleared, dominated

def keep_efficient(pts):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    pts = pts[pts.sum(1).argsort()[::-1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = (pts[i+1:] < pts[i]).any(1) 
        # keep points undominated so far
        pts = pts[undominated[:n]]
    return pts


plt.rcParams.update({'font.size': 10})
matplotlib.rcParams['text.usetex'] = False

version='43'
infix = 'version'+version

folder = '~results/%s/' % infix

nsim = 148234

nr_scenarios = 8

deathss=[]
casess=[]
infectionss=[]
entropiess=[]
yllss=[]

deathss_black = []
deathss_white = []
casess_black = []
casess_white = []
infectionss_black = []
infectionss_white = []

allocation_idss = []
all_slurm_ids = []
id_scenarios = []
for scen in range(nr_scenarios):
    homophily_ethnicity = [0.8,0][scen//4] #,1
    multipler_highcontact_jobs = [3.,1.][(scen%4)//2]
    relative_difference_in_high_contact_jobs_wrt_ethnicity = [3.,1.][scen%2] #,10 #if <1, there are proportionately more WorA in high-risk jobs, otherwise more non-W and non-A
    infix = 'version'+version
    id_scenario = 'homophily%i_multiplier%i_reldiff%i' % (int(homophily_ethnicity*100),int(multipler_highcontact_jobs*100),int(relative_difference_in_high_contact_jobs_wrt_ethnicity*100))
    infix+='_'+id_scenario
    id_scenarios.append(id_scenario)
    
    deaths_per_age=[[] for _ in range(4)]
    cases_per_age=[[] for _ in range(4)]
    infections_per_age=[[] for _ in range(4)]
    deaths_per_ethnicity=[[] for _ in range(2)]
    cases_per_ethnicity=[[] for _ in range(2)]
    infections_per_ethnicity=[[] for _ in range(2)]
    
    max_infections=[]
    ylls = []
    allocation_ids = []
    time_in_seconds = []
    counter = 0
    slurm_ids = []
    for fname in os.listdir(folder):
        if fname.endswith('%s.txt' % id_scenario) and 'nsim%i' % nsim in fname:
            SLURM_ID=int(fname.split('_')[3][2:])
            #if SLURM_ID>0:#delete at end
            #    continue
            slurm_ids.append(SLURM_ID)
            #print(fname)
            f = open(folder+fname,'r')
            textsplit = f.read().splitlines()
            f.close()
            counter_d=0
            counter_c=0
            counter_i=0
            if counter==0:
                filename = fname.split('_seed')[0]
                dict_fixed_parameters = {}
            for line in textsplit:
                line_split = line.split('\t')
                if line_split[0] == 'time in seconds':
                    time_in_seconds.append(int(line_split[1]))
                if 'deaths_in_age_group' in line_split[0]:
                    deaths_per_age[counter_d].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_d+=1
                if 'cases_in_age_group' in line_split[0]:
                    cases_per_age[counter_c].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_c+=1
                if 'infections_in_age_group' in line_split[0]:
                    infections_per_age[counter_i].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_i+=1
                if 'deaths_among' in line_split[0]:
                    deaths_per_ethnicity[counter_d-4].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_d+=1  
                if 'cases_among' in line_split[0]:
                    cases_per_ethnicity[counter_c-4].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_c+=1  
                if 'infections_among' in line_split[0]:
                    infections_per_ethnicity[counter_i-4].extend(list(map(int,map(float,line_split[1:])))) 
                    counter_i+=1  
                elif line_split[0] == 'allocation ID':
                    allocation_ids.extend(list(map(int,line_split[1:]))) 
                elif len(line_split)==2:
                    if counter==0:
                        dict_fixed_parameters.update({line_split[0]:line_split[1]})
            counter+=1
            
    all_slurm_ids.extend(slurm_ids)
    allocation_ids = np.array(allocation_ids)
    indices = np.argsort(allocation_ids)
    deaths_per_age = np.array(deaths_per_age)
    cases_per_age = np.array(cases_per_age)
    infections_per_age = np.array(infections_per_age)

    deaths_per_age = deaths_per_age[:,indices]
    cases_per_age = cases_per_age[:,indices]
    infections_per_age = infections_per_age[:,indices]
    allocation_ids = allocation_ids[indices]
    
    deaths_per_ethnicity = np.array(deaths_per_ethnicity)[:,indices]
    cases_per_ethnicity = np.array(cases_per_ethnicity)[:,indices]
    infections_per_ethnicity = np.array(infections_per_ethnicity)[:,indices]
    
    #get total deaths
    total_deaths = np.sum(deaths_per_age,0)
    
    #get yll
    average_years_of_life_left = np.array([71.45519038, 41.66010998, 16.82498872,  7.77149779])
    ylls = np.dot(average_years_of_life_left,deaths_per_age)
    
    #get total cases
    total_cases = np.sum(cases_per_age,0)
    
    #get total infections
    total_infections = np.sum(infections_per_age,0)
    
    #equitable
    total_deaths_prop = np.multiply(deaths_per_age,1/total_deaths)
    entropies = np.sum(np.multiply(total_deaths_prop,np.log(total_deaths_prop)),0)

    deathss_black.append(deaths_per_ethnicity[0])
    deathss_white.append(deaths_per_ethnicity[1])    
    casess_black.append(cases_per_ethnicity[0])
    casess_white.append(cases_per_ethnicity[1])
    infectionss_black.append(infections_per_ethnicity[0])
    infectionss_white.append(infections_per_ethnicity[1])
    
    deathss.append(total_deaths)
    casess.append(total_cases)
    infectionss.append(total_infections)
    entropiess.append(entropies*1000)
    yllss.append(ylls)
    allocation_idss.append(allocation_ids)


number_of_phases = np.array(list(map(max,source.all_vacopts)))+1
different_numbers_of_phases = list(set(number_of_phases))
different_numbers_of_phases.sort()
number_of_different_numbers_of_phases = len(different_numbers_of_phases)
indices_specific_phase = [np.where(number_of_phases==x)[0] for x in different_numbers_of_phases]



def differentiates_by_ethnicity(allocation):
    '''determines for each allocation if it there is at least one population group
    where the two ethnic groups do not get vaccinated in the same phase, in which
    case it returns True. Otherwise, False.'''
    if allocation[0]!=allocation[1]:
        return True
    if allocation[2]!=allocation[4]:
        return True
    if allocation[3]!=allocation[5]:
        return True
    if allocation[6]!=allocation[7]:
        return True
    if allocation[8]!=allocation[9]:
        return True
    return False

def vaccinates_all_POC_before_any_WA(allocation):
    '''determines for each allocation if it there is at least one population group
    where the two ethnic groups do not get vaccinated in the same phase, in which
    case it returns True. Otherwise, False.'''
    if max(np.array(allocation)[source.race_class==0]) < min(np.array(allocation)[source.race_class==1]):
        return True
    else:
        return False
    
def vaccinates_all_WA_before_any_POC(allocation):
    '''determines for each allocation if it there is at least one population group
    where the two ethnic groups do not get vaccinated in the same phase, in which
    case it returns True. Otherwise, False.'''
    if max(np.array(allocation)[source.race_class==1]) < min(np.array(allocation)[source.race_class==0]):
        return True
    else:
        return False

allocation_differentiates_by_ethnicities = np.array(list(map(int,map(differentiates_by_ethnicity,source.all_vacopts))))
allocation_vaccinates_all_POC_before_any_WA = np.array(list(map(int,map(vaccinates_all_POC_before_any_WA,source.all_vacopts))))
allocation_vaccinates_all_WA_before_any_POC = np.array(list(map(int,map(vaccinates_all_WA_before_any_POC,source.all_vacopts))))


#record the optimal allocations
types= ['deaths','cases','infections','yll','entropy','deaths POC','deaths WA','cases POC','cases WA']
n_types = len(types)

optimal_allocationss = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4,source.Ngp))
optimal_idss = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
optimal_values = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
mean_values = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4,source.Ngp))
deaths_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
cases_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
infections_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
ylls_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
entropies_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
deaths_black_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
deaths_white_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
cases_black_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))
cases_white_optimal = np.zeros((nr_scenarios,n_types,number_of_different_numbers_of_phases,4))

f,ax=plt.subplots()
for i in range(20):
    ax.plot(range(148234*i,148234*i+148234),deathss[0][148234*i:148234*i+148234])

for j in range(nr_scenarios):
    for i,vec in enumerate([deathss[j],casess[j],infectionss[j],yllss[j],entropiess[j],deathss_black[j],deathss_white[j],casess_black[j],casess_white[j]]):
        for k in range(len(indices_specific_phase)):
            for IND in range(4):
                indices = indices_specific_phase[k]
                lowest_index = indices[0]
                if k==5 and IND==1:
                    continue
                if k==0 and IND>1:
                    continue
                if IND==1:
                    indices_not_distinguishing_by_ethnicity = np.where(allocation_differentiates_by_ethnicities[indices]==0)[0]
                    indices = indices[indices_not_distinguishing_by_ethnicity]
                    dummy_allocations = allocation_ids[lowest_index + indices_not_distinguishing_by_ethnicity]
                    OPTIMAL_ID = dummy_allocations[np.argmin(vec[indices])]
                elif IND==2:
                    indices_allocation_vaccinates_all_POC_before_any_WA = np.where(allocation_vaccinates_all_POC_before_any_WA[indices]==1)[0]
                    indices = indices[indices_allocation_vaccinates_all_POC_before_any_WA]
                    dummy_allocations = allocation_ids[lowest_index + indices_allocation_vaccinates_all_POC_before_any_WA]
                    OPTIMAL_ID = dummy_allocations[np.argmin(vec[indices])]                    
                elif IND==3:
                    indices_allocation_vaccinates_all_WA_before_any_POC = np.where(allocation_vaccinates_all_WA_before_any_POC[indices]==1)[0]
                    indices = indices[indices_allocation_vaccinates_all_WA_before_any_POC]
                    dummy_allocations = allocation_ids[lowest_index + indices_allocation_vaccinates_all_WA_before_any_POC]
                    OPTIMAL_ID = dummy_allocations[np.argmin(vec[indices])]  
                else:
                    OPTIMAL_ID = lowest_index + np.argmin(vec[indices])
                optimal_idss[j,i,k,IND] = allocation_ids[OPTIMAL_ID]
                optimal_allocationss[j,i,k,IND] = source.all_vacopts[OPTIMAL_ID]
                optimal_values[j,i,k,IND] = vec[OPTIMAL_ID]
                mean_values[j,i,k,IND] = np.mean(vec[indices])
                deaths_optimal[j,i,k,IND] = deathss[j][OPTIMAL_ID]
                cases_optimal[j,i,k,IND] = casess[j][OPTIMAL_ID]
                infections_optimal[j,i,k,IND] = infectionss[j][OPTIMAL_ID]
                ylls_optimal[j,i,k,IND] = yllss[j][OPTIMAL_ID]
                entropies_optimal[j,i,k,IND] = entropiess[j][OPTIMAL_ID]
                deaths_black_optimal[j,i,k,IND] = deathss_black[j][OPTIMAL_ID]
                deaths_white_optimal[j,i,k,IND] = deathss_white[j][OPTIMAL_ID]
                cases_black_optimal[j,i,k,IND] = casess_black[j][OPTIMAL_ID]
                cases_white_optimal[j,i,k,IND] = casess_white[j][OPTIMAL_ID]



ending = '.pdf'
infix_short = 'version%s' % version

#how much better do we do with more phases
which_scenarios = np.arange(8)#np.array([0,3,4,7])
which_scenarios = np.array([0,1,3,4,5,7])

for id_label,label,vec in zip([0,1],['deaths','cases'],[deaths_optimal,cases_optimal]):
    data = vec[which_scenarios,id_label,:,0]
    f,ax = plt.subplots()
    for i in range(len(which_scenarios)):
        ax.plot(data[i])
    ax.set_xticks(range(number_of_different_numbers_of_phases))
    ax.set_xticklabels(list(map(str,different_numbers_of_phases)))
    ax.set_ylabel('predicted minimal '+label)
    ax.set_xlabel('exact number of phases in allocation strategy')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('%s_improvement_with_more_phases_%s%s' % (infix_short,label,ending),bbox_inches = "tight")


    #doesn't work nicely
    data_mins = np.min(vec[which_scenarios,id_label,:,0],1)
    dummy = []
    dummy2 = []
    for i in range(len(which_scenarios)):
        dummy.append(data[i]/data_mins[i])
        dummy2.append(data[i]-data_mins[i])
        #dummy.append(np.log2(data[i]/data_mins[i]))
        #dummy.append(100*(data[i]-data_mins[i])/data_mins[i])
    dummy = np.array(dummy)
    f,ax = plt.subplots()
    to_plot = np.log(dummy)#[:,1:]
    #ax.imshow(to_plot,vmin=0,vmax=np.max(to_plot),cmap=cm.Greens) 
    ax.imshow(to_plot,cmap=cm.Greens)
    
    #creating Table 1
    A = pd.DataFrame((dummy-1),columns = list(map(str,different_numbers_of_phases)),index = np.array(id_scenarios)[which_scenarios])
    A.to_csv('%s_optimal_%s_different_phase_numbers_and_scenarios.csv' % (infix_short,label))

    A = pd.DataFrame(dummy2,columns = list(map(str,different_numbers_of_phases)),index = np.array(id_scenarios)[which_scenarios])
    A.to_csv('%s_optimal_%s_absolute_different_phase_numbers_and_scenarios.csv' % (infix_short,label))
    
    
#optimal allocations under different optimal strategies
indices_objectives = np.array([0,1,5,6,7,8])
index_scenario=0
index_phases = 5
optimal_numbers = []
for vec,factor in zip([deaths_optimal,cases_optimal,deaths_black_optimal,deaths_white_optimal,cases_black_optimal,cases_white_optimal],[3,6,3,3,6,6]):
    optimal_numbers.append(vec[index_scenario,indices_objectives,index_phases,0]/10**factor)
optimal_numbers = np.array(optimal_numbers)
proportions = np.divide(np.array(optimal_numbers).T,np.min(optimal_numbers,1)).T-1
#A = pd.DataFrame(1+optimal_allocationss[index_scenario,indices_objectives,index_phases,0,:].T,columns=np.array(types)[indices_objectives])
#A.to_csv('%s_optimal_allocations_scenario%i_%iphases.csv' % (infix_short,index_scenario,different_numbers_of_phases[index_phases]))

B = pd.DataFrame(np.r_[1+optimal_allocationss[index_scenario,indices_objectives,index_phases,0,:].T,optimal_numbers,proportions],columns=np.array(types)[indices_objectives])
B.to_csv('%s_optimal_allocations_scenario%i_%iphases.csv' % (infix_short,index_scenario,different_numbers_of_phases[index_phases]))
 
    
which_scenario = 0#np.array([0,3,4,7])
for id_label,label,vec in zip([0,1],['deaths','cases'],[deathss,casess]):
    f,ax = plt.subplots()
    for k,indices in enumerate(indices_specific_phase):
        data = vec[which_scenario][indices]
        ax.boxplot(data,positions=[k])
        ax.set_xticks(range(number_of_different_numbers_of_phases))
        ax.set_xticklabels(list(map(str,different_numbers_of_phases)))
        ax.set_xlabel('exact number of phases in allocation strategy')
        

#How much better do we do considering ethnicity
dummy = deaths_optimal[which_scenarios,0,1:5,1] - deaths_optimal[which_scenarios,0,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'deaths'))

#How much better do we do considering ethnicity
dummy = cases_optimal[which_scenarios,1,1:5,1] - cases_optimal[which_scenarios,1,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'cases'))

#How much better do we do considering ethnicity
dummy = deaths_black_optimal[which_scenarios,5,1:5,1] - deaths_black_optimal[which_scenarios,5,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'deaths_black'))

#How much better do we do considering ethnicity
dummy = deaths_white_optimal[which_scenarios,6,1:5,1] - deaths_white_optimal[which_scenarios,6,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'deaths_white'))

#How much better do we do considering ethnicity
dummy = cases_black_optimal[which_scenarios,7,1:5,1] - cases_black_optimal[which_scenarios,7,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'cases_black'))

#How much better do we do considering ethnicity
dummy = cases_white_optimal[which_scenarios,8,1:5,1] - cases_white_optimal[which_scenarios,8,1:5,0]
A = pd.DataFrame(dummy,columns = list(map(str,different_numbers_of_phases[1:-1])),index = np.array(id_scenarios)[which_scenarios])
A.to_csv('%s_optimal_%s_difference_eth_vs_noeth_different_phase_numbers_and_scenarios.csv' % (infix_short,'cases_white'))











        
        
x_vec = deathss
y_vec = casess
x_vec_optimal = deaths_optimal
y_vec_optimal = cases_optimal
x_title = 'deaths'
y_title = 'cases'

# x_vec = deathss_black
# y_vec = deathss_white
# x_vec_optimal = deaths_black_optimal
# y_vec_optimal = deaths_white_optimal
# x_title = 'deaths_black'
# y_title = 'deaths_white'

indices_scenarios = [0,3,7]
indices_phases = list(range(1,6))

types_legend = types
colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(8),cm.tab10(7),cm.tab10(6),cm.tab10(5)]
for i in indices_scenarios:#range(nr_scenarios):        
    f,ax = plt.subplots(figsize=(3,3))
    for ii in indices_phases:
        dummy = is_pareto_efficient_2d(x_vec[i][indices_specific_phase[ii]],y_vec[i][indices_specific_phase[ii]],True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        ax.plot(pareto[:,0],pareto[:,1],'-',color=colors[ii],label=str(different_numbers_of_phases[ii]))
        #counter=0
        #ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],label='CDC',zorder=3)
        # for j in range(len(types_legend)):
        #     if j in [2,3,4]:
        #         continue
        #     counter+=1
        #     ax.plot([x_vec_optimal[i,j,ii,0]],[y_vec_optimal[i,j,ii,0]],'X',color=colors[counter],label=types_legend[j])  
        # #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
    ax.set_title(id_scenarios[i])
    ax.legend(loc='center right',bbox_to_anchor = [1.7,0.5],frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('%s [%s]' % (x_title, 'thousands' if 'deaths' in x_title else 'millions'))
    ax.set_xticklabels([int(el//(1e3)) if 'deaths' in x_title else round(el/1e6,1) for el in ax.get_xticks()])
    ax.set_ylabel('%s [%s]' % (y_title, 'thousands' if 'deaths' in y_title else 'millions'))
    ax.set_yticklabels([int(el//(1e3)) if 'deaths' in y_title else round(el/1e6,1) for el in ax.get_yticks()])    
    plt.savefig('%s_overall_pareto_%s_vs_%s_%iphases_%s%s' % (infix,x_title,y_title,different_numbers_of_phases[ii],id_scenarios[i],ending),bbox_inches = "tight")
    
        
        
        
        
        
        
        
        
        

#optimal allocations
which_scenarios = np.array([0,1,3,4,5,7])

for number_phases in [4,5,10]:
    try:
        index = different_numbers_of_phases.index(number_phases)
    except IndexError:
        continue
    
    index_names = ['age','ethnicity','occupation'] + list(np.array(id_scenarios)[which_scenarios])
    A = pd.DataFrame(np.r_[source.indices_out.T,optimal_allocationss[which_scenarios,0,index,0,:]+1].T,columns=index_names) 
    for i,dictionary in enumerate([source.dict_ages,{0: 'POC', 1: 'WA'},{0: 'n.a. / low',1: 'high'}]):
        A.iloc[:,i] = [dictionary[int(el)] for el in A.iloc[:,i]]
    A.to_csv('%s_optimalallocation_with_%i_phases%s' % (infix_short,number_phases,'.csv'))


SHOW_LEGEND_TITLE = False

x_vec = deathss
y_vec = casess
x_vec_optimal = deaths_optimal
y_vec_optimal = cases_optimal
x_title = 'deaths'
y_title = 'cases'

# x_vec = deathss_black
# y_vec = deathss_white
# x_vec_optimal = deaths_black_optimal
# y_vec_optimal = deaths_white_optimal
# x_title = 'deaths_black'
# y_title = 'deaths_white'

indices_scenarios = [0,3,7]
indices_phases = [4,5]
markers = ['P','o','P','P','o','o']
types_legend = types
colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(8),cm.tab10(7),cm.tab10(6),cm.tab10(5)]
colors = ['r','r','k','k','k','k']
types_legend = types
for i in indices_scenarios:#range(nr_scenarios):        
    for ii in indices_phases:
        dummy = is_pareto_efficient_2d(x_vec[i][indices_specific_phase[ii]],y_vec[i][indices_specific_phase[ii]],True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        f,ax = plt.subplots(figsize=(3,3))
        ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
        counter=0
        #ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],label='CDC',zorder=3)
        for j in range(len(types_legend)):
            if j in [2,3,4]:
                continue
            ax.plot([x_vec_optimal[i,j,ii,0]],[y_vec_optimal[i,j,ii,0]],'X',color=colors[counter],marker=markers[counter],label=types_legend[j],ms=10,markerfacecolor="None" if counter in [3,5] else colors[counter],markeredgecolor=colors[counter], markeredgewidth=1)
            counter+=1
        #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
        if SHOW_LEGEND_TITLE:
            ax.set_title(id_scenarios[i]+'\n'+str(different_numbers_of_phases[ii])+' phases')
            ax.legend(loc='center right',bbox_to_anchor = [1.7,0.5],frameon=False)
        #ax.legend(loc='center right',bbox_to_anchor = [0.5,1.5],frameon=False,ncol=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('%s [%s]' % (x_title, 'thousands' if 'deaths' in x_title else 'millions'))
        ax.set_xticklabels([int(el//(1e3)) if 'deaths' in x_title else round(el/1e6,1) for el in ax.get_xticks()])
        ax.set_ylabel('%s [%s]' % (y_title, 'thousands' if 'deaths' in y_title else 'millions'))
        ax.set_yticklabels([int(el//(1e3)) if 'deaths' in y_title else round(el/1e6,1) for el in ax.get_yticks()])    
        plt.savefig('%s_overall_pareto_%s_vs_%s_%iphases_%s%s' % (infix,x_title,y_title,different_numbers_of_phases[ii],id_scenarios[i],ending),bbox_inches = "tight")




x_vec = deathss_black
y_vec = deathss_white
x_vec_optimal = deaths_black_optimal
y_vec_optimal = deaths_white_optimal
x_title = 'deaths POC'
y_title = 'deaths WA'

indices_scenarios = [0,3,7]
indices_phases = [4,5]
markers = ['P','o','P','P','o','o']
types_legend = types
colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(8),cm.tab10(7),cm.tab10(6),cm.tab10(5)]
colors = ['r','r','k','k','k','k']
for i in indices_scenarios:#range(nr_scenarios):        
    for ii in indices_phases:
        dummy = is_pareto_efficient_2d(x_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==0])*100,y_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==1])*100,True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        f,ax = plt.subplots(figsize=(3,3))
        ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
        counter=0
        for j in range(len(types_legend)):
            if j in [2,3,4]:
                continue
            ax.plot([x_vec_optimal[i,j,ii,0]/sum(source.Nsize[source.race_class==0])*100],[y_vec_optimal[i,j,ii,0]/sum(source.Nsize[source.race_class==1])*100],'X',color=colors[counter],marker=markers[counter],label=types_legend[j],ms=10,markerfacecolor="None" if counter in [3,5] else colors[counter],markeredgecolor=colors[counter], markeredgewidth=1)  
            counter+=1
        #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
        if SHOW_LEGEND_TITLE:
            ax.set_title(id_scenarios[i]+'\n'+str(different_numbers_of_phases[ii])+' phases')
            ax.legend(loc='center right',bbox_to_anchor = [1.7,0.5],frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('%s [%s]' % (x_title, '% of total'))
        #ax.set_xticklabels([int(el//(1e3)) if 'deaths' in x_title else round(el/1e6,1) for el in ax.get_xticks()])
        ax.set_ylabel('%s [%s]' % (y_title, '% of total'))
        #ax.set_yticklabels([int(el//(1e3)) if 'deaths' in y_title else round(el/1e6,1) for el in ax.get_yticks()])    
        plt.savefig('%s_overall_pareto_%s_vs_%s_proportion_%iphases_%s%s' % (infix,x_title,y_title,different_numbers_of_phases[ii],id_scenarios[i],ending),bbox_inches = "tight")






x_vec = casess_black
y_vec = casess_white
x_vec_optimal = cases_black_optimal
y_vec_optimal = cases_white_optimal
x_title = 'cases POC'
y_title = 'cases WA'

indices_scenarios = [0,3,7]
indices_phases = [4,5]
markers = ['P','o','P','P','o','o']
types_legend = types
colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(8),cm.tab10(7),cm.tab10(6),cm.tab10(5)]
colors = ['r','r','k','k','k','k']
for i in indices_scenarios:#range(nr_scenarios):        
    for ii in indices_phases:
        dummy = is_pareto_efficient_2d(x_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==0])*100,y_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==1])*100,True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        f,ax = plt.subplots(figsize=(3,3))
        ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
        counter=0
        for j in range(len(types_legend)):
            if j in [2,3,4]:
                continue
            ax.plot([x_vec_optimal[i,j,ii,0]/sum(source.Nsize[source.race_class==0])*100],[y_vec_optimal[i,j,ii,0]/sum(source.Nsize[source.race_class==1])*100],'X',color=colors[counter],marker=markers[counter],label=types_legend[j],ms=10,markerfacecolor="None" if counter in [3,5] else colors[counter],markeredgecolor=colors[counter], markeredgewidth=1)  
            counter+=1
        #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
        if SHOW_LEGEND_TITLE:
            ax.set_title(id_scenarios[i]+'\n'+str(different_numbers_of_phases[ii])+' phases')
            ax.legend(loc='center right',bbox_to_anchor = [1.7,0.5],frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('%s [%s]' % (x_title, '% of total'))
        #ax.set_xticklabels([int(el//(1e3)) if 'deaths' in x_title else round(el/1e6,1) for el in ax.get_xticks()])
        ax.set_ylabel('%s [%s]' % (y_title, '% of total'))
        #ax.set_yticklabels([int(el//(1e3)) if 'deaths' in y_title else round(el/1e6,1) for el in ax.get_yticks()])    
        plt.savefig('%s_overall_pareto_%s_vs_%s_proportion_%iphases_%s%s' % (infix,x_title,y_title,different_numbers_of_phases[ii],id_scenarios[i],ending),bbox_inches = "tight")













x_vec = deathss_black
y_vec = deathss_white
x_vec_optimal = deaths_black_optimal
y_vec_optimal = deaths_white_optimal
x_title = 'deaths POC'
y_title = 'deaths WA'

indices_scenarios = [0,7]
indices_phases = [4]
markers = ['P','o','P','P','o','o']
types_legend = types
colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(8),cm.tab10(7),cm.tab10(6),cm.tab10(5)]
colors = ['r','r','k','k','k','k']
for i in indices_scenarios:#range(nr_scenarios):        
    for ii in indices_phases:
        dummy = is_pareto_efficient_2d(x_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==0])*100,y_vec[i][indices_specific_phase[ii]]/sum(source.Nsize[source.race_class==1])*100,True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        
        
        indices = indices_specific_phase[ii]
        lowest_index = indices[0]
        indices_allocation_vaccinates_all_POC_before_any_WA = np.where(allocation_vaccinates_all_POC_before_any_WA[indices]==1)[0]
        indices = indices[indices_allocation_vaccinates_all_POC_before_any_WA]
        
        
        
        f,ax = plt.subplots(figsize=(3,3))
        ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
        ax.plot(x_vec[i][lowest_index+indices]/sum(source.Nsize[source.race_class==0])*100,y_vec[i][lowest_index+indices]/sum(source.Nsize[source.race_class==1])*100,'X',color='b',marker='o',label=types_legend[j],ms=7)  
        #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
        if SHOW_LEGEND_TITLE:
            ax.set_title(id_scenarios[i]+'\n'+str(different_numbers_of_phases[ii])+' phases')
            ax.legend(loc='center right',bbox_to_anchor = [1.7,0.5],frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('%s [%s]' % (x_title, '% of total'))
        #ax.set_xticklabels([int(el//(1e3)) if 'deaths' in x_title else round(el/1e6,1) for el in ax.get_xticks()])
        ax.set_ylabel('%s [%s]' % (y_title, '% of total'))
        #ax.set_yticklabels([int(el//(1e3)) if 'deaths' in y_title else round(el/1e6,1) for el in ax.get_yticks()])    
        plt.savefig('%s_overall_pareto_%s_vs_%s_proportion_%iphases_%s%s' % (infix,x_title,y_title,different_numbers_of_phases[ii],id_scenarios[i],ending),bbox_inches = "tight")












## Spearman correlations
#types= ['Deaths','Cases','Infections','YLL','Equity']
types= ['deaths','cases','deaths POC','deaths WA','cases POC','cases WA']

ending = '.pdf'
spearman = np.ones((nr_scenarios,len(types),len(types)))
index_phase=5
n_types=len(types)
for j in [7]:#range(nr_scenarios):
    #vecs = [deathss[j],casess[j],total_infectionss[j],yllss[j]]
    #vecs = [deathss[j],casess[j],yllss[j],entropiess[j]]
    vecs = [deathss[j],casess[j],deathss_black[j],deathss_white[j],casess_black[j],casess_white[j]]
    vecs = np.array(vecs)
    vecs = vecs[:,indices_specific_phase[index_phase]]
    for ii in range(len(vecs)):
        for jj in range(ii+1,len(vecs)):
            spearman[j,ii,jj] = stats.spearmanr(vecs[ii],vecs[jj])[0]
            spearman[j,jj,ii] = spearman[j,ii,jj]

    f,ax = plt.subplots(figsize=(3.7,3.7))
    #cax = ax.imshow(spearman[j],cmap=matplotlib.cm.RdBu,vmin=-1,vmax=1,extent=(-0.5, spearman.shape[2]-0.5, -0.5, spearman.shape[1]-0.5))
    cax = ax.imshow(spearman[j],cmap=matplotlib.cm.RdBu,vmin=-1,vmax=1,extent=(-0.5, spearman.shape[2]-0.5, -0.5, spearman.shape[1]-0.5))
    for ii in range(len(vecs)):
        for jj in range(len(vecs)): 
            ax.text(ii,5-jj,str(round(spearman[j,ii,jj],2)) if spearman[j,ii,jj]<1 else '1',va='center',ha='center',color='k' if spearman[j,ii,jj]<0.65 else 'white')
    #cbar = ax.figure.colorbar(im, ax=ax)
    #cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
    ax.set_xticks(np.arange(n_types))
    ax.set_yticks(np.arange(n_types))
    ax.set_xticklabels(types,rotation=90)
    ax.set_yticklabels(types[::-1])
    ax.set_ylim([-.5,(n_types)-.5])
    ax.set_xlim([-.5,(n_types)-.5])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="4%", pad=0.1)
    cbar=f.colorbar(cax,cax=caxax)
    #cbar.ax.set_ylabel('Spearman correlation (Scenario %i)' % (j+1), rotation=-90, va="bottom")
    cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
    plt.savefig(('%s_spearman_correlations_scenario%i_%iphases' % (infix,j+1,different_numbers_of_phases[index_phase]))+ending,bbox_inches = "tight")











