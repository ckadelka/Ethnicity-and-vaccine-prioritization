#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:28:36 2022

@author: ckadelka
"""

import pickle
import itertools


def possibly_optimal_allocation(allocation,TEST_NUMBER_PHASES,number_different_phases):
    if TEST_NUMBER_PHASES and number_different_phases!=len(set(allocation)):
        return False
    if allocation[3]>allocation[2]:
        return False
    if allocation[5]>allocation[4]:
        return False
    return True

Ngp = 10
all_vacopts = []
for nr_phases in range(1,5+1):
    for allocation in itertools.product(range(nr_phases),repeat=Ngp):
        if possibly_optimal_allocation(allocation,1,nr_phases):
            all_vacopts.append(allocation)
        
for allocation in itertools.permutations(range(Ngp)):
    if possibly_optimal_allocation(allocation,0,Ngp):
        all_vacopts.append(allocation)
    
pickle.dump( all_vacopts, open( "all_vacopts_1to5phases_and_10phases.p", "wb" ) )
#all_vacopts = pickle.load( open( "all_vacopts_1to5phases_and_10phases.p", "rb" ) )