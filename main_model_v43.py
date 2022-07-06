#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:33:04 2022

@author: ckadelka
"""

import sys
import os 
import pickle 
import numpy as np
from numba import jit # , prange
import pandas as pd
import get_yll_and_mean_age
import get_vaccination_rates_new
import linear_equations_contact_matrix_v43 as cm 
import get_prevalence_comorbidities

import time

import datetime

def days_between(earlier_date,later_date,str_separator='-'):
    assert type(earlier_date) in [str,datetime.date] and type(later_date) in [str,datetime.date]
    if type(earlier_date)==str:
        earlier_date = datetime.date(*list(map(lambda x: int(x.lstrip('0')),earlier_date.split(str_separator))))
    if type(later_date)==str:
        later_date = datetime.date(*list(map(lambda x: int(x.lstrip('0')),later_date.split(str_separator))))
    return int((later_date-earlier_date).days)

def days_since_20201214(date):
    return days_between(datetime.date(2020,12,14),date)


filename =  os.path.basename(sys.argv[0])
if filename!='':
    version= filename.split('.')[0][-2:] 
else: #set and update manually
    version='43'
    
output_folder = 'results/version%s/' % version
if not os.path.exists(output_folder):
    os.makedirs(output_folder,exist_ok=True) 

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1]) 
except:
    filename =  os.path.basename(sys.argv[0])
    SLURM_ID = 0#random.randint(0,100)
    
if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 2964680//20 #148234

n_different_scenarios = 8 #8

dummy = SLURM_ID%n_different_scenarios
homophily_ethnicity = [0.8,0][dummy//4] #,1
multipler_highcontact_jobs = [3.,1.][(dummy%4)//2]
relative_difference_in_high_contact_jobs_wrt_ethnicity = [3.,1.][dummy%2] #,10 #if <1, there are proportionately more WorA in high-risk jobs, otherwise more non-W and non-A

#Real U.S. data
empirical_contact_matrix = np.array(pd.read_csv('age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
census_data = np.array([ 60570126, 213610414,  31483433,  22574830])
n_ages = len(census_data)
prevalence_WorA_ethnicity = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])

proportion_of_all_16_64_yo_in_high_contact_jobs = 0.25
prevalence_high_contact = np.array([[0,0],[relative_difference_in_high_contact_jobs_wrt_ethnicity*proportion_of_all_16_64_yo_in_high_contact_jobs/(relative_difference_in_high_contact_jobs_wrt_ethnicity+prevalence_WorA_ethnicity[1]-relative_difference_in_high_contact_jobs_wrt_ethnicity*prevalence_WorA_ethnicity[1]),proportion_of_all_16_64_yo_in_high_contact_jobs/(relative_difference_in_high_contact_jobs_wrt_ethnicity+prevalence_WorA_ethnicity[1]-relative_difference_in_high_contact_jobs_wrt_ethnicity*prevalence_WorA_ethnicity[1])],[0,0],[0,0]]) 

CONSIDER_COMORBIDITIES = False
if CONSIDER_COMORBIDITIES:
    prevalence_comorbidities = get_prevalence_comorbidities.prevalence_comorbidities #https://wwwnc.cdc.gov/EID/article/26/8/20-0679-T1
    prevalence_comorbidities[0] = 0  #comment out if we don't want to consider comorbid children
    (contact_matrix,Nsize,indices_out) = cm.get_contact_matrix_ISMART(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs,prevalence_comorbidities)
    comorbidity_class = indices_out[:,3] # np.array([0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
else:
    (contact_matrix,Nsize,indices_out) = cm.get_contact_matrix_ISMART_nocomorbidities(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs)
#contact_matrix = np.round(contact_matrix,6)
#contact_matrix[contact_matrix<1e-6] = 0


#q=0.7*np.ones(17,dtype=np.float64)
Ngp=len(Nsize)#number of groups
Ncomp=20 #number of compartments per group
gp_in_agegp =indices_out[:,0] # np.array([0,0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3]) #age of each group
race_class = indices_out[:,1]
T=365+17
dt=1/2 #needs to be 1/natural number so there is an exact match between modeled case and death count and observed daily death count

def function(x,x0,k):
    return 1+(max(relative_transmissibility)*100-1)/(1+np.exp(-k*(x-x0)))

def q_based_on_q18(q18,weighted_mean_q = 0.7):
   q=np.zeros(Ngp,dtype=np.float64)
   param=0.02
   # N_4 = np.asarray(get_contact_matrix.N_4,dtype=np.float64)
   N_4 = census_data.copy()
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(N_4,q_4)/np.sum(N_4)
   q_4 = q_4 + (weighted_mean_q-mean_q)
   param = param*(q18-weighted_mean_q)/(q_4[3]-weighted_mean_q)
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(N_4,q_4)/np.sum(N_4)
   q_4 = q_4 + (weighted_mean_q-mean_q)
   for i in range(Ngp):
       q[i] = q_4[gp_in_agegp[i]]
   return  q 


@jit(nopython=True) 
def parm_to_beta(param):
   beta=np.zeros(Ngp,dtype=np.float64)
   for i in range(Ngp):
       beta[i] = param[0] + mean_age_in_agegp[i] * param[1]
   return  beta 

######
#  Global Variables
####### 

## fixed parameter choices≥

mean_4agegp = get_yll_and_mean_age.mean_4agegp #from SSA
mean_age_in_agegp = mean_4agegp[gp_in_agegp]

yll_4agegp = get_yll_and_mean_age.yll_4agegp #from SSA

hesitancy=0.3 #based on most recent estimate 
option_vaccine_projection = 2 #2 = linearly decreasing daily vaccinations, 1= constant at level of May 4, 2021 until all willing individuals are vaccinated


#Nsize = #wrong order: np.array([0.1012,0.0834,0.0800,0.1959,0.0431,0.1055,0.0426,0.1044,0.0230,0.0562,0.0548,0.0215,0.0140,0.0055,0.0443,0.0123,0.0095,0.0026],dtype=np.float64)*328238803
 #from US 2019 census data
CFR_per_age_group = get_prevalence_comorbidities.CFR_per_age_group
if CONSIDER_COMORBIDITIES:
    multiplier_CFR_comorbid = get_prevalence_comorbidities.multiplier_CFR_comorbid
    CFR_for_the_noncomorbid_in_each_age_group = [CFR_per_age_group[i]/(prevalence_comorbidities[i]*multiplier_CFR_comorbid+(1-prevalence_comorbidities[i])) for i in range(n_ages)]
    CFR = np.array([CFR_for_the_noncomorbid_in_each_age_group[index_age]*(1 if index_comorb==0 else multiplier_CFR_comorbid) for index_age,index_comorb in zip(gp_in_agegp,comorbidity_class)],dtype=np.float64)
else:
    CFR = np.array([CFR_per_age_group[i] for i in gp_in_agegp])

mu_E = 1/3.7 #https://bmjopen.bmj.com/content/10/8/e039652.abstract
mu_P = 1/2.1 #from https://www.nature.com/articles/s41591-020-0962-9.pdf
mu_C = 1/2.7227510411724856 #fitted from N=22507139 available CDC cases as of April 9, 2021
mu_Q = 1/(22-1/mu_C) #22 days yield best overlay of case and death counts using John Hopkins data 
mu_A = 1/5 #from https://www.nature.com/articles/s41591-020-0962-9.pdf, (1/mu_P+1/mu_C)


q18=0.85
q = q_based_on_q18(q18)
cases_prop=get_prevalence_comorbidities.cases_prop #only distinguish age
death_prop = get_prevalence_comorbidities.death_prop #only distinguish age


A = pd.read_csv('nyt_cases_and_deaths.csv')
dates = np.array(A.date)
cumcases = np.array(A.cases)
cumdeaths = np.array(A.deaths)
daily_cases = cumcases[1:]-cumcases[:-1]
for i in range(1,len(cumcases)-1):
    if daily_cases[i] < 0:
        approximated_daily_cases = (daily_cases[i-1] + daily_cases[i+1])/2
        cumcases[i:] += (int(approximated_daily_cases) - daily_cases[i])
        daily_cases[i] = approximated_daily_cases
daily_deaths = cumdeaths[1:]-cumdeaths[:-1]
for i in range(1,len(cumdeaths)-1):
    if daily_deaths[i] < 0:
        approximated_daily_deaths = (daily_deaths[i-1] + daily_deaths[i+1])/2
        cumdeaths[i:] += (int(approximated_daily_deaths) - daily_deaths[i])
        daily_deaths[i] = approximated_daily_deaths

index_2020_12_14 = days_between(dates[0],'2020-12-14')

number_of_daily_vaccinations = np.array(list(map(lambda x: get_vaccination_rates_new.vaccinations_on_day_t(x,hesitancy,option_vaccine_projection),range(500))),dtype=np.float64)[1:]
number_of_daily_vaccinations = np.append(np.zeros(days_between(dates[0],'2020-12-14')),number_of_daily_vaccinations)



#A = pd.read_excel('variants.xlsx')
#A = pd.read_csv('variants.csv') #https://khub.net/documents/135939561/405676950/Increased+Household+Transmission+of+COVID-19+Cases+-+national+case+study.pdf/7f7764fb-ecb0-da31-77b3-b1a8ef7be9aa
A = pd.read_csv('variants_old.csv') #https://khub.net/documents/135939561/405676950/Increased+Household+Transmission+of+COVID-19+Cases+-+national+case+study.pdf/7f7764fb-ecb0-da31-77b3-b1a8ef7be9aa
beta_multiplier = []
relative_transmissibility = np.array(A['transmissibility'])/100
dates_variants = [str(el)[:10] for el in A.columns[2:]]
dates_variants_index = [list(dates).index(d)-7 for d in dates_variants]
A_data = np.array(A.iloc[:,2:])


from scipy.optimize import curve_fit
params, covs = curve_fit(function, range(len(dates_variants_index)),np.dot(relative_transmissibility,A_data))
xs = np.arange(-dates_variants_index[0]/14,100,1/14)
#
# f,ax = plt.subplots()
# ax.plot(dates_variants_index,np.dot(relative_transmissibility,A_data),'x')
# x0 = np.arange(3,5,0.1)
# ax.plot(dates_variants_index[0]+14*xs,function(xs,params[0],params[1]))

overall_transmissibility = function(xs,params[0],params[1])
overall_transmissibility = overall_transmissibility[index_2020_12_14:]
#overall_transmissibility = (overall_transmissibility-1)*10+1

## fitted parameter choices 

#beta
#midc
#exponent



data = pd.read_csv('v39_GA_output_mod.csv')

dummy = 0
param = np.array(np.array(data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
beta = parm_to_beta(param[0:2])
midc=param[2]
exponent=param[3]**2

## variable parameter choices
f_A = 0.75 #CDC best estimate
f_V=0.5 #big unknown, we'll vary it from 0 to 1
q18 = 0.85

# dummy = SLURM_ID%9 
# if dummy<3:
#     f_A = np.array([0.25,0.75,1])[dummy%3]
# elif dummy<6:
#     f_V = np.array([0,0.5,1])[dummy%3]
# elif dummy<9:
#     q18 = np.array([0.7,0.85,1])[dummy%3]
q = q_based_on_q18(q18)

first_indices_of_each_age_group = np.array([list(gp_in_agegp).index(i) for i in range(n_ages)])

ve=0.9
sigma = 0.7* np.ones(Ngp,dtype=np.float64)#np.array([0,1-np.sqrt(1-ve),ve])[index3] * np.ones(17,dtype=np.float64)
delta = 1-(1-ve)/(1-sigma)#np.array([ve,1-np.sqrt(1-ve),0])[index3] * np.ones(17,dtype=np.float64)

## calculated values based on choice of mu_C


# active_filtered_cases = np.zeros(len(filtered_daily_cases)+1,dtype=np.float64)
# active_filtered_cases[0]=daily_cases[0]
# for i in np.arange(len(filtered_daily_cases)):
#     dummy=np.arange(min(max_days_back,i+1))
#     active_filtered_cases[i+1] = np.dot(filtered_daily_cases[i-dummy],(1-mu_C)**dummy)

# inferred_recovered = cumcases-cumdeaths-active_filtered_cases-mu_C/mu_Q*active_filtered_cases



##### All functions 
@jit(nopython=True)
def get_initial_values_2020_01_21(Nsize,mu_A,mu_C,mu_Q,q,number_initial_cases=1):
    cases_per_subpopulation = number_initial_cases * Nsize/sum(Nsize)
    initial_values=np.zeros(Ncomp*Ngp,dtype=np.float64)
    for i in range(Ngp):
        j=i*Ncomp
    
        initial_values[j+15] = cases_per_subpopulation[i]  #clinical
        initial_values[j+16] = 0. #clinical -- vaccinated
        initial_values[j+19] = 0. #clinical but no longer spreading, Q
        initial_values[j+17] = 0. #dead
        initial_values[j+18] = 0. #recov
        
        
        initial_values[j+12]= mu_C/mu_P*(1-hesitancy)*initial_values[j+15] #pre-clinical, willing to vaccinate
        initial_values[j+13]= mu_C/mu_P*(hesitancy)*initial_values[j+15] #pre-clinical, not willing to vaccinate
        initial_values[j+14]=0. #pre-clinical, vaccinated
        
        initial_values[j+3] = 1/q[i]*mu_C/mu_E*(1-hesitancy) *initial_values[j+15] #expo, willing to vaccinate
        initial_values[j+4] = 1/q[i]*mu_C/mu_E*hesitancy *initial_values[j+15] #expo, not willing to vaccinate
        initial_values[j+5] =0. #expo, vaccinated
        
        initial_values[j+6] = 1/q[i]*(1-q[i])*mu_C/mu_A*(1-hesitancy) *initial_values[j+15]  #asympto, willing to vaccinate
        initial_values[j+7] = 1/q[i]*(1-q[i])*mu_C/mu_A*hesitancy *initial_values[j+15]  #asympto, not willing to vaccinate
        initial_values[j+8] = 0. #asympto, vaccinated
        
        initial_values[j+9] =0. #asymptomatic recovered, willing to vaccinate
        initial_values[j+10]=0. #asymptomatic recovered, not willing to vaccinate
        initial_values[j+11]=0. #asymptomatic recovered, vaccinated
        
        total_susceptible= Nsize[i] - np.sum(initial_values[(j+3):(j+20)])
        
        initial_values[j]= (1-hesitancy)*total_susceptible #susceptible, willing to vaccinate
        initial_values[j+1] = hesitancy*total_susceptible #susceptible, not willing to vaccinate
        initial_values[j+2] = 0. #susceptible, vaccinated

    return initial_values

#@jit(nopython=True)
def get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,hesitancy=0.3,index_starting_date=index_2020_12_14):
    ts = np.arange(0, index_2020_12_14, dt) 
    X = get_initial_values_2020_01_21(Nsize,mu_A,mu_C,mu_Q,q)
    #YY = RK4(SYS_ODE_VAX_RK4,X,ts,vacopt,beta,exponent,midc,fatigue_slope,amplitude,fold_change_contacts_N_vs_W,mu_A,mu_C,mu_Q,q,r_infections_old,r_symptoms_old,r_infections_new,r_symptoms_new,r_deaths_old,r_deaths_new,ratio_new_transmissibility,waning_rate_old,waning_rate_new,f_A,f_V_old,f_V_new,number_of_daily_boosters_against_new_variant,True)
    YY = RK4(SYS_ODE_VAX_RK4,X,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY=False,ASSIGN_VACCINE_IN_PHASES=False,vacopt=np.zeros(Ngp,dtype=int),USE_HISTORIC_CASE_NUMBERS = True)   
    #initial_values = YY[-1,:-1]
    initial_values = YY[-1]
    
    # #Run this to create a plot of the fit
    # Y,Y_cases = YY[:,:-1],YY[:,-1]
    # sol_by_compartment = np.zeros((ts.shape[0],Ncomp),dtype=np.float64)
    # for i in range(Ncomp):
    #     sol_by_compartment[:,i]=np.sum(Y[:,i::Ncomp],axis=1)
    # f,ax = plt.subplots(2, 1,figsize=(6,8))
    # ax[0].plot(sol_by_compartment[:,17][::2],label='D')
    # ax[0].plot(cumdeaths[:index_2020_12_14] ,label='real D')
    
    # #ax[1].plot(Y_cases[::2],label='C')
    # ax[1].plot(sol_by_compartment[:,15][::2] + sol_by_compartment[:,16][::2] + sol_by_compartment[:,17][::2] + sol_by_compartment[:,18][::2] + sol_by_compartment[:,19][::2],label='C')
    # ax[1].plot(cumcases[:index_2020_12_14],label='real C')
    
    # ax[0].legend(loc='best',frameon=False) 
    # ax[1].legend(loc='best',frameon=False)
    
    # f,ax = plt.subplots()
    # ax.plot()
    
    # plt.savefig('fit_to_historic_cases_first_%i_days.pdf' % index_2021_09_01)
    return initial_values

@jit(nopython=True) 
def parm_to_beta(param):
   beta=np.zeros(Ngp,dtype=np.float64)
   for i in range(Ngp):
       beta[i] = param[0] + mean_age_in_agegp[i] * param[1]
   return  beta 

@jit(nopython=True) 
def parm_to_q(param):
   q=np.zeros(Ngp,dtype=np.float64)
   #N_4 = np.asarray(get_contact_matrix.N_4,dtype=np.float64)
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(census_data,q_4)/np.sum(census_data)
   q_4 = q_4 + (0.7-mean_q)
   for i in range(Ngp):
       q[i] = q_4[gp_in_agegp[i]]
   return  q 

@jit(nopython=True)
def mat_vecmul(matrix1,vector):
    rvector = np.zeros(matrix1.shape[0],dtype=np.float64)
    for i in range(matrix1.shape[0]):
        for k in range(matrix1.shape[0]):
            rvector[i] += matrix1[i][k] * vector[k]
    return rvector


@jit(nopython=True)#parallel=True)
def SYS_ODE_VAX_RK4(X,t,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS):     
    f = np.zeros(Ngp,dtype=np.float64)
    for i in range(Ngp):
        f[i] = (f_A*(X[6::Ncomp][i]+X[7::Ncomp][i]+f_V*X[8::Ncomp][i]) + X[12::Ncomp][i]+X[13::Ncomp][i]+f_V*X[14::Ncomp][i] + X[15::Ncomp][i]+f_V*X[16::Ncomp][i])/Nsize[i]
    
    dummy = mat_vecmul(contact_matrix,f)
    F = np.multiply(beta,dummy)

    Xprime = np.zeros(Ngp*Ncomp,dtype=np.float64)
    
    total_C = np.sum(X[15::Ncomp]) + np.sum(X[16::Ncomp]) + np.sum(X[19::Ncomp])
    r = 1/(1+(midc/np.log10(total_C+1))**exponent)
    rv = r
    
    variant_transmissibility = overall_transmissibility[int(t)-0]
    
    nu = np.zeros(Ngp,dtype=np.float64)
    daily_doses = number_of_daily_vaccinations[max(0,int(t))]
    if ASSIGN_VACCINE_CONTINUOUSLY:
        nr_of_ppl_to_be_vaccinated = sum(X[::Ncomp]+X[3::Ncomp]+X[6::Ncomp]+X[9::Ncomp]+X[12::Ncomp])
        if nr_of_ppl_to_be_vaccinated>0:
            nu = min(daily_doses/nr_of_ppl_to_be_vaccinated,1.)*np.ones(Ngp,dtype=np.float64) 
    if ASSIGN_VACCINE_IN_PHASES:
        #check what the current phase is
        nr_of_ppl_vaccinated_and_not_yet_symptomatic = X[2::Ncomp]+X[5::Ncomp]+X[8::Ncomp]+X[11::Ncomp]+X[14::Ncomp]    
        for phase in range(max(vacopt),-1,-1):
            if np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_vaccinated_and_not_yet_symptomatic)>0:
                current_phase = phase
                break
        else:
            current_phase = 0

    
        nr_of_ppl_to_be_vaccinated = X[::Ncomp]+X[3::Ncomp]+X[6::Ncomp]+X[9::Ncomp]+X[12::Ncomp]
        nr_of_ppl_to_be_vaccinated_per_phase = [np.dot(np.asarray(vacopt==0,dtype=np.float64),nr_of_ppl_to_be_vaccinated)]  
        phase=0
        if nr_of_ppl_to_be_vaccinated_per_phase[0]>0.:
            nu = min(daily_doses/nr_of_ppl_to_be_vaccinated_per_phase[0],1.)*np.asarray(vacopt==phase,dtype=np.float64)
        else:
            nu = np.zeros(Ngp,dtype=np.float64)
        for count in range(max(vacopt)):
            if sum(nr_of_ppl_to_be_vaccinated_per_phase)<daily_doses or current_phase > count:
                phase+=1
                nr_of_ppl_to_be_vaccinated_per_phase.append(np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated))
                if nr_of_ppl_to_be_vaccinated_per_phase[-1]>0.:
                    nu[vacopt==phase] = min((daily_doses-sum(nr_of_ppl_to_be_vaccinated_per_phase[:-1]))/nr_of_ppl_to_be_vaccinated_per_phase[-1],1.)
            
    new_infections_willing = np.zeros(Ngp,dtype=np.float64)
    new_infections_not_willing = np.zeros(Ngp,dtype=np.float64)
    new_infections_vaccinated = np.zeros(Ngp,dtype=np.float64)
    
    new_deaths = np.zeros(Ngp,dtype=np.float64)
    
    for i in range(Ngp):
        j=i*Ncomp

        new_infections_willing[i]     = (1-r) * X[j+0] * F[i] * variant_transmissibility
        new_infections_not_willing[i] = (1-r) * X[j+1] * F[i] * variant_transmissibility
        new_infections_vaccinated[i]  = (1-rv) * (1-sigma[i]) * X[j+2] * F[i] * variant_transmissibility

        
        Xprime[j]   = - nu[i]*X[j]#susceptible, willing to vaccinate
        #Xprime[j+1] =-(1-r)*X[j+1]*F[i]*variant_transmissibility  #susceptible, not willing to vaccinate
        Xprime[j+2] = + nu[i]*X[j]#susceptible, vaccinated
        
        Xprime[j+3] = - mu_E*X[j+3] - nu[i]*X[j+3]#expo, willing to vaccinate
        Xprime[j+4] = - mu_E*X[j+4] #expo, not willing to vaccinate
        Xprime[j+5] = - mu_E*X[j+5] + nu[i]*X[j+3] #expo, vaccinated
        
        Xprime[j+6] =mu_E * (1-q[i]) * X[j+3] - mu_A * X[j+6] - nu[i]*X[j+6]  #asympto, willing to vaccinate
        Xprime[j+7] =mu_E * (1-q[i]) * X[j+4] - mu_A * X[j+7] #asympto, not willing to vaccinate
        Xprime[j+8] =mu_E * (1-q[i]*(1-delta[i])) * X[j+5] - mu_A * X[j+8] + nu[i]*X[j+6]#asympto, vaccinated
        
        Xprime[j+9] =mu_A * X[j+6] - nu[i]*X[j+9]#asymptomatic recovered, willing to vaccinate
        Xprime[j+10]=mu_A * X[j+7] #asymptomatic recovered, not willing to vaccinate
        Xprime[j+11]=mu_A * X[j+8] + nu[i]*X[j+9] #asymptomatic recovered, vaccinated
        
        Xprime[j+12]=mu_E*q[i]*X[j+3] - mu_P*X[j+12] - nu[i]*X[j+12]#pre-clinical, willing to vaccinate
        Xprime[j+13]=mu_E*q[i]*X[j+4] - mu_P*X[j+13] #pre-clinical, not willing to vaccinate
        Xprime[j+14]=mu_E*q[i]*(1-delta[i])*X[j+5] - mu_P*X[j+14] + nu[i]*X[j+12]#pre-clinical, vaccinated
        
        Xprime[j+15]=mu_P*(X[j+12]+X[j+13]) - mu_C*X[j+15] #clinical, not vaccinated
        Xprime[j+16]=mu_P*X[j+14] - mu_C*X[j+16] #clinical, vaccinated
        
        Xprime[j+19]=mu_C*(X[j+15]+X[j+16]) - mu_Q*X[j+19] #clinical but no longer spreading, Q
        
        new_deaths[i] = CFR[i]*mu_Q*(X[j+19])
        #Xprime[j+17]=CFR[i]*mu_Q*(X[j+19]) #dead
        Xprime[j+18]=(1-CFR[i])*mu_Q*(X[j+19]) #recov
    
    if USE_HISTORIC_CASE_NUMBERS:
        historic_cases_per_agegroup = daily_cases[max(0,int(t)+6)]*cases_prop
        
        #ensure new infections with the old variant match the historic case numbers per age group from six days later (6 = average days from infection to symptoms)
        for ii in range(4):
            sum_all_infections_according_model = sum(new_infections_willing[gp_in_agegp==ii]) + sum(new_infections_not_willing[gp_in_agegp==ii]) + sum(new_infections_vaccinated[gp_in_agegp==ii])
            if sum_all_infections_according_model>0:
                new_infections_willing[gp_in_agegp==ii] = new_infections_willing[gp_in_agegp==ii] * historic_cases_per_agegroup[ii] / sum_all_infections_according_model
                new_infections_not_willing[gp_in_agegp==ii] = new_infections_not_willing[gp_in_agegp==ii] * historic_cases_per_agegroup[ii] / sum_all_infections_according_model
                new_infections_vaccinated[gp_in_agegp==ii] = new_infections_vaccinated[gp_in_agegp==ii] * historic_cases_per_agegroup[ii] / sum_all_infections_according_model
                
        #not all infections lead to cases: take into account the rate of asymptomatic infections and immunity-induced reduced rate of symptoms, per sub-population and immunity level 
        for i in range(Ngp):
            new_infections_willing[i] = new_infections_willing[i] / q[i]
            new_infections_not_willing[i] = new_infections_not_willing[i] / q[i]
            new_infections_vaccinated[i] = new_infections_vaccinated[i] / (q[i] * (1 - delta[j]))

        historic_deaths_per_agegroup = daily_deaths[max(0,int(t))]*death_prop
        #ensure new infections with the old variant match the historic case numbers per age group from six days later (6 = average days from infection to symptoms)
        for ii in range(4):
            sum_all_deaths_according_model = sum(new_deaths[gp_in_agegp==ii])
            if sum_all_deaths_according_model>0:
                new_deaths[gp_in_agegp==ii] = new_deaths[gp_in_agegp==ii] * historic_deaths_per_agegroup[ii] / sum_all_deaths_according_model
    
    #update susceptible and exposed at the end, after a possible adjustment to match the historic daily case numbers, 
    #distributed across the subpopulations at the same proportions as the model
    for i in range(Ngp):
        j=i*Ncomp
        Xprime[j+0] = Xprime[j+0] - new_infections_willing[i]
        Xprime[j+1] = Xprime[j+1] - new_infections_not_willing[i]
        Xprime[j+2] = Xprime[j+2] - new_infections_vaccinated[i]
        Xprime[j+3] = Xprime[j+3] + new_infections_willing[i]
        Xprime[j+4] = Xprime[j+4] + new_infections_not_willing[i]
        Xprime[j+5] = Xprime[j+5] + new_infections_vaccinated[i]  

        Xprime[j+17] = new_deaths[i]
        Xprime[j+18] = Xprime[j+18] + (CFR[i]*mu_Q*(X[j+19]) - new_deaths[i])

    return Xprime

@jit(nopython=True)#(nopython=True)
def RK4(func, X0, ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS=False): 
    """
    Runge Kutta 4 solver.
    """
    dt = ts[1] - ts[0]
    nt = len(ts)
    X  = np.zeros((nt, X0.shape[0]),dtype=np.float64)
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], ts[i],beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS)
        k2 = func(X[i] + dt/2. * k1, ts[i] + dt/2.,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS)
        k3 = func(X[i] + dt/2. * k2, ts[i] + dt/2.,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS)
        k4 = func(X[i] + dt    * k3, ts[i] + dt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,USE_HISTORIC_CASE_NUMBERS)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

@jit(nopython=True)
def get_total_deaths(sol):
    return np.sum(sol[-1,17::Ncomp])

# @jit(nopython=True)
# def model_evaluation_rk4_18(initial_values,t0,dt,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt,contact_matrix):  #proportion_vaccinated is of length 18
#     #initial_values=initial_values.copy()
#     ts = np.arange(t0, 382, dt) 
#     sol = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt)   
#     total_deaths = get_total_deaths(sol)
#     return total_deaths#,sol

# def model_evaluation_for_one_step(initial_values,t0,dt,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt):  #proportion_vaccinated is of length 18
#     #proportion_vaccinated = np.append(proportion_vaccinated,np.zeros(18,dtype=np.float64))
#     ts = np.array([t0,t0+dt])
#     sol = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,ASSIGN_VACCINE_CONTINUOUSLY,ASSIGN_VACCINE_IN_PHASES,vacopt)   
#     return sol[-1,:] #IMPORTANT -- I changed it to -1


###  Ploting and related functions 
dict_all_races = {'wa':'White or Asian','bama':'Not White and not Asian'}
dict_ages = {0:'0-15',1:'16-64',2:'65-74',3:'75+'}
dict_comorb = {0:'no risk factors',1:'known risk factors'} 

@jit(nopython=True)#(nopython=True)#parallel=True)
def fitfunc_short(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta,contact_matrix):
    vacopt = np.asarray(vacopt,dtype=np.float64)
    Ncomp = 20
    #ts = np.arange(-0, 365+17, dt)
    ts = np.arange(index_2020_12_14, index_2020_12_14+365+17, dt)
    Y = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,0,1,vacopt)
    dead_per_group = Y[-1,17::Ncomp]
    cases_per_group = dead_per_group + Y[-1,15::Ncomp] + Y[-1,16::Ncomp] + Y[-1,18::Ncomp] + Y[-1,19::Ncomp]
    infections_per_group = Nsize - Y[-1,0::Ncomp] - Y[-1,1::Ncomp] - Y[-1,2::Ncomp]

    output = []
    output.extend([np.sum(dead_per_group[gp_in_agegp==i]) for i in range(4)])
    output.extend([np.sum(cases_per_group[gp_in_agegp==i]) for i in range(4)])
    output.extend([np.sum(infections_per_group[gp_in_agegp==i]) for i in range(4)])

    output.extend([np.sum(dead_per_group[race_class==i]) for i in range(2)])
    output.extend([np.sum(cases_per_group[race_class==i]) for i in range(2)])
    output.extend([np.sum(infections_per_group[race_class==i]) for i in range(2)])
    return output

@jit(nopython=True)#(nopython=True)#parallel=True)
def fitfunc(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta,contact_matrix):
    vacopt = np.asarray(vacopt,dtype=np.float64)
    #ts = np.arange(-0, 365+17, dt)
    ts = np.arange(index_2020_12_14, index_2020_12_14+365+17, dt)
    Y = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,0,1,vacopt)
    return Y    

def plot_incidence_cases_black_vs_white(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta):
    import matplotlib.pyplot as plt
    vacopt = np.asarray(vacopt,dtype=np.float64)
    #ts = np.arange(-0, 365+17, dt)
    ts = np.arange(index_2020_12_14, index_2020_12_14+365+17, dt)
    
    Y = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,0,1,vacopt,False)
    #Y = RK4(SYS_ODE_VAX_RK4,X,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,ASSIGN_VACCINE_CONTINUOUSLY=False,ASSIGN_VACCINE_IN_PHASES=False,vacopt=np.zeros(Ngp,dtype=int),USE_HISTORIC_CASE_NUMBERS = True)   
    white_deaths = np.sum(Y[:,17::20][:,race_class==1],1)
    black_deaths = np.sum(Y[:,17::20][:,race_class==0],1)
    total_deaths = white_deaths + black_deaths
    new_white_deaths = white_deaths[1:]-white_deaths[:-1]
    new_total_deaths = total_deaths[1:]-total_deaths[:-1]
    
    white_infections = sum(Nsize[race_class==1]) - (np.sum(Y[:,0::20][:,race_class==1],1) + np.sum(Y[:,1::20][:,race_class==1],1) + np.sum(Y[:,2::20][:,race_class==1],1))
    black_infections = sum(Nsize[race_class==0]) - (np.sum(Y[:,0::20][:,race_class==0],1) + np.sum(Y[:,1::20][:,race_class==0],1) + np.sum(Y[:,2::20][:,race_class==0],1))
    total_infections = white_infections + black_infections
    new_white_infections = white_infections[1:]-white_infections[:-1]
    new_total_infections = total_infections[1:]-total_infections[:-1]
    
    f,ax = plt.subplots()
    ax.plot(ts[:-1]-index_2020_12_14,new_white_deaths/new_total_deaths,'b-',label='white deaths',zorder=1)
    #ax.fill_between(ts[:-1],np.zeros(len(ts[:-1])),new_white_deaths/new_total_deaths,color='#DDDDDD')
    #ax.fill_between(ts[:-1],new_white_deaths/new_total_deaths,np.ones(len(ts[:-1])),color='#AAAAAA')
    ax.plot(ts[:-1]-index_2020_12_14,new_white_infections/new_total_infections,'r--',label='white infections',zorder=1)
#    ax2 = ax.twinx()
#    ax2.bar(ts[:-1],new_total_infections,alpha=0.3)
    ax.set_xlabel('Days since 12/14/2020')
    ax.set_ylabel('proportion')
    ax.set_ylim([0,1])
    ax.set_title('vacopt = '+', '.join(list(map(str,map(int,vacopt)))))
    
    heatmap_data = []
    for i in range(Ngp):
        heatmap_data.append((1-hesitancy)*(1-Y[:,0+Ncomp*i]/Y[0,0+Ncomp*i]))
    heatmap_data = np.array(heatmap_data)
    

    # ax.set_ylim([0,1.5])
    # for i in range(Ngp):
    #     points = np.array([ts, (1+i*0.05)*np.ones(len(ts))]).T.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     norm = plt.Normalize(0,1)
    #     lc = LineCollection(segments, cmap='Greens', norm=norm)
    #     lc.set_array(heatmap_data[i])
    #     lc.set_linewidth(4)
    #     line = ax.add_collection(lc)
    # bar = f.colorbar(line, ax=ax)
    # bar.set_label('proportion vaccinated')
    # ax.legend(loc='best',frameon=False)
    
    # ax.set_ylim([0,1.5])
    # for i in range(2):
    #     points = np.array([ts, (1+i*0.1)*np.ones(len(ts))]).T.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     norm = plt.Normalize(0,1)
    #     lc = LineCollection(segments, cmap='binary', norm=norm)
    #     lc.set_array(np.mean(heatmap_data[race_class==i,:],0))
    #     lc.set_linewidth(4)
    #     line = ax.add_collection(lc)
    # bar = f.colorbar(line, ax=ax)
    # bar.set_label('proportion vaccinated')
    # ax.legend(loc='best',frameon=False)
    
    ax2 = ax.twinx()
    for i in range(2):
        ax2.plot(ts-index_2020_12_14,np.mean(heatmap_data[race_class==i,:],0))
    ax2.set_ylim([0,1])
    ax2.set_ylabel('proportion vaccinated')
    ax.legend(loc='best',frameon=False)        
    

#initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q)

# for i in range(2):
#     print(sum(initial_values[17::Ncomp][race_class==i])/sum(Nsize[race_class==i])*1e5)
    
# for i in range(2):
#     print(np.round(np.divide(initial_values[17::Ncomp][race_class==i],Nsize[race_class==i])*1e5,0))
    
# vacopt=np.zeros(10)
# ts = np.arange(index_2020_12_14, index_2020_12_14+365+17, dt)
# Y = RK4(SYS_ODE_VAX_RK4,initial_values,ts,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations,contact_matrix,0,1,vacopt,False)

# for i in range(2):
#     print(sum(Y[-1,17::Ncomp][race_class==i])/sum(Nsize[race_class==i])*1e5)
    
# for i in range(2):
#     print(np.round(np.divide(Y[-1,17::Ncomp][race_class==i],Nsize[race_class==i])*1e5,0))
    
# import matplotlib.pyplot as plt
# f,ax = plt.subplots()
# for i in range(2):
    




# n_grid = 4
# values_homophily = np.arange(n_grid)/(n_grid-1)*0.9
# values_multiplier = np.linspace(1,3,n_grid)
# values_rel_diff = np.linspace(1,3,n_grid)

# deaths_by_ethnic_group_per100k = np.zeros((n_grid,n_grid,n_grid,2))
# deaths_by_subpopulation = np.zeros((n_grid,n_grid,n_grid,Ngp))

# for ii,homophily_ethnicity in enumerate(values_homophily):
#     for jj,multipler_highcontact_jobs in enumerate(values_multiplier):
#         for kk,relative_difference_in_high_contact_jobs_wrt_ethnicity in enumerate(values_rel_diff):
#             #Real U.S. data
#             prevalence_high_contact = np.array([[0,0],[relative_difference_in_high_contact_jobs_wrt_ethnicity*proportion_of_all_16_64_yo_in_high_contact_jobs/(relative_difference_in_high_contact_jobs_wrt_ethnicity+prevalence_WorA_ethnicity[1]-relative_difference_in_high_contact_jobs_wrt_ethnicity*prevalence_WorA_ethnicity[1]),proportion_of_all_16_64_yo_in_high_contact_jobs/(relative_difference_in_high_contact_jobs_wrt_ethnicity+prevalence_WorA_ethnicity[1]-relative_difference_in_high_contact_jobs_wrt_ethnicity*prevalence_WorA_ethnicity[1])],[0,0],[0,0]]) 
#             (contact_matrix,Nsize,indices_out) = cm.get_contact_matrix_ISMART_nocomorbidities(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs)

#             initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q)
            
#             for i in range(2):
#                 deaths_by_ethnic_group_per100k[ii,jj,kk,i] = sum(initial_values[17::Ncomp][race_class==i])/sum(Nsize[race_class==i])*1e5
                
#             deaths_by_subpopulation[ii,jj,kk] =  np.divide(initial_values[17::Ncomp],Nsize)*1e5

# import matplotlib.pyplot as plt
# f,ax = plt.subplots()
# im = ax.imshow(deaths_by_ethnic_group_per100k[:,:,-1,0]/deaths_by_ethnic_group_per100k[:,:,-1,1])
# cbar = f.colorbar(im)


# f,ax = plt.subplots()
# im = ax.imshow(deaths_by_ethnic_group_per100k[:,:,0,0]/deaths_by_ethnic_group_per100k[:,:,0,1])
# cbar = f.colorbar(im)

# f,ax = plt.subplots()
# im = ax.imshow(deaths_by_ethnic_group_per100k[-1,:,:,0]/deaths_by_ethnic_group_per100k[-1,:,:,1])
# cbar = f.colorbar(im)

# for fixed_index in [0,n_grid-1]:
#     f,ax = plt.subplots()
#     im = ax.imshow(deaths_by_ethnic_group_per100k[fixed_index,:,:,0]/deaths_by_ethnic_group_per100k[fixed_index,:,:,1])
#     cbar = f.colorbar(im)
#     cbar.set_label('proportion deaths per 100k black / white')
#     ax.set_title('homophily = '+str(values_homophily[fixed_index]))
#     ax.set_xlabel('multiplier')
#     ax.set_xticks(np.arange(n_grid))
#     ax.set_xticklabels(list(map(str,map(lambda x: np.round(x,2),values_multiplier))))
#     ax.set_ylabel('relative proportion high-risk black / white')
#     ax.set_yticks(np.arange(n_grid))
#     ax.set_yticklabels(list(map(str,map(lambda x: np.round(x,2),values_rel_diff))))
#     plt.savefig('initial_values_%i_multiplier_vs_relprop.pdf',bbox_inches='tight')

#     f,ax = plt.subplots()
#     im = ax.imshow(deaths_by_ethnic_group_per100k[:,:,fixed_index,0]/deaths_by_ethnic_group_per100k[:,:,fixed_index,1])
#     cbar = f.colorbar(im)
#     cbar.set_label('proportion deaths per 100k black / white')
#     ax.set_title('relative proportion high-risk black / white = '+str(values_rel_diff[fixed_index]))
#     ax.set_xlabel('homophily')
#     ax.set_xticks(np.arange(n_grid))
#     ax.set_xticklabels(list(map(str,map(lambda x: np.round(x,2),values_homophily))))
#     ax.set_ylabel('multiplier')
#     ax.set_yticks(np.arange(n_grid))
#     ax.set_yticklabels(list(map(str,map(lambda x: np.round(x,2),values_multiplier))))
#     plt.savefig('initial_values_%i_homophily_vs_multiplier.pdf',bbox_inches='tight')

#     f,ax = plt.subplots()
#     im = ax.imshow(deaths_by_ethnic_group_per100k[:,fixed_index,:,0]/deaths_by_ethnic_group_per100k[:,fixed_index,:,1])
#     cbar = f.colorbar(im)
#     cbar.set_label('proportion deaths per 100k black / white')
#     ax.set_title('multiplier = '+str(values_multiplier[fixed_index]))
#     ax.set_xlabel('homophily')
#     ax.set_xticks(np.arange(n_grid))
#     ax.set_xticklabels(list(map(str,map(lambda x: np.round(x,2),values_homophily))))
#     ax.set_ylabel('relative proportion high-risk black / white')
#     ax.set_yticks(np.arange(n_grid))
#     ax.set_yticklabels(list(map(str,map(lambda x: np.round(x,2),values_rel_diff))))
#     plt.savefig('initial_values_%i_homophily_vs_relprop.pdf',bbox_inches='tight')








# #vacopt = [9,8,7,6,5,4,3,2,1,0]
# vacopt = race_class
# plot_incidence_cases_black_vs_white(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)
# plot_incidence_cases_black_vs_white(1-vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)
# plot_incidence_cases_black_vs_white([8,9,6,1,7,2,3,5,0,4],initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)
# plot_incidence_cases_black_vs_white([9,8,7,2,6,1,5,3,4,0],initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)
# plot_incidence_cases_black_vs_white([9,8,7,2,6,1,3,5,0,4],initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)


# vacopt = np.ones(Ngp,dtype=int)
# vacopt[-2:] = 0
# vacopt[:2] = 3
# vacopt[2:6] = 2
# initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q)
# res_from_run = fitfunc_short(np.array(vacopt),initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta)
# print(sum(res_from_run[:4]))

if __name__ == '__main__':
    all_vacopts = pickle.load( open( "all_vacopts_1to5phases_and_10phases.p", "rb" ) )

    TIME = time.time()
    dummy = SLURM_ID//n_different_scenarios
    res_from_runs = []
    vacopts_ids = []
    initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q)
    for ID in range(dummy*nsim,(dummy+1)*nsim):
        vacopt = list(all_vacopts[ID])
        #res_from_run = fitfunc_with_jump(np.array(vacopt),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,new_hesitancy=new_hesitancy)
        res_from_run = fitfunc_short(np.array(vacopt),initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta,contact_matrix)
        res_from_runs.append(res_from_run)
        vacopts_ids.append(ID)
    vacopts_ids = np.array(vacopts_ids)
    res_from_runs = np.array(res_from_runs)
    
    f = open(output_folder+'output_nsim%i_SLURM_ID%i_homophily%i_multiplier%i_reldiff%i.txt' % (nsim,SLURM_ID,int(homophily_ethnicity*100),int(multipler_highcontact_jobs*100),int(relative_difference_in_high_contact_jobs_wrt_ethnicity*100)) ,'w')
    f.write('filename\t'+filename+'\n')
    f.write('SLURM_ID\t'+str(SLURM_ID)+'\n')
    f.write('nsim\t'+str(nsim)+'\n')
    f.write('time in seconds\t'+str(int(time.time()-TIME))+'\n')
    f.write('allocation ID\t'+'\t'.join(list(map(str,vacopts_ids)))+'\n')
    counter=0
    for name in ['deaths','cases','infections']:
        for j in range(4):
            f.write(name+'_in_age_group_'+str(j+1)+'\t'+'\t'.join(list(map(lambda x: str(float(np.round(x,3))),res_from_runs[:,counter])))+'\n')
            counter+=1
    for name in ['deaths','cases','infections']:
        for j in range(2):
            f.write(name+'_among_'+np.array(['bama','wa'])[j]+'\t'+'\t'.join(list(map(lambda x: str(float(np.round(x,3))),res_from_runs[:,counter])))+'\n')
            counter+=1
    f.close()    
    
