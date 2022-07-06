# Ethnicity-and-vaccine-prioritization

This repository contains all files needed to run the COVID-19 vaccine allocation model, described in the recent article xxx.

To run the model for a single choice of parameters and vaccine allocation strategy, run the following:
```
import main_model_v43 as main
vacopt = [5,5,4,2,4,2,3,3,1,1]
initial_values=main.get_initial_values(main.Nsize,main.mu_A,main.mu_C,main.mu_Q,main.q)
results = main.fitfunc_short([0,0,0,0,0,1,1,1,1,1],initial_values,main.beta,main.q,main.midc,main.exponent,main.f_A,main.f_V,main.sigma,main.delta,main.contact_matrix)
deaths_per_age_group = results[0:4]
cases_per_age_group = results[4:8]
infections_per_age_group = results[8:12]
deaths_per_ethnicity_group = results[12:14]
cases_per_ethnicity_group = results[14:16]
infections_per_ethnicity_group = results[16:18]
```

To run the model for all possibly optimal vaccine allocation strategies, which assign vaccine access to the ten sub-populations considered in this study using 1-5 or 10 distinct phases, execute generate_all_vacopts_1to5phases_and_10phases.py once before running bash_v43_unix on a HPC cluster. The bash file calls the model once for each allocation strategy and for eight different scenarios regarding ethnic homophily and occupational hazard-related parameters.
