# Coupled multi-agent systems model for the Bhima basin, India
A coupled multi-agent systems model that integrates human-environment interactions and responses to future drought, population, and economic conditions in the Bhima basin, India. The model evaluates policy options for addressing future urban freshwater insecurity in the Pune agglomeration.


## Key Features
- Integrates CWatM, urban, farmers profit-optimization and agricultural water use simulation code
- Runs for both historical and future periods
- Includes different future climate change and socio-economic development scenarios
- Evaluates a suite of policy interventions developed for the Pune agglomeration 

## Installation
```
# Get the repository 
# ... copy files 

# Create environment 
conda env create -f py38_bhima_requirement.yml

# Activate environment
conda activate py38_bhima

# Import files
# Import folder “HydroTest_2021-11-01_v2” to be under the directory: modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2/
# The folder and file can be downloaded via this link: https://drive.google.com/drive/folders/12Wm_dXWd7jg37RlHl4oZtAnTseSwe5w6?usp=sharing 
```

## Usage
```
# Submit a batch job using SLURM to run historical case and all interventions under scenario1
sbatch run_model_scenario1.sbatch

# Or directly run the historical case with Python 
python integrated_model_run.py -p 1 n output_folder
```

## Project Structure
```
.
├── README.md                         
├── basepath_file.py
├── py38_bhima_requirement.yml
├── modules                      # module inputs 
├── CWatM                        # CWatM code (hydrological modeling)
├── netsim                       # module code (for urban/farmer agents)
├── runs                         # systems model integration code        
├── integrated_model.py          
├── integrated_model_run.py
├── integrated_model_stage.py
├── model_run_ak_urban.xlsx      # determine runs (scenario x intervention)
├── scenario_intervention_parameters_urban.xlsx   # parameters for scenarios and interventions
├── scenario_intervention_input.py    # read parameters
├── run_model_scenario1.sbatch   # SLURM batch job scripts
├── global_var.py                # global variables
├── helper_objects               # helper functions
├── python_packages	             # external dependencies (Pynsim)
├── OUT_MAP_Daily.txt            # output daily variables 
├── OUT_MAP_MonthAvg.txt         # output monthly average variables
├── OUT_MAP_MonthEnd.txt         # output monthly end variables
├── OUT_MAP_MonthTot.txt         # output monthly total variables
├── OUT_MAP_TotalEnd.txt         # output total end variables
├── outputs                      # raw data & processed data output
└── figures                      # notebooks & excel files to reproduce figures
```

## Key Model Inputs
- Hydrological model inputs  ```modules/hydro/hydro_inputs```
- Farmer and Agricultural module inputs  ```modules/ag_seasonal/ag_inputs```
- Urban module inputs  ```modules/urban_module/Inputs```
- Socio-economic development scenario inputs ```modules/urban_module/Inputs & modules/urban_module/urban_growth_files```
- Climate change scenario inputs ```modules/hydro/hydro_inputs/meteo/GCM```


## Configuration 
- Design simulation run with different scenarios and interventions  ```model_run_ak_urban.xlsx```
- Parameters for each scenario and intervention  ```scenario_intervention_parameters_urban.xlsx```


## Generate Outputs
- Raw data output
- Data processing scripts
- Processed data output


## Reproduce All the Figures
The following files can then be used to reproduce the figures in the main text and in the supplementary material:
|Figure  | Notebooks & Excel files  | 
|---------  | --------- |  
|Figure 2  | figure2_ssp_rcp_scenarios.xlsx   |         
|Figure 3 | figure3_resourcestatusmetrics.ipynb                |
|Figure 4 | figure4_wellbeingmetrics_scenarios.ipynb              |
|Figure 5 | figure5_wellbeingmetrics_interventions.ipynb                |
|Figure 6 | figure6_strategic_intervention_implementation.ipynb            |
|Figure S1 | SI_figureS1_climate.ipynb               |
|Figure S2 | SI_figureS2_temp.xlsx             |
|Figure S3-S12 | SI_figureS3_S12.ipynb              |



