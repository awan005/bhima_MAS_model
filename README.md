Coupled multi-agent systems model for the Bhima basin, India


Overview
A coupled multi-agent systems model that integrates human-environment interactions and responses to future drought, population, and economic conditions in the Bhima basin, India. The model evaluates policy options for addressing future urban freshwater insecurity in the Pune agglomeration.


Key Features:
•	Integrates CWatM, urban, farmers profit-optimization and agricultural water use simulation code
•	Runs for both historical and future periods
•	Includes different future climate change and socio-economic development scenarios 
•	Evaluates a suite of policy interventions developed for the Pune agglomeration 


Quick Start
Installation
# Get the repository 
# ... copy files 

# Create environment 
conda env create -f py38_bhima_requirement.yml

# Activate environment
conda activate py38_bhima


Usage
# Submit a batch job using SLURM to run historical and all interventions under scenario1
sbatch run_model_scenario1.sbatch

# Or directly run the historical case with Python 
python integrated_model_run.py -p 1 n output_folder


Project Structure
.
├── README.md                         
├── basepath_file.py
├── py38_bhima_requirement.yml
├── modules                      # module inputs and outputs
├── CWatM                        # CWatM code (hydrological modeling)
├── netsim                       # module code (for urban/farmer agents)
├── runs                         # systems model integration code        
├── integrated_model.py          
├── integrated_model_run.py
├── integrated_model_stage.py
├── model_run_ak_urban.xlsx      # determine runs (scenario x intervention)
├── scenario_intervention_parameters_urban.xlsx   # parameters 
├── scenario_intervention_input.py    # read parameter inputs
├── run_model_scenario1.sbatch   # SLURM batch job scripts
├── global_var.py                # global variables
├── helper_objects               # helper functions
├── python_packages	             # external dependencies (Pynsim)
├── OUT_MAP_Daily.txt            # output daily variables 
├── OUT_MAP_MonthAvg.txt         # output monthly average variables
├── OUT_MAP_MonthEnd.txt         # output monthly end variables
├── OUT_MAP_MonthTot.txt         # output monthly total variables
└── OUT_MAP_TotalEnd.txt         # output total end variables


Key Model Inputs:
•	Hydrological model inputs  modules/hydro/hydro_inputs
•	Farmer and Agricultural module inputs  modules/ag_seasonal/ag_inputs
•	Urban module inputs  modules/urban_module/Inputs
•	Socio-economic development scenario inputs modules/urban_module/Inputs & modules/urban_module/urban_growth_files
•	Climate change scenario inputs modules/hydro/hydro_inputs/meteo/GCM


Key Model Outputs:
•	Reservoir storage level
•	Groundwater depth
•	River discharge
•	Urban water use (source-wise)
•	Agricultural water use (source-wise)


Configuration:
•	Design simulation run with different scenarios and interventions  model_run_ak_urban.xlsx
•	Parameters for each scenario and intervention  scenario_intervention_parameters_urban.xlsx


