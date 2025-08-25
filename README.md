
### data processing

to access waveform data need to create a physionet account & get access
**export WFDB_USERNAME='your_username'**
**export WFDB_PASSWORD='your_password'**

generate_input_data.sh

- create_ic_p_mimic.py => downloads numerics data

- create_numerics_parquets.py

=> from numeric parquets + meds parquet => generate tensors
in that order

- 1) create_med_tensors
- 2) create_ic_targets.py
- 3) create_context_tensors.py

### Hybrid SDE

run.sh

### Ablations

+/- encoder model

hybrid_sde_ablations.sh

### Baselines

zenker_baseline.sh
