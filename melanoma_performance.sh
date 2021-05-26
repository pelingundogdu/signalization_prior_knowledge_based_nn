#!/bin/bash

analysis_var='performance'
optimizer_var='Adam'
activation_var='relu'
ds_var='processed/exper_melanoma/reference_log1p.pck'

date

echo "PERFORMANCE ANALYSIS for MELANOMA EXPERIMENT"
############################################### MELANOMA EXPERIMENT ###############################################
# 1-LAYER DENSE
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False
    
############################################### MELANOMA EXPERIMENT ###############################################