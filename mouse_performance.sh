#!/bin/bash

analysis_var='performance'
optimizer_var='SGD'
activation_var='tanh'
ds_var='processed/exper_mouse/mouse_learning_sw_gw.pck'

date

echo "PERFORMANCE ANALYSIS for MOUSE EXPERIMENT"
############################################### MOUSE EXPERIMENT ###############################################
# 1-LAYER DENSE100
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense100 \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# 1-LAYER and 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var\
    -analysis $analysis_var \
    -filter_gene_space False

python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# 1-LAYER and 2-LAYER METABOLIC and SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_metabolic_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False
    
python notebooks/4.0-pg-model.py \
    -design 2_layer_metabolic_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False
    
# 1-LAYER SIGNALING + 100dense
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling+100dense \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# 1-LAYER METABOLIC and SIGNALING + 100dense
python notebooks/4.0-pg-model.py \
    -design 1_layer_metabolic_signaling+100dense \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False

# PPI+100 DESIGN
python notebooks/4.0-pg-model.py \
    -design 1_layer_ppi100 \
    -first_hidden_layer_pbk pbk_layer_mmu_ppi.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer optimizer_var \
    -activation activation_var \
    -ds ds_var \
    -analysis $analysis_var \
    -filter_gene_space False
    
# PPITF+100 DESIGN
python notebooks/4.0-pg-model.py \
    -design 1_layer_ppitf100 \
    -first_hidden_layer_pbk pbk_layer_mmu_ppitf.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space False
    
############################################### MOUSE EXPERIMENT ###############################################