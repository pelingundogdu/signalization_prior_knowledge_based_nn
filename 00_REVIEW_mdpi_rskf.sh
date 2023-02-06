#!/bin/bash

design_name='circuits_signaling_1_layer'
bio_knowledge='pbk_circuit_hsa_sig.txt'
stratified_split=10
activation='relu'
optimizer='Adam'
epochs=100 #100
batch_size=10 #10

date

echo "RepeatedStratifiedKFold"
############################################### PBMC EXPERIMENT ###############################################
# 1-LAYER SIGNALING
dataset_name='processed/exper_pbmc/pbmc_sw_log1p.pck'
stratified_repeat=50
nohup python paper2_figures_for_Review/00_experiment.py \
        -design_name $design_name \
        -dataset_name $dataset_name \
        -bio_knowledge $bio_knowledge \
        -stratified_split $stratified_split \
        -stratified_repeat $stratified_repeat \
        -activation $activation \
        -optimizer $optimizer \
        -epochs $epochs \
        -batch_size $batch_size > paper2_figures_for_Review/nohup/experiment_pbmc.out 2> paper2_figures_for_Review/nohup/experiment_pbmc.err
############################################### PBMC EXPERIMENT ###############################################

############################################## IMMUNE EXPERIMENT ##############################################
# 1-LAYER SIGNALING
dataset_name='processed/exper_immune/immune_new.pck'
stratified_repeat=30
nohup python paper2_figures_for_Review/00_experiment.py \
        -design_name $design_name \
        -dataset_name $dataset_name \
        -bio_knowledge $bio_knowledge \
        -stratified_split $stratified_split \
        -stratified_repeat $stratified_repeat \
        -activation $activation \
        -optimizer $optimizer \
        -epochs $epochs \
        -batch_size $batch_size > paper2_figures_for_Review/nohup/experiment_immune.out 2> paper2_figures_for_Review/nohup/experiment_immune.err &
############################################## IMMUNE EXPERIMENT ##############################################

