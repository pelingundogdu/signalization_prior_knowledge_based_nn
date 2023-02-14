#!/bin/bash

bio_knowledge='pbk_circuit_hsa_sig.txt'
stratified_split=10
stratified_repeat=1
activation='relu'
optimizer='Adam'
epochs=100 #100
batch_size=10 #10
dataset_name='processed/exper_pbmc/pbmc_sw_log1p.pck'

date

# echo "RepeatedStratifiedKFold"
# ############################################### PBMC EXPERIMENT ###############################################
# # 1-LAYER SIGNALING
# design_name='circuits_signaling_1_layer'
# second_hidden_layer='False'
# nohup python paper2_figures_for_Review/00_experiment_fake.py \
#         -design_name $design_name \
#         -dataset_name $dataset_name \
#         -bio_knowledge $bio_knowledge \
#         -stratified_split $stratified_split \
#         -stratified_repeat $stratified_repeat \
#         -activation $activation \
#         -optimizer $optimizer \
#         -epochs $epochs \
#         -batch_size $batch_size \
#         -second_hidden_layer $second_hidden_layer > paper2_figures_for_Review/nohup/experiment_pbmc_fake_1layer.out 2> paper2_figures_for_Review/nohup/experiment_pbmc_fake_1layer.err &
# ############################################### PBMC EXPERIMENT ###############################################


############################################### PBMC EXPERIMENT ###############################################
# 2-LAYER SIGNALING

design_name='circuits_signaling_2_layer'
second_hidden_layer='True'

nohup python paper2_figures_for_Review/00_experiment_fake.py \
        -design_name $design_name \
        -dataset_name $dataset_name \
        -bio_knowledge $bio_knowledge \
        -stratified_split $stratified_split \
        -stratified_repeat $stratified_repeat \
        -activation $activation \
        -optimizer $optimizer \
        -epochs $epochs \
        -batch_size $batch_size \
        -second_hidden_layer $second_hidden_layer > paper2_figures_for_Review/nohup/experiment_pbmc_fake_2layer.out 2> paper2_figures_for_Review/nohup/experiment_pbmc_fake_2layer.err &
############################################### PBMC EXPERIMENT ###############################################
