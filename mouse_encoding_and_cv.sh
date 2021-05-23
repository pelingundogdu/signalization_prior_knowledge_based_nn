# #!/bin/bash

data

############################################### MOUSE EXPERIMENT ###############################################

echo "MOUSE EXPERIMENT"

## ENCODING and CROSS-VALIDATION plots generated ONLY FOR SIGNALING DESING
###################### MOUSE EXPERIMENT - ENCODING ######################
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -analysis encoding \
    -filter_gene_space False 
    
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -analysis encoding \
    -filter_gene_space False
###################### MOUSE EXPERIMENT - ENCODING ######################


################## MOUSE EXPERIMENT - CROSS-VALIDATION ##################
# python notebooks/4.0-pg-model.py \
#     -design 1_layer_signaling \
#     -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
#     -first_hidden_layer_dense 0 \
#     -second_hidden_layer False \
#     -optimizer Adam \
#     -activation relu \
#     -ds processed/exper_mouse/mouse_learning_ss.pck \
#     -split RepeatedStratifiedKFold \
#     -filter_gene_space False 
    
# python notebooks/4.0-pg-model.py \
#     -design 2_layer_signaling \
#     -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
#     -first_hidden_layer_dense 0 \
#     -second_hidden_layer True \
#     -optimizer Adam \
#     -activation relu \
#     -ds processed/exper_mouse/mouse_learning_ss.pck \
#     -split RepeatedStratifiedKFold \
#     -filter_gene_space False
################## MOUSE EXPERIMENT - CROSS-VALIDATION ##################

