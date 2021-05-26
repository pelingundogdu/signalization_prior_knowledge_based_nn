# #!/bin/bash

date

echo "EVALUATE RepeatedStratifiedKFold for MOUSE EXPERIMENT"
############################################### MOUSE EXPERIMENT ###############################################
# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_retrieval_sw_gw_cv.pck \
    -analysis evaluate_rskf \
    -filter_gene_space False
    
# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_retrieval_sw_gw_cv.pck \
    -analysis evaluate_rskf \
    -filter_gene_space False
############################################### MOUSE EXPERIMENT ###############################################

date