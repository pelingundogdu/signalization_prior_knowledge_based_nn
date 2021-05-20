# #!/bin/bash

date

echo "RETRIEVAL - LocalOutlierFactor for MELANOMA EXPERIMENT"
############################################## MELANOMA EXPERIMENT ##############################################
# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -analysis retrieval_lof \
    -filter_gene_space False
    
# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -analysis retrieval_lof \
    -filter_gene_space False 
############################################## MELANOMA EXPERIMENT ##############################################

date