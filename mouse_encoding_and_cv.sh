# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# os.chdir(ROOT_DIR)
# sys.path.append(ROOT_DIR)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

### hsa --> homo sapiens(human), org.Hs.eg.db
### mmu --> mus musculus(mouse), org.Mm.eg.db

###  Genome Wide annotation
### https://bioconductor.org/packages/3.12/data/annotation/

### StandardScaler\(\)
### FunctionTransformer\(np.log1p\)

# #!/bin/bash

data

############################################### MOUSE EXPERIMENT ###############################################

echo "MODEL TRAINING for MOUSE EXPERIMENT"

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
    -split train_test_split \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split train_test_split \
    -filter_gene_space False
###################### MOUSE EXPERIMENT - ENCODING ######################


################## MOUSE EXPERIMENT - CROSS-VALIDATION ##################
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split RepeatedStratifiedKFold \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split RepeatedStratifiedKFold \
    -filter_gene_space False
################## MOUSE EXPERIMENT - CROSS-VALIDATION ##################

