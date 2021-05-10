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

date

############################################### MOUSE EXPERIMENT ###############################################

echo "MODEL TRAINING for MOUSE EXPERIMENT"

##################### MOUSE EXPERIMENT - RETRIEVAL #####################
# 1-LAYER DENSE100
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense100 \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False

# 1-LAYER and 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False

# 1-LAYER and 2-LAYER METABOLIC and SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_metabolic_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_metabolic_signaling \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False
    
    
# 1-LAYER SIGNALING + 100dense
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling+100dense \
    -first_hidden_layer_pbk pbk_layer_mmu_sig.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False &&

# 1-LAYER METABOLIC and SIGNALING + 100dense
python notebooks/4.0-pg-model.py \
    -design 1_layer_metabolic_signaling+100dense \
    -first_hidden_layer_pbk pbk_layer_mmu_metsig.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False &&

# PPI+100 and PPITF+100 DESIGN
python notebooks/4.0-pg-model.py \
    -design 1_layer_ppi100 \
    -first_hidden_layer_pbk pbk_layer_mmu_ppi.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 1_layer_ppitf100 \
    -first_hidden_layer_pbk pbk_layer_mmu_ppitf.txt \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer SGD \
    -activation tanh \
    -ds processed/exper_mouse/mouse_learning_ss.pck \
    -split None \
    -filter_gene_space False
##################### MOUSE EXPERIMENT - RETRIEVAL #####################
############################################### MOUSE EXPERIMENT ###############################################
