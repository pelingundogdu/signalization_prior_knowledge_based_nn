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

echo "Exporting signaling pathway information from hipathia --- scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r executing..."
# Rscript scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r -sp hsa -src hipathia
# Rscript scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r -sp mmu -src hipathia

echo "Processing the pathway list, removing disease related pathways --- scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py executing..."
# python scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py -sp hsa -src hipathia
# python scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py -sp mmu -src hipathia

echo "Exporting gene list based on processed pathway list in 1.2-pg-remove-disease-cancer.py --> scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r executing..."
# Rscript scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r -sp hsa -src hipathia
# Rscript scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r -sp mmu -src hipathia

echo "Converting entrex id value into gene symbol -->  scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r executed!!"
# Rscript scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r -sp hsa -src hipathia -ga org.Hs.eg.db
# Rscript scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r -sp mmu -src hipathia -ga org.Mm.eg.db

echo "Creating prior biological knowledge information to include the nn design in first hidden layer --> scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py executing..."
# python scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py -sp hsa -src hipathia
# python scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py -sp mmu -src hipathia

echo "PATHWAY INFORMATION EXPORTED!!!"

echo "Exporting data/processed/pbk_layer_{BIO_SOURCE} files"
# python scripts/bio_layer_data/1.0-pg-exporting-bio-layer.py


echo "PREPROCESSING EXPERIMENTS' DATASETS"

# $ python notebooks/2.0-pg-preprocessing-dataset.py -exp {EXPERIMENT NAME}
#                                                    -ds  {DATASET NAME} 
#                                                    -sc  {SCALER, StandardScaler(ss), MinMaxScaker(mms), Log1p} 
#                                                    -tci {TARGET COLUMN INDEX}
#                                                    -ofn {OUTPUT FILE NAME}

echo "PREPROCESSING of EXPERIMENT MELANOMA DATASET"

python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_melanoma \
    -ds reference.pck \
    -sw False \
    -sc log1p \
    -tci -1 \
    -ofn reference &&
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_melanoma \
    -ds query.pck \
    -sw False \
    -sc log1p \
    -tci -1 \
    -ofn query
    
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_melanoma \
    -ds query_reference_wo_negcell.pck \
    -sw False \
    -sc log1p \
    -tci -1 \
    -ofn query_reference_wo_negcell

echo "PREPROCESSING of EXPERIMENT MOUSE DATASET"

python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_mouse \
    -ds 1-3_integrated_NNtraining.pck \
    -sw False \
    -sc ss \
    -tci 0 \
    -ofn mouse_learning &&
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_mouse \
    -ds 3-33_integrated_retrieval_set.pck \
    -sw False \
    -sc ss \
    -tci 0 \
    -ofn mouse_retrieval
    
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_mouse \
    -ds 3-33_integrated_retrieval_set_cv.pck \
    -sw False \
    -sc ss \
    -tci -1 \
    -ofn mouse_retrieval_cv

echo "PREPROCESSING of EXPERIMENT PBMC DATASET"
######## MAGIC ########
## magic with sample-wise and log normalization
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_pbmc -loc processed -ds Immune_magic_sw.pck -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1

######### RAW #########

python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_pbmc \
    -ds Immune.pck \
    -sw True \
    -sc log1p \
    -tci -1 \
    -ofn pbmc_fig2

echo "PREPROCESSING of EXPERIMENT IMMUNE DATASET"

python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_immune \
    -ds Fig3g.pck \
    -sw False \
    -sc None \
    -tci -1 \
    -ofn immune_fig3


echo "NEURAL NETWORK TRAINING"

############################################## MELANOMA EXPERIMENT ##############################################
echo "MODEL TRAINING for MELANOMA EXPERIMENT"

# 1-LAYER DENSE100
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense100 \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split StratifiedKFold \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense100 \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split train_test_split \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 1_layer_dense100 \
    -first_hidden_layer_pbk None \
    -first_hidden_layer_dense 100 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split LeaveOneGroupOut \
    -filter_gene_space False

# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split StratifiedKFold \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split train_test_split \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split LeaveOneGroupOut \
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
    -split StratifiedKFold \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split train_test_split \
    -filter_gene_space False &&
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split LeaveOneGroupOut \
    -filter_gene_space False 
    
## CROSS-VALIDATION
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split RepeatedStratifiedKFold \
    -filter_gene_space False    
python notebooks/4.0-pg-model.py \
    -design 2_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_melanoma/reference_log1p.pck \
    -split RepeatedStratifiedKFold \
    -filter_gene_space False
    
    
############################################## MELANOMA EXPERIMENT ##############################################




############################################### MOUSE EXPERIMENT ###############################################

echo "NEURAL NETWORK TRAINING for PBMC EXPERIMENT"
### python notebooks/4.0-pg-model-training.py -exp exper_pbmc -loc processed -ds Immune_magic_sw_log1p.pck -pbk pbk_layer_hsa.txt -split KFold -nncv NN
# python notebooks/4.0-pg-model-training.py -exp exper_pbmc -loc processed -ds Immune_sw_log1p.pck -pbk pbk_layer_hsa.txt -split KFold -nncv NN

echo "NEURAL NETWORK TRAINING for IMMUNE EXPERIMENT"
python notebooks/4.0-pg-model.py \
    -design 1_layer_signaling \
    -first_hidden_layer_pbk pbk_layer_hsa_sig.txt \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer Adam \
    -activation relu \
    -ds processed/exper_immune/immune_fig3.pck \
    -split train_test_split \
    -filter_gene_space False


echo "CROSS VALIDATION"
echo "CROSS VALIDATION for HUMAN EXPERIMENT"
# python notebooks/5.0-pg-cross-validation.py -exp exper_melanoma -loc processed -ds reference_query_log1p.pck -pbk pbk_layer_hsa.txt -nncv CV

echo "CROSS VALIDATION for PBMC EXPERIMENT"
# python notebooks/5.0-pg-cross-validation.py -exp exper_pbmc -loc processed -ds Immune_sw_log1p.pck -pbk pbk_layer_hsa.txt -nncv CV


echo "RETRIEVAL ANALYSIS for MOUSE EXPERIMENT"


python notebooks/8.0-paper-retrieval.py models/exper_mouse/None 0 saved_model 0 all



python notebooks/8.1-paper-getting-retrieval-summary.py reports/retrieval/exper_mouse/

date