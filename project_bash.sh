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

# python config/settings.py

# Exporting signaling pathway information from hipathia
# Rscript src/data/1.1-pg-pathway-from-hipathia.r -sp hsa -src hipathia
# Rscript src/data/1.1-pg-pathway-from-hipathia.r -sp mmu -src hipathia
echo "Script 1.1-pg-pathway-from-hipathia.r executed!!"

# Processing the pathway list, removing disease related pathways
# python src/data/1.2-pg-remove-disease-cancer.py -sp hsa -src hipathia
# python src/data/1.2-pg-remove-disease-cancer.py -sp mmu -src hipathia
echo "Script 1.2-pg-remove-disease-cancer.py executed!!"

# Exporting gene list based on processed pathway list in 1.2-pg-remove-disease-cancer.py
# Rscript src/data/1.3-pg-gene-from-hipathia.r -sp hsa -src hipathia
# Rscript src/data/1.3-pg-gene-from-hipathia.r -sp mmu -src hipathia
echo "Script 1.3-pg-gene-from-hipathia.r executed!!"

# Converting entrex id value into gene symbol
# Rscript src/data/1.4-pg-gene-id-entrez-converter.r -sp hsa -src hipathia -ga org.Hs.eg.db
# Rscript src/data/1.4-pg-gene-id-entrez-converter.r -sp mmu -src hipathia -ga org.Mm.eg.db
echo "Script 1.4-pg-gene-id-entrez-converter.r executed!!"

# Creating prior biological knowledge information to include the nn design in first hidden layer 
# python src/data/1.5-pg-creating-biological-layer.py -sp hsa -src hipathia
# python src/data/1.5-pg-creating-biological-layer.py -sp mmu -src hipathia
echo "Script 1.5-pg-creating-biological-layer.py executed!!"

echo "COMPLETE EXTERNAL DATA SOURCE OPERTATION!!!"

echo "PREPROCESSING EXPERIMENTS' DATASETS"
echo "PREPROCESSING of EXPERIMENT HUMAN DATASET"
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_melanoma -loc external -ds reference.rds -pbk pbk_layer-hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_melanoma -loc external -ds query.rds -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1

echo "PREPROCESSING of EXPERIMENT MOUSE DATASET"
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_mouse -loc processed -ds mouse_training_sw.pck -pbk pbk_layer_mmu.txt -sc StandardScaler\(\) -tci -1
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_mouse -loc processed -ds mouse_retrieval_sw.pck -pbk pbk_layer_mmu.txt -sc StandardScaler\(\) -tci -1

echo "PREPROCESSING of EXPERIMENT PBMC DATASET"
######## MAGIC ########
## magic with sample-wise and log normalization
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_pbmc -loc processed -ds Immune_magic_sw.pck -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1

######### RAW #########
## raw with sample-wise and log normalization
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_pbmc -loc processed -ds Immune_sw.pck -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1

echo "PREPROCESSING of EXPERIMENT IMMUNE DATASET"
# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_immune -loc processed -ds exper_immune_raw_sw.pck -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1


# python notebooks/3.1-pg-preprocessing-experiment-dataset.py -exp exper_immune -loc external -ds Fig3g.pck -pbk pbk_layer_hsa.txt -sc FunctionTransformer\(np.log1p\) -tci -1

echo "NEURAL NETWORK TRAINING"
echo "NEURAL NETWORK TRAINING for HUMAN EXPERIMENT"
# python notebooks/4.0-pg-model-training.py -exp exper_melanoma -loc processed -ds reference_log1p.pck -pbk pbk_layer_hsa.txt -split StratifiedKFold -nncv NN -save True
# python notebooks/4.0-pg-model-training.py -exp exper_melanoma -loc processed -ds reference_log1p.pck -pbk pbk_layer_hsa.txt -split train_test_split -nncv NN -save True

echo "NEURAL NETWORK TRAINING for MOUSE EXPERIMENT"
# python notebooks/4.0-pg-model-training.py -exp exper_mouse -loc processed -ds mouse_training_sw_StandardScaler.pck -pbk pbk_layer_mmu.txt -split KFold -nncv NN
# python notebooks/4.0-pg-model-training.py -exp exper_mouse -loc processed -ds mouse_training_sw_StandardScaler.pck -pbk pbk_layer_mmu.txt -split train_test_split -nncv NN -save True

echo "NEURAL NETWORK TRAINING for PBMC EXPERIMENT"
### python notebooks/4.0-pg-model-training.py -exp exper_pbmc -loc processed -ds Immune_magic_sw_log1p.pck -pbk pbk_layer_hsa.txt -split KFold -nncv NN
# python notebooks/4.0-pg-model-training.py -exp exper_pbmc -loc processed -ds Immune_sw_log1p.pck -pbk pbk_layer_hsa.txt -split KFold -nncv NN



echo "CROSS VALIDATION"
echo "CROSS VALIDATION for HUMAN EXPERIMENT"
# python notebooks/5.0-pg-cross-validation.py -exp exper_melanoma -loc processed -ds reference_query_log1p.pck -pbk pbk_layer_hsa.txt -nncv CV

echo "CROSS VALIDATION for PBMC EXPERIMENT"
# python notebooks/5.0-pg-cross-validation.py -exp exper_pbmc -loc processed -ds Immune_sw_log1p.pck -pbk pbk_layer_hsa.txt -nncv CV

echo "ENCODING INFORMATION"
echo "MELANOMA"
# python notebooks/4.0-pg-model-training.py -exp exper_melanoma -loc processed -ds reference_log1p.pck -pbk pbk_layer_hsa.txt -split KFold -nncv NN -model False
echo "MOUSE"
# python notebooks/4.0-pg-model-training.py -exp exper_mouse -loc processed -ds mouse_training_sw_StandardScaler.pck -pbk pbk_layer_mmu.txt -split train_test_split -nncv NN -model True
echo "IMMUNE"
# python notebooks/4.0-pg-model-training.py -exp exper_immune -loc processed -ds exper_immune_raw_sw_log1p.pck -pbk pbk_layer_hsa.txt -split train_test_split -nncv NN -model True

python notebooks/4.0-pg-model-training.py -exp exper_immune -loc processed -ds Fig3g_log1p.pck -pbk pbk_layer_hsa.txt -split train_test_split -nncv NN -model True

python notebooks/4.0-pg-model-training.py -exp exper_immune -loc processed -ds Fig3g_log1p.pck -pbk pbk_layer_hsa.txt -split StratifiedKFold -nncv NN -model True


echo "LocalOutlierFactor Analysis"
# python notebooks/4.0-pg-model-training.py -exp exper_melanoma -loc processed -ds reference_log1p.pck -pbk pbk_layer_hsa.txt -split train_test_split -nncv NN -model True





date