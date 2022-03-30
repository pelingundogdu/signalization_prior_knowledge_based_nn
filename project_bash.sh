# hsa --> homo sapiens(human), org.Hs.eg.db
#  Genome Wide annotation
# https://bioconductor.org/packages/3.12/data/annotation/

############################## 1.EXPORTING PRIOR BIOLOGICAL KNOWLEDGE LAYER (START) ###############################
# Exporting pathway info
echo "Exporting signaling pathway information from hipathia --- scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r executing..."
Rscript scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r -sp hsa -src hipathia
# Remowing disease related pathways
echo "Processing the pathway list, removing disease related pathways --- scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py executing..."
python scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py -sp hsa -src hipathia
# exporting gene list
echo "Exporting gene list based on processed pathway list in 1.2-pg-remove-disease-cancer.py --> scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r executing..."
Rscript scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r -sp hsa -src hipathia
# Converting entrez id into gene symbol
echo "Converting entrez id value into gene symbol -->  scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r executed!!"
Rscript scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r -sp hsa -src hipathia -ga org.Hs.eg.db
# Creating pathway-gene matrix
echo "Creating prior biological knowledge information to include the nn design in first hidden layer --> scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py executing..."
python scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py -sp hsa -src hipathia
echo "PATHWAY INFORMATION EXPORTED!!!"
echo "Exporting data/processed/pbk_layer_{BIO_SOURCE} files"
python scripts/bio_layer_data/1.0-pg-exporting-bio-layer.py

# Creating circuit matrix
echo "Exporting circuits matrix --> scripts/pathway_layer_data/1.6-pg-creating-biological-layer_circuits.py"
python scripts/pathway_layer_data/1.6-pg-creating-biological-layer_circuits.py
############################### 1.EXPORTING PRIOR BIOLOGICAL KNOWLEDGE LAYER (END) ################################

######################################### 2.DATASET PREPROCESSING (START) #########################################
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
    -ofn reference_log1p &&
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_melanoma \
    -ds query.pck \
    -sw False \
    -sc log1p \
    -tci -1 \
    -ofn query_log1p

echo "PREPROCESSING of EXPERIMENT PBMC DATASET"
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_pbmc \
    -ds Immune.pck \
    -sw True \
    -sc log1p \
    -tci -1 \
    -ofn pbmc_sw_log1p

echo "PREPROCESSING of EXPERIMENT IMMUNE DATASET"
python notebooks/2.0-pg-preprocessing-dataset.py \
    -exp exper_immune \
    -ds Fig3g.pck \
    -sw False \
    -sc None \
    -tci -1 \
    -ofn immune_new
########################################## 2.DATASET PREPROCESSING (END) ##########################################

############################################### 3.EXPERIMENT (START) ##############################################
echo "NEURAL NETWORK TRAINING"
# network parameters
optimizer_var='Adam'
activation_var='relu'
tuning='False'
filter_space='False'
# prior biological knowledge detail
pbk_var='pbk_circuit_hsa_sig.txt' #circuits
pbk_info='circuits'

########################################### a.IMMUNE EXPERIMENT (START) ###########################################
echo "IMMUNE EXPERIMENT"
# analysis_var='encoding'
# analysis_var='performance'
analysis_var='evaluate_rskf'
ds_var='processed/exper_immune/immune_new.pck'

# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_1_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning 
    
# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_2_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning
############################################ a.IMMUNE EXPERIMENT (END) ############################################

############################################ b.PBMC EXPERIMENT (START) ############################################
echo "PBMC EXPERIMENT"
analysis_var='evaluate_rskf'
ds_var='processed/exper_pbmc/pbmc_sw_log1p.pck'

# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_1_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning 
    
# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_2_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning
############################################# b.PBMC EXPERIMENT (END) #############################################

######################################### 3c.MELANOMA EXPERIMENT (START) ##########################################
echo "MELANOMA EXPERIMENT"
# analysis_var='None'
analysis_var='encoding'
ds_var='processed/exper_melanoma/reference_log1p.pck'

# 1-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_1_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer False \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning

# 2-LAYER SIGNALING
python notebooks/4.0-pg-model.py \
    -design ${pbk_info}_2_layer \
    -first_hidden_layer_pbk $pbk_var \
    -first_hidden_layer_dense 0 \
    -second_hidden_layer True \
    -optimizer $optimizer_var \
    -activation $activation_var \
    -ds $ds_var \
    -analysis $analysis_var \
    -filter_gene_space $filter_space \
    -hp_tuning $tuning
########################################## 3c.MELANOMA EXPERIMENT (END) ###########################################

################################################ 3.EXPERIMENT (END) ###############################################