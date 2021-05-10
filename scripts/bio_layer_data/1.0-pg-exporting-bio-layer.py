#!/usr/bin/python

'''
DESCRIPTION 
-----------
    Creating prior biological knowledge matrix. The matrix will implemented into NN design in first hidden layer.

USAGE
-----
    [PROJECT_PATH]/$ python src/data/1.6-pg-creating-ppitf.py

RETURN
------
    bio_layer_{BIO}.txt : txt file
        The gene x pathway matrix which stores the prior biological knowledge
        
EXPORTED FILE(s) LOCATION
-------------------------
    ./data/processed/pbk_layer_hsa.txt
    ./data/processed/pbk_layer_mmu.txt
'''
import os, argparse, sys
sys.path.append('./')
import pandas as pd

# HSA - HIPATHIA INFORMATION (SIGNALIZATION PATHWAYS)
df_hsa_hipathia = pd.read_csv('./data/processed/hsa/hipathia/bio_layer_hsa.txt')
print('hsa/hipathia ->', df_hsa_hipathia.shape)
df_hsa_hipathia.head()

# MMU - HIPATHIA INFORMATION (SIGNALIZATION PATHWAYS)
df_mmu_hipathia = pd.read_csv('./data/processed/mmu/hipathia/bio_layer_mmu.txt')
print('mmu/hipathia ->', df_mmu_hipathia.shape)
df_mmu_hipathia.head()

# MMU - PPI INFORMATION
df_ppi_org = pd.read_csv('./data/raw/ppi_tf/ppi_weight.txt')
print('mmu/ppi ->', df_ppi_org.shape)

# MMU - TF INFORMATION
df_tf_org = pd.read_csv('./data/raw/ppi_tf/tf_weight.txt')
print('mmu/tf ->', df_tf_org.shape)

# MMU - USED GENE LIST
df_gene = pd.read_csv('./data/raw/ppi_tf/NN_training_PPITF_9437_genes.txt', sep='\t')
print(df_gene)

df_ppi = df_ppi_org[df_ppi_org['symbol'].isin(df_gene.columns)]
print('mmu/ppi with NN_training gene list ->', df_ppi.shape)

df_tf = df_tf_org[df_tf_org['symbol'].isin(df_gene.columns)]
print('mmu/tf with NN_training gene list ->', df_tf.shape)

df_ppitf = pd.merge(left=df_tf, right=df_ppi, on=['symbol'], how='outer').fillna(0.0)
print('mmu/ ppi+tf ->', df_ppitf.shape)


### EXPORTING
df_hsa_hipathia.to_csv('./data/processed/pbk_layer_hsa_sig.txt', index=False)
print('HSA-signalization-pathways saved into ./data/processed/pbk_layer_hsa_sig.txt')

df_mmu_hipathia.to_csv('./data/processed/pbk_layer_mmu_sig.txt', index=False)
print('MMU-signalizaiton-pathways saved into ./data/processed/pbk_layer_mmu_sig.txt')

df_ppi.to_csv('./data/processed/pbk_layer_mmu_ppi.txt',index=False)
print('MMU-ppi saved into ./data/processed/pbk_layer_mmu_ppi.txt')

df_ppitf.to_csv('./data/processed/pbk_layer_mmu_ppitf.txt', index=False)
print('MMU-ppi+tf saved into ./data/processed/pbk_layer_mmu_ppitf.txt')

