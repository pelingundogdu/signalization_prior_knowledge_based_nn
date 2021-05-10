#!/usr/bin/env Rscript

# DESCRIPTION
# -----------
#     Exporting gene list of signaling pathway from hipathia.

# USAGE 
# -----
#     [PROJECT_PATH]/$ Rscript scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r -sp {SPECIES} -src {SOURCE}

# RETURN
# ------
#     gene_list_all.csv : csv file
#         The gene information of all pathways
#     gene_list.csv     : csv file
#         The gene information of pathway which removed disease related ones

# EXPORTED FILE(s) LOCATION
# -------------------------
#     ./data/raw/hsa/hipathia/gene_list_all.csv
#     ./data/processed/hsa/hipathia/gene_list.csv

library(package = 'argparse', quietly = TRUE) # getting given argument

gene_from_hipathia <- function(species, source){

    library(package = 'tidyverse', quietly = TRUE)
    library(package = 'hipathia', quietly = TRUE) # hipathia package
    library(package = 'reticulate', quietly = TRUE) # embedding a Python session within R session

    print(paste0('hipathia pathways are exporting for ', species))
    # Enabling Python environment
    # use_python("/opt/anaconda3/bin/python")
    use_virtualenv("gpu_env")

    # importing py script
    loaded_script<-py_run_file('./scripts/config.py')
    po <- py_run_file(file.path(loaded_script$DIR_CONFIG, 'path_scripts.py'))

    # creating output folders
    output_folder_raw = po$define_folder(file.path(loaded_script$DIR_DATA_RAW, species, source))
    output_folder_processed = po$define_folder(file.path(loaded_script$DIR_DATA_PROCESSED, species, source))
    output_folder_detail = po$define_folder(file.path(loaded_script$DIR_DATA_PROCESSED, species, source, 'details'))

    # importing pathway information file
    dataset_hp <- read.table(file.path(loaded_script$DIR_DATA_PROCESSED, species, source, 'pathway_ids_and_names.csv'), sep=',', header=TRUE)

    # loading pathways 
    pathways <- load_pathways(species = species)
    df_pathways = data.frame(pathways$all.labelids)

    # Loaded 146 pathways
    print(paste0('Number of pathway      , ', length(unique(df_pathways$path.id)) ))
    print(paste0('Number of sub-pathways , ', nrow(df_pathways)))
    # [1] "Number of pathway      , 146"
    # [1] "Number of sub-pathways , 6826"

    print(' ')
    print('SIGNALING PATHWAYS WITHOUT CANCER/DISEASE')
    df_pathways = merge(dataset_hp, df_pathways)
    print(paste0('Number of pathway      , ', nrow(dataset_hp)))
    print(paste0('Number of sub-pathways , ', nrow(df_pathways)))
    list_path_id = unique(df_pathways$path.id)
    # [1] "SIGNALING PATHWAYS WITHOUT CANCER/DISEASE"
    # [1] "Number of pathway      , 93"
    # [1] "Number of sub-pathways , 4502"

    # GETTING GENE LIST

    # Creating empty data frame to store genes
    df_gene <- data.frame(pathways$all.genes)
    # updating column name as 'entrez'
    colnames(df_gene) = c('entrez')
    l_gene_final<-c()

    for (all_pathways_ in c(1:length(list_path_id))) {
    # for (all_pathways_ in c(1:2)) {
        df_merge = df_gene
        l_main<-c()
    #     number of sub-pathway in 
        length_subpathways = length(pathways$pathigraphs[[list_path_id[all_pathways_]]]$effector.subgraphs)
    #     details for sub-pathway
        for (sub_pathways_ in c(1:length_subpathways)) {
            l_sub<-c()
            genes_circuits = V(pathways$pathigraphs[[list_path_id[all_pathways_]]]$effector.subgraphs[[sub_pathways_]])$genesList
            sub_path_name = names(pathways$pathigraphs[[list_path_id[all_pathways_]]]$effector.subgraphs[sub_pathways_][1])
    #         name of sub-patway
    #         print(sub_path_name)
            for (sub_circuits in c(1:length(genes_circuits))){
                for (genes_ in c(1:length(genes_circuits[sub_circuits][1][[1]]))){
                    gene_value = genes_circuits[sub_circuits][1][[1]][genes_]
                    if (!is.na(gene_value) && gene_value != '/' && (gene_value == 'NA') == FALSE){
                        l_sub <-append(l_sub,gene_value)
                    }
                }
                l_sub = unique(l_sub)
            }
    #             Combining genes obtaining from sub-pathways
            if (length(l_sub) != 0) {
                l_main<-append(l_main, l_sub)
                df_temp <- data.frame(l_sub, 1)
                names(df_temp) = c('entrez', sub_path_name)
    #             print(df_temp)
            }
    #         Inner join
            df_merge = merge(df_temp,df_merge, by='entrez', all=T)
        }
        l_main = unique(l_main)
    #     Assigning all NA's as 0
        df_merge[!is.na(df_merge)] 
        indices_genes <- as.vector(which(df_merge$entrez %in% l_main, arr.ind = TRUE))
        df_path_genes <- (df_merge[c(indices_genes), ])
        rownames(df_path_genes) <- 1:nrow(df_path_genes)
        l_gene_final = append(l_gene_final, l_main)

    #     Exporting gene set of each pathways (93 txt file for using pathways for hsa)
        write.table(df_path_genes, paste0(output_folder_detail,list_path_id[all_pathways_],'_gene_list.txt'), sep=',', row.names=FALSE)
    }

    write.table(df_gene$entrez[(df_gene$entrez) != 'NA'], paste0(output_folder_raw,'gene_list_all.csv'),sep=',',row.names = FALSE)
    write.table(unique(l_gene_final), paste0(output_folder_processed,'gene_list.csv'),sep=',', row.names = FALSE)

    print('THE GENES LIST EXPORTED!!!')
    print(paste0('     ',output_folder_raw,'gene_list_all.csv'))
    print(paste0('     ',output_folder_processed,'gene_list.csv'))
}

main <- function() {
    parser <- ArgumentParser()
    args <- commandArgs(trailingOnly = TRUE)

    parser$add_argument('-sp', '--species', help='specify the species, the location of species in ./data/raw/{SPECIES}')
    parser$add_argument('-src', '--source', help='specify the source, the location of source in ./data/raw/{SPECIES}/{SOURCE}')
    parser$add_argument('-ga', '--genome_annotation', help='specify genome wide annotition package', default=NULL)

    if(length(args)==0){
        parser$print_help()
        print("ERROR!! Please give species and source information.")
        quit(status=1)
    }

    args <- parser$parse_args()
    
    gene_from_hipathia(args$species, args$source)
}

if (getOption('run.main', default=TRUE)) {
   main()
}