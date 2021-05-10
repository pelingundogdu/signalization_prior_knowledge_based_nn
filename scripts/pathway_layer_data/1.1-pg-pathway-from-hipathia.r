#!/usr/bin/env Rscript

# DESCRIPTION
# -----------
#     Exporting signaling pathway information from hipathia.

# USAGE 
# -----
#     [PROJECT_PATH]/$ Rscript scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r -sp {SPECIES} -src {SOURCE}

# RETURN
# ------
#     pathway_ids_and_names.csv : csv file
#         The information about pathway id and pathway name 

# EXPORTED FILE(s) LOCATION
# -------------------------
#     ./data/raw/hsa/hipathia/pathway_ids_and_names.csv

library(package = 'argparse', quietly = TRUE) # getting given argument

pathway_from_hipathia <- function(species, source){

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

    # create ../data/input/pathways/ folder
    output_folder = po$define_folder(file.path(loaded_script$DIR_DATA_RAW, species, source))
    print(output_folder)
    # loading pathways 
    pathways <- load_pathways(species = species)
    df_pathways = data.frame(pathways$all.labelids)
    paste0('Number of sub-pathways, ', nrow(df_pathways))

    control1 = nrow(df_pathways[c("path.id", "path.name")] %>% distinct(path.id, path.name, .keep_all = TRUE))
    control2 = length(unique(df_pathways$path.id))

    # checking consistency of the pathway number
    if (control1 == control2){
        paste0('Number of pathways', control1)    
        df_export = df_pathways[c("path.id", "path.name")] %>% distinct(path.id, path.name, .keep_all = TRUE)
    #     exporting path
        output_file = paste0(output_folder,'pathway_ids_and_names.csv')
        write.table(x=df_export, file = output_file, sep=',', row.names = FALSE, col.names = TRUE)    
        paste0('hiPathia pathways EXPORTED!! - ', output_file)
    } else{
        print('PROBLEM in pathway numbers!!')
    }
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
    
    pathway_from_hipathia(args$species, args$source)
}

if (getOption('run.main', default=TRUE)) {
   main()
}