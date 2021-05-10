#!/usr/bin/env Rscript

# DESCRIPTION
# -----------
#     Converting entrex id value into gene symbol

# USAGE 
# -----
#     [PROJECT_PATH]/$ Rscript scripts/pathway_layer_data/1.4-pg-gene-id-entrez-converter.r -sp {SPECIES} -src {SOURCE} -ga {GENOME_ANNOTATION}

# RETURN
# ------
#     entrez_and_symbol.csv : csv file
#         entrez id and gene symbol conversion

# EXPORTED FILE(s) LOCATION
# -------------------------
#     ./data/processed/hsa/hipathia/entrez_and_symbol.csv

library(package = 'argparse', quietly = TRUE) # getting given argument

gene_id_entrez_converter <- function(species, source, genome_annotation){
    BiocManager::install(genome_annotation)
    library(genome_annotation, character.only = TRUE) # genome wide annotation
    
    library(package = 'reticulate', quietly = TRUE) # embedding a Python session within R session
    library('AnnotationDbi')  # the gene name conversion

    # Enabling Python environment
    # use_python("/opt/anaconda3/bin/python")
    use_virtualenv("gpu_env")

    # importing py script
    loaded_script<-py_run_file('./scripts/config.py')
    po <- py_run_file(file.path(loaded_script$DIR_CONFIG, 'path_scripts.py'))

    # creating output folders
    output_folder_processed = po$define_folder(file.path(loaded_script$DIR_DATA_PROCESSED, species, source))

    # entrez and symbol informatinon
    entrez_keys <- keys(eval(parse(text = genome_annotation)), keytype="ENTREZID")
    # entrez_name_pair <- select(org.Hs.eg.db, keys=mmu_entrez, columns=c("ENTREZID","SYMBOL"), keytype="ENTREZID")

    entrez_symbol_pair <- select(eval(parse(text = genome_annotation)), keys=entrez_keys, columns=c("ENTREZID","SYMBOL"), keytype="ENTREZID")
    colnames(entrez_symbol_pair) = c('gene_id', 'symbol')

    ## pathway genes list
    df_h <- read.table(paste0(output_folder_processed, 'gene_list.csv'), quote='\"', comment.char='')
    colnames(df_h) = c('gene_id')
    print(paste0('gene list head 5 - ', head(df_h)))

    df_h_es = merge(df_h, entrez_symbol_pair, by='gene_id', all.x='True')
    df_h_es <- df_h_es[which(is.na(df_h_es$symbol) == FALSE ), ]

    write.table(df_h_es, paste0(output_folder_processed,'entrez_and_symbol.csv'),sep=',',row.names = FALSE)

    print(paste0('THE length of entrez_and_symbol_list, ', nrow(df_h_es) ))
    print(paste0('FILE Exported!! - ', output_folder_processed,'entrez_and_symbol.csv'))

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
    
    if( is.null(args$genome_annotation) ){
        print("ERROR!! Please specify genome annotation package")
        quit(status=1)
    }
    
    gene_id_entrez_converter(args$species, args$source, args$genome_annotation)
}

if (getOption('run.main', default=TRUE)) {
   main()
}