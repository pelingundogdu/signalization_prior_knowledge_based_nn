data/external
==============================

==============================================================================================================================
FOLDER NAME                     : exper_pbmc
URL                             : https://github.com/PaulingLiu/SciBet/tree/master/example.data
INFO                            : Shared by author, ***no information about reproccessing steps***
implemented into author version : ***sample wise + log1p***

==============================================================================================================================
FOLDER NAME                     : exper_immune
URL                             : http://scibet.cancer-pku.cn/dataset/Fig3g.data.rds.gz
INFO                            : Shared the ***multiple*** pre-proccessed version by authors.
                                  "processed including integration, normalization, clustering, embedding and re-annotation"
implemented into author version : ***Nothing***

==============================================================================================================================
FOLDER NAME                     : exper_melanoma
URL                             : http://scibet.cancer-pku.cn/reference.rds.gz
                                  http://scibet.cancer-pku.cn/query.rds.gz  
INFO                            : Shared by author, ***no information about reproccessing steps***
                                  assuming TPM, pseudo value one to handle 0 values + filtering gene space
                                  because the shared version is different than TPM version shared in GEO website
                                  GEO TPM version ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978
implemented into author version : ***log1p***

==============================================================================================================================
FOLDER NAME                     : exper_mouse
URL                             : http://sb.cs.cmu.edu/scnn/
INFO                            : Shared the ***TPM*** normalization version by authors.
implemented into author version : ***sample wise and gene normalization***
                                  (old version ***StandarScaler***)

==============================================================================================================================
**The final version of dataset after our implementations which defined in "implemented into author version" are located into data/proccessd/{EXPERIMENT_NAME}**