# C. Lin, S. Jain, H. Kim, and Z. Bar-Joseph, “Using neural networks for reducing 
# the dimensions of single-cell RNA-Seq data,” Nucleic Acids Res, vol. 45, no. 17, 
# pp. e156–e156, Sep. 2017, doi: 10.1093/nar/gkx681.

import os
import re
import sys
import dotenv
import numpy as np
import pandas as pd
import itertools

dotenv.load_dotenv()
project_dir = os.path.abspath(dotenv.find_dotenv())
# project_dir = os.path.dirname(project_dir)

def main(path):
    all_retrieval = []

    # filter by cell_types used in table 4 in [1]
    cell_types = [
        "HSC",
        "4cell",
        "ICM",
        "spleen",
        "8cell",
        "neuron",
        "zygote",
        "2cell",
        "ESC",
    ]
    
    if not os.path.exists(os.path.join(project_dir, path)):
        print('EXECUTE retrieval_analysis.sh FIRST!!')
    output_dir = os.path.join(project_dir, path)
    print('\nOUTPUT directory, ', output_dir)
#     print(sorted(os.listdir(output_dir)))
    print('\nexperiment_dir, ', os.listdir(output_dir))
    retrieval_summary_list = sorted([ string for string in os.listdir(output_dir) if re.search('\w+_retrieval.csv', string) ])
    print('\nretrieval_summary_list, ',retrieval_summary_list)
    for i_retrieval in retrieval_summary_list:
        retrieval_dir = os.path.join(output_dir, i_retrieval)
        print('retrieval_dir, ',retrieval_dir)
        # manipulating experiment name, for analysis purposes

        summary_table = pd.read_csv(retrieval_dir, sep=",", index_col=0)        
        summary = summary_table.groupby("celltype")[summary_table.columns[-1]].agg('mean').loc[cell_types]
        summary["mean"] = summary.mean()
        
        all_retrieval.append(list(itertools.chain( [i_retrieval], np.array(pd.DataFrame(summary).T.iloc[0].values) )))

    col_list = ['model', ] + cell_types + ['mean']
    df_summary = pd.DataFrame(all_retrieval)
    df_summary.columns = col_list
    print(df_summary)
    df_summary.to_csv(os.path.join(output_dir+'/retrieval_all.csv'))
    
    
if __name__ == "__main__":
    _, path = sys.argv

    main(path)
