# C. Lin, S. Jain, H. Kim, and Z. Bar-Joseph, “Using neural networks for reducing 
# the dimensions of single-cell RNA-Seq data,” Nucleic Acids Res, vol. 45, no. 17, 
# pp. e156–e156, Sep. 2017, doi: 10.1093/nar/gkx681.

# Code modified for this project.
import os, sys
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIR_ROOT = os.path.dirname(os.path.abspath('__file__'))
print(f'DIR_ROOT {DIR_ROOT} --retrieval.py')
os.chdir(DIR_ROOT)
sys.path.append(DIR_ROOT)


import os
import re
import sys
import dotenv
import numpy as np
import datetime as dt
import glob
import itertools
from collections import defaultdict

import scripts.config as src
from scipy.spatial import distance

important_folder=os.path.abspath('./data/external/exper_mouse/')
print('IMPORTANT FOLDER, ',important_folder)
# data_file_name = "3-33_integrated_retrieval_set.txt"
# data_file_name = "TPM_mouse_7_8_10_PPITF_gene_9437.txt"

retrieval_topcell = 100
retrieval_map = 1

project_dir = os.path.abspath(dotenv.find_dotenv())
project_dir = os.path.dirname(project_dir)
data_dir = os.path.join(project_dir, "data")
# data_file_path = os.path.join(important_folder, data_file_name)
# print('\nFILE for USING RETRIEVAL ANALYSIS IS ',data_file_path)

def AvgPrecision(ans, pred):
    # ans is a single integer denoting the class
    # pred is a vector of the retrieved items
    correct = 0
    total = 0
    ret = 0.0
    for p in pred:
        total += 1
        if p == ans:
            correct += 1
            ret += correct / float(total)
    if correct > 0:
        ret /= float(correct)
    return ret

def MeanAvgPrecision(anss, preds):
    ret = 0.0
    for ans, pred in zip(anss, preds):
        ret += AvgPrecision(ans, pred)
    ret /= float(len(anss))
    return ret


def load_integrated_data(
    filename,
    landmark=False,
    sample_normalize=True,
    gene_normalize=False,
    ref_gene_file=None,
    log_trans=False,
):
    all_data = []
    all_sample_ID = []
    labeled_sample_ID = []
    labeled_data = []
    unlabeled_data = []
    gene_names = []
    all_label = []
    labeled_label = []
    
    sample_normalize = bool(int(sample_normalize))

    if ref_gene_file != "all":
        lines = open(ref_gene_file).readlines()
        group_genes = []
        if len(lines) == 1:
            group_genes = lines[0].split("\t")
        else:
            for line in lines:
                group_genes.append(line.replace("\n", "").split("\t")[0].lower())

    lines = open(filename).readlines()
    Sample_ID = lines[0].replace("\n", "").split("\t")[1:]
    labels = lines[1].replace("\n", "").split("\t")[1:]
    all_weights = lines[2].replace("\n", "").split("\t")[1:]
    all_sample_ID = Sample_ID

    label_index = [i for i, val in enumerate(labels) if val != "None"]
    unlabeled_index = [i for i, val in enumerate(labels) if val == "None"]
    labeled_sample_ID = [
        all_sample_ID[i] for i, val in enumerate(labels) if val != "None"
    ]
    unlabeled_sample_ID = [
        all_sample_ID[i] for i, val in enumerate(labels) if val == "None"
    ]
    unlabeled_weights = [
        all_weights[i] for i, val in enumerate(labels) if val == "None"
    ]
    labeled_weights = [all_weights[i] for i, val in enumerate(labels) if val != "None"]

    label_unique_list = ["None"] + list(
        set([val for i, val in enumerate(labels) if val != "None"])
    )

    for lab in labels:
        all_label.append(label_unique_list.index(lab))
        if lab != "None":
            labeled_label.append(label_unique_list.index(lab))

    sum_all_data=[]
    for line in lines[3:]:
        splits = line.replace("\n", "").split("\t")
        gene = splits[0]
        
        sum_all_data.append(list(itertools.chain(np.array(splits[1:], dtype="float") ) ))
        if ref_gene_file != "all" and gene not in group_genes:
            continue
        gene_names.append(gene)
        all_data.append(splits[1:])
    all_data = np.array(all_data, dtype="float32")
    
#     if sample_normalize:
#         print('SAMPLE WISE NORMALIZATION APPLIED!!')
#         s = np.array(sum_all_data)[:, :].astype('float').sum(axis=0)
#         all_data = all_data / s * 1000000

    if sample_normalize:
        print('SAMPLE WISE NORMALIZATION APPLIED!!')
        for j in range(all_data.shape[1]):
            s = np.sum(all_data[:, j])
            if s == 0:
                s = 0
#                 print ('normalize sum==0: sample',j)
            else:
                all_data[:, j] = all_data[:, j] / s * 1000000
                
    if log_trans:
        print('LOG1P NORMALIZATION APPLIED!!')
        all_data = np.log(all_data + 1)
        
    if gene_normalize:
        print('GENE WISE NORMALIZATION APPLIED!!')
        for j in range(all_data.shape[0]):
            mean = np.mean(all_data[j, :])
            std = np.std(all_data[j, :])
            if std == 0:
                std = 0
                # print 'gene_normalize: std==0 data: ',j,mean,std
            else:
                all_data[j, :] = (all_data[j, :] - mean) / std
            # print all_data[j,:]
            
    labeled_data = np.zeros((all_data.shape[0], len(label_index)), dtype="float32")
    unlabeled_data = np.zeros(
        (all_data.shape[0], len(unlabeled_index)), dtype="float32"
    )
    count = 0

    for i in label_index:
        labeled_data[:, count] = all_data[:, i]
        count += 1
    # print count
    count = 0
    for i in unlabeled_index:
        unlabeled_data[:, count] = all_data[:, i]
        count += 1
    all_label = np.array(all_label)
    labeled_label = np.array(labeled_label)

    all_data = np.transpose(all_data)
    labeled_data = np.transpose(labeled_data)
    unlabeled_data = np.transpose(unlabeled_data)

    all_weights = np.transpose(all_weights)
    labeled_weights = np.transpose(labeled_weights)
    unlabeled_weights = np.transpose(unlabeled_weights)
    
    print('    Lenght of gene list, ', len(all_data))
    print('    all_data shape, ', all_data.shape)
    print('    sum_all_data shape, ', np.asarray(sum_all_data).shape)
    

    return (
        all_data,
        labeled_data,
        unlabeled_data,
        label_unique_list,
        all_label,
        labeled_label,
        all_weights,
        labeled_weights,
        unlabeled_weights,
        all_sample_ID,
        labeled_sample_ID,
        unlabeled_sample_ID,
        gene_names,
    )


def compute_retrieval_scores(model_path, n_epochs, analysis, snorm, ref_gene_file, sub_output_dir, out_file, gnorm, scaler, data_file_path, data_training):
    # nearest neighbor retrieval things
    # data_file_name='hannah_mouse_data/TPM_6_8_9_15_25_41_44_45_46_.txt'
    # model_name='3layer_SN1_GN1_BS32_hls100_mls696_seed0_classifier_merge0_tanh'
    # nn_iteration=100
    print('compute_retrieval_scores ----->', data_file_path)
    (
        all_data,
        labeled_data,
        unlabeled_data,
        label_unique_list,
        all_label,
        labeled_label,
        all_weights,
        labeled_weights,
        unlabeled_weights,
        all_sample_ID,
        labeled_sample_ID,
        unlabeled_sample_ID,
        gene_names,
    ) = load_integrated_data(
        data_file_path,
        sample_normalize=snorm,
        gene_normalize=gnorm,
        log_trans=0,
        ref_gene_file=ref_gene_file,
    )
    
    code, time = encode_data(model_path, n_epochs, analysis, all_data, scaler, data_training)
    print(all_data)
    print("all_data.shape: ", all_data.shape)
    # code = get_nn_code(model_name,nn_iteration,all_data)
    # code=transform_data
    print("code.shape: ", code.shape)
    print("all_weights: ", all_weights)
#     print("\n".join(label_unique_list))
    verify_lab = [
        "2cell",
        "4cell",
        "ICM",
        "zygote",
        "8cell",
        "ESC",
        "lung",
        "TE",
        "thymus",
        "spleen",
        "HSC",
        "neuron",
    ]

    for index, lab in enumerate(label_unique_list):
        if "cortex" in lab:
            label_unique_list[index] = "neuron"
        if "CNS" in lab:
            label_unique_list[index] = "neuron"
        if "brain" in lab:
            label_unique_list[index] = "neuron"
        for vl in verify_lab:
            if vl in lab:
                label_unique_list[index] = vl
    dataset_sets = map(str, sorted(map(int, list(set(all_weights)))))
    ntop = 10
    # vtop=100

#     out_file = os.path.basename(model_path)
#     out_file = os.path.splitext(out_file)[0]
    out_file_1 = os.path.join(sub_output_dir, out_file + "_retrieval.csv")
    out_file_1 = os.path.abspath(out_file_1)
    out_ret = open(out_file_1, "w")
    out_ret.write(
        "dataset,sample,celltype,retrieval #cell in top100, total #cell,retrieval_ratio\n"
    )
    out_file_2 = os.path.join(sub_output_dir, out_file + "_retrieval_summary.csv")
    out_file_2 = os.path.abspath(out_file_2)
    out_ret2 = open(out_file_2, "w")
    out_ret2.write("dataset,celltype,#cell,mean retrieval ratio\n")
    for ds in dataset_sets:
        meindex = np.where(all_weights == ds)
        nmeindex = np.where(all_weights != ds)
        # isme= all_data[meindex]
        # isnme= all_data[nmeindex]
        isme = code[meindex]
        ####################################################################################################
#         print("isme")
#         print(isme)
#         print("isme.shape")
#         print(isme.shape)
        isnme = code[nmeindex]
        dismat = distance.cdist(isme, isnme, "euclidean")
#         print(dismat.shape)
        ####################################################################################################
        cell_dict = defaultdict(lambda: [])
        for index, row in enumerate(dismat):
            # print index, row
            now_label = label_unique_list[all_label[meindex[0][index]]]
            if now_label not in verify_lab:
                continue
            now_sample = all_sample_ID[meindex[0][index]]
#             aprint("now dataset: ", ds)
#             aprint("now_sample: ", now_sample)
#             aprint("now_label: ", now_label)
            sort_index = np.argsort(row)
            temp_lab = []
            temp_dist = []
            temp_set = []
            total_vl = len(
                [
                    label_unique_list[all_label[nmeindex[0][x]]]
                    for x in range(len(row))
                    if label_unique_list[all_label[nmeindex[0][x]]] == now_label
                ]
            )
            if retrieval_topcell > 0:
                total_vl = retrieval_topcell
            for si in sort_index[:total_vl]:
                temp_set.append(all_weights[nmeindex[0][si]])
                temp_dist.append(row[si])
                temp_lab.append(label_unique_list[all_label[nmeindex[0][si]]])
            vtop_vl = len(
                [temp_lab[x] for x in range(len(temp_lab)) if temp_lab[x] == now_label]
            )
#             print("# of hit in top 100 cells: ", vtop_vl)
#             print("total # of cell in reference cells:", total_vl)
            ratio = vtop_vl / float(total_vl)
#             print("ratio: ", ratio)
            if retrieval_map:
                AP = AvgPrecision(now_label, temp_lab)
#                 aprint("AP=", AP)
                ratio = AP
            cell_dict[now_label].append(ratio)
            out_ret.write(
                str(ds)
                + ","
                + now_sample
                + ","
                + now_label
                + ","
                + str(vtop_vl)
                + ","
                + str(total_vl)
                + ","
                + str(ratio)
                + "\n"
            )
#             print("top ", ntop, " neighbor distance, label, and dataset: ")
#             print(temp_dist[:ntop])
#             print(temp_lab[:ntop])
#             print(temp_set[:ntop])
        for key, val in cell_dict.items():
#             print(key, np.mean(val))
            out_ret2.write(
                str(ds)
                + ","
                + key
                + ","
                + str(len(val))
                + ","
                + str(np.mean(val))
                + "\n"
            )
    
    return time

def encode_from_saved_model(model, data):
    '''
    Parameters
    ----------
    model : model
        The trained model
    data : dataframe
        The data which is performing the retriefval analysis

    Returns
    -------
    code : numpy array
        The encoding information for given model
    '''
    
    import tensorflow.keras.backend as K

    K.clear_session()
    
    print(model.summary())
    code=model.predict(data)
    
    K.clear_session()
    return code

def encode_pca(data, n_epochs, scaler, analysis, data_training):
    """Encode using l1-driven PCA.

    Parameters
    ----------
    data : np.array
        Samples per genes dataframe

    Returns
    -------
    np.array
        Encoded dataframe
    """

    from sklearn.decomposition import SparsePCA, PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    
    if scaler == True:
        print('STANDARSCALER APPLIED!!')
        data = StandardScaler().fit_transform(data)
    
    pca = PCA(n_components=n_epochs)
    
    if re.search('pretrained', analysis)==None:
        print('PCA-ML applied!!')
        code = pca.fit_transform(data)
    else:
        print('PCA-pretrained!!')
        print('PCA fitted with training dataset and .transform for retrieval dataset!!')
        pca.fit(data_training)
        code = pca.transform(data)
    return code


def encode_data(model_path, n_epochs, analysis, data, scaler, data_training):
    if re.search('pca', analysis):
        print('\nRETRIEVAL ANALYSIS IS PERFORMING for PCA')
        code = encode_pca(data, n_epochs, scaler, analysis, data_training)
    else:
        code = encode_from_saved_model(model_path, data)

    return code, dt.datetime.now()


def main(model_path, n_epochs, analysis, snorm, ref_gene_file, gnorm, scaler, design_name, data_training):

    try:
        print(f'model_path, {model_path}\nn_epoch, {n_epochs}\nanalysis, {analysis}\nsnorm, {snorm}\nref_gene_file, {ref_gene_file}\ngnorm, {gnorm}\nscaler, {scaler}\ndesign_name, {design_name}')
#         \ndata_training, {data_training.shape}
        
        time_start = dt.datetime.now()#.strftime('%d/%m/%y, %H:%M:%S')

        if re.search('signaling', analysis):
            data_file_name = "3-33_integrated_retrieval_set_signaling.txt"
        elif re.search('metsig', analysis):
            data_file_name = "3-33_integrated_retrieval_set_metsig.txt"
        else:
            data_file_name = "3-33_integrated_retrieval_set.txt"

        print(data_file_name)
        data_file_path = os.path.join(important_folder, data_file_name)
        print('\nFILE for USING RETRIEVAL ANALYSIS IS ',data_file_path)

        experiment_name = 'mouse'
        sub_output_dir=f'./reports/retrieval/exper_'+experiment_name
    #     print(f'snorm type {type(eval(snorm))}, gnorm type {type(gnorm)}')
        print(f'snorm --> {snorm}')
        print(f'gnorm --> {gnorm}')
        print(f'scaler --> {scaler}')
    #     print(f'bool(snorm) --> {bool(eval(snorm))}')

        if not os.path.exists(os.path.join(sub_output_dir)):
            src.define_folder(os.path.join(sub_output_dir))

        time_encoding = compute_retrieval_scores(model_path, n_epochs, analysis, snorm, ref_gene_file, sub_output_dir, design_name, gnorm, scaler, data_file_path, data_training)

    except NameError as e:
        print(e)
        
    except TypeError as e:
        print(e)
        
if __name__ == "__main__":
    _, model_path, n_epochs, analysis, snorm, ref_gene_file, gnorm, scaler, design_name, data_training= sys.argv

    main(model_path, n_epochs, analysis, bool(eval(snorm)), ref_gene_file, bool(eval(gnorm)), bool(eval(scaler)), design_name, data_training)
    

