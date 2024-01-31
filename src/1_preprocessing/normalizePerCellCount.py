import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import hstack,vstack,csr_matrix
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', default="./data/",
                    help='Location to save pseudobulks data')
parser.add_argument('--filename', 
                        default="adata_peak_matrix.h5",
                    help='Name of the anndata file')
parser.add_argument('--name', default="pseudobulks",
                    help='Name pseudobulks data')



def main():
    args = parser.parse_args()
    dir_path = args.path
    filename = args.filename
    adata_ = ad.read_h5ad(dir_path + filename)
    key_c = "chrm"
    name = args.name
    annot = adata_.var
    celltypes = adata_.obs["celltype"].sort_values().unique()
    print(celltypes)
    savename=name
    subjects = adata_.obs["Sample_num"].unique()
    df_mix = []
    df_labels = []
    separate = []
    nb_cells = []
    newsavename = "cell_count_norm_" + savename
    if True:
        print("processing pseudobulks ...")
        for i,it in enumerate(subjects):
            df_mix.append(pd.read_csv(dir_path 
                + savename + "_pseudobulk_data_subject_%s.csv"%it))
            df_labels.append(pd.read_csv(dir_path + savename + "_labels_data_subject_%s.csv"%it))
            nb_cells.append(np.load(dir_path + savename + "_nb_cells_per_mixtures_%s.npy"%it)) 

        df_mix_t = pd.concat(df_mix)
        df_labels_t = pd.concat(df_labels)
        nb_cells_t = np.concatenate(nb_cells)

        df_mix_t.iloc[:,:-1] = df_mix_t.iloc[:,:-1].values/np.float64(nb_cells_t)[:,None]
        #df_mix_t.to_csv(dir_path + newsavename + "_pseudobulk_data_with_sparse.csv", index=False)
        df_mix_t.to_parquet(dir_path + newsavename + "_pseudobulk_data_with_sparse.parquet.gzip",  compression='gzip', index=False)
        df_labels_t.to_csv(dir_path + newsavename + "_labels_synthsize_bulk_data_with_sparse.csv", index=False)
        np.save(dir_path + newsavename + "_nb_cells_per_mixtures.npy", 
                nb_cells)
        annot.to_csv(dir_path + newsavename + "_annotations.csv")
        df_mix_t = 0
        df_mix = 0
        df_labels_t = 0
        df_labels = 0

    nb_cells = []
    print("processing cell type-specific pseudobulks ...")
    for i,it in enumerate(subjects):
        separate.append(np.load(dir_path + savename 
                        + "_celltype_specific_subject_%s.npz"%it)["mat"])
        nb_cells.append(np.load(dir_path + savename + "_nb_cells_per_mixtures_%s.npy"%it)) 
    nb_cells_t = np.concatenate(nb_cells)
    separate = np.concatenate(separate)
    separate = separate/np.float64(nb_cells_t[:, None, None])
    np.savez_compressed(dir_path + newsavename + "_concatenate_celltype_specific.npz", mat=separate)
if __name__ == "__main__":
    main()

