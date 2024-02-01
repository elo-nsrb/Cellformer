import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm
import pandas as pd
import copy
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
import random

SAMPLE_ID_TEST = "13_1226_SMTG"
SAMPLE_ID_VAL = "13_0038_SMTG"
#MIXTUREFIX = "_pseudobulk_data_with_sparse.csv"
MIXTUREFIX = "_pseudobulk_data_with_sparse.parquet.gzip"
PORTIONFIX = "_labels_synthsize_bulk_data_with_sparse.csv"
SEPARATEFIX = "_concatenate_celltype_specific.npz"
class SeparationDataset(Dataset):
    def __init__(self,  mixtures, portions, 
            separate_signal, celltypes, hdf_dir, partition, 
            in_memory=False,
            data_transform=None, binarize=False, normalize=False,
            binarize_input=False, ratio=False, offset=1, level=None,
            force_rewriting=False, normalize_peaks=False, 
            normalizeMax=False,
            pure=False, cut=False,
            #logtransform=False,
            use_only_all_cells_mixtures=False,
            mean_expression=None, celltype_to_use=None, **kwargs):
        '''
        Initialises a source separation dataset
        :param data: HDF cell data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the cell for each example (subsampling the cell)
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        self.cut = cut
        self.normalizeMax = normalizeMax
        self.logtransform = logtransform
        self.level = level
        os.makedirs(hdf_dir, exist_ok=True)
        self.celltypes = copy.deepcopy(celltypes)
        self.partition = partition

        if self.level is not None:
            self.hdf_dir = os.path.join(hdf_dir, 
                            partition + str(self.level)+ ".hdf5")
        elif use_only_all_cells_mixtures:
            self.hdf_dir = os.path.join(hdf_dir, 
                                    partition 
                                    + str(self.level)
                                    + "only_mixtures" +
                                    ".hdf5")
        elif pure:
            self.hdf_dir = os.path.join(hdf_dir, 
                                    partition 
                                    + str(self.level)
                                    + "pure" +
                                    ".hdf5")

        else:
            self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
            self.level = len(self.celltypes)
        print(self.hdf_dir)
        #print(self.level)
        self.celltypes_to_use = celltype_to_use
        self.in_memory = in_memory
        print(self.celltypes)
        
        if not os.path.exists(self.hdf_dir) or force_rewriting:
            self.mixtures = mixtures
            self.portions = portions
            
            self.cell_transform = data_transform
            (self.celltype,
             separate_signal,
             ) = gatherCelltypes(celltype_to_use, 
                                            separate_signal, self.celltypes)

            #print(self.celltype) 
            #print(self.celltypes_to_use)
            if binarize:
                separate_signal = binarizeSeparateSignal(separate_signal,
                                                    self.celltypes)
            if binarize_input:
                self.mixtures.iloc[:,:-1] = binarizeSignal(mixtures.iloc[:,:-1])

            if normalize_peaks:
                self.mixtures.iloc[:,:-1] = normalizePeaks(mixtures.iloc[:,:-1],
                                                            mean_expression)
                separate_signal = normalizePeaks_signal(separate_signal, 
                                                            mean_expression)

            if normalize:
                #separate_signal = normalizeSignal(separate_signal)
                self.mixtures.iloc[:,:-1]  = normalizeMixture(self.mixtures.iloc[:,:-1] )

            if ratio:
                separate_signal = ratioSignal(separate_signal, 
                                              mixtures.iloc[:,:-1].values,
                                                offset=offset)
            self.label = np.count_nonzero(portions, axis=1)
            # PREPARE HDF FILE

            # Check if HDF file exists already
                # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                #f.attrs["Sample.ID"] = self.mixtures["Sample.ID"].astype('str')
                f.attrs["instruments"] = self.celltypes
                f.attrs["celltypes"] = self.celltypes

                print("Adding atac-seq files to dataset (preprocessing)...")
                real_idx = 0
                for idx, (index, row) in enumerate(tqdm(self.mixtures.iterrows())):
                            mix_cell = np.asarray(row[:-1]).reshape((1,-1)).astype(np.float)
                            source_cells = []
                            source_cells =  separate_signal[idx, :, : ]# np.stack(source_cells, axis=0)
                            assert(source_cells.shape[1] == mix_cell.shape[1])

                            # Add to HDF5 file
                            grp = f.create_group(str(real_idx))
                            grp.create_dataset("inputs", shape=mix_cell.shape,
                                    dtype=mix_cell.dtype, data=mix_cell)
                            grp.create_dataset("targets",
                                        shape=source_cells.shape, 
                                        dtype=source_cells.dtype, 
                                        data=source_cells)
                            lab = [row[-1].encode("ascii", "ignore")]
                            grp.create_dataset("label", 
                                    shape=len(lab),
                                    dtype="S10",
                                        data=lab)
                            grp.attrs["length"] = mix_cell.shape[1]
                            grp.attrs["target_length"] = source_cells.shape[1]
                            real_idx +=1


        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            nb_sample = len(f)
            lengths = [f[str(song_idx)].attrs[
                    "target_length"] for song_idx in range(len(f))]

        self.length = nb_sample
        print("lenght : " + str(self.length))

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        cell = self.hdf_dataset[str(index)]["inputs"][:,:].astype(np.float32)
        targets = self.hdf_dataset[str(index)]["targets"][:,:].astype(np.float32)
        if self.normalizeMax:
            cell, targets = normalizeMaxPeak(cell, targets)
        mix = torch.from_numpy(cell.squeeze())
        ilens = mix.shape[0]
        ref = torch.from_numpy(targets)
        if self.partition == "test":
            label = self.hdf_dataset[str(index)]["label"][0].astype(str)
            return [mix, ref, label]

        return [mix, ref]

    def __len__(self):
        return self.length


def binarizeSeparateSignal(separate_signal, celltypes):
    new_signals = np.zeros_like(separate_signal)
    for k, cell in enumerate(celltypes):
        tmp = separate_signal[:,k,:]
        tmp[tmp != 0] = 1
        new_signals[:,k,:] = tmp
    return new_signals


def normalizeSignal(separate_signal):
    normalizeSignal = np.zeros_like(separate_signal)
    for k in range(separate_signal.shape[1]):
        for i in range(separate_signal.shape[0]):
            max_s = separate_signal[i,k,:].max()
            min_s = separate_signal[i,k,:].min()
            if max_s != 0:
                normalizeSignal[i,k,:] = (separate_signal[i,k,:] - min_s)/(max_s-min_s)
    return normalizeSignal

def normalizeMixture(mixture):
    normalizeSignal = np.zeros_like(mixture.values)
    for i in range(mixture.shape[0]):
        max_s = mixture.iloc[i,:].max()
        min_s = mixture.iloc[i,:].min()
        if max_s != 0:
            normalizeSignal[i,:] = (mixture.iloc[i,:] - min_s)/(max_s-min_s)
    return normalizeSignal


def binarizeSignal(mixtures):
    new_signals = np.zeros_like(mixtures.values)
    new_signals[mixtures.values > 0] = 1
    return new_signals

def ratioSignal(separate_signal, mixtures, offset=1):
    ratioSignal = np.zeros_like(separate_signal)
    for k in range(separate_signal.shape[1]):
        for i in range(separate_signal.shape[0]):
            ratioSignal[i,k,:] = (separate_signal[i,k,:])/(mixtures[i,:] + offset)
    return ratioSignal

def normalizePeaks(mixture, mean_expression):
    return mixture - mean_expression

def normalizePeaks_signal(signals, mean_expression):
    return np.subtract(signals, mean_expression)

def normalizeMaxPeak(mixture, signals):
    max_val = np.max(mixture)
    mixture /= max_val
    signals /= max_val
    return mixture, signals

def logtransform(mixture, signals):
    mixture = np.log(mixture)
    signals = np.log(signals)
    return mixture, signals

def filter_data(mat, annot, key="peak_type", value="Intergenic", type="mixture"):
    index_genes= np.asarray(annot[~(annot[key] ==value)].index.tolist())
    if type == "mixture":
        mixture = mat.loc[:,np.asarray(index_genes).astype(str)]
        mixture["Sample_num"] = mat["Sample_num"]
        return mixture


    elif type=="separate":
        return mat[:,:,index_genes]
    else:
        raise "NotImplemented"


def prepareData(partition, 
        sample_id_test=SAMPLE_ID_TEST,
        sample_id_val=SAMPLE_ID_VAL,
        sample_id_train=None,
        holdout=True,
        cut_val=0.2,
        binarize=False,
        name="filter_promoter_ctrl",
        dataset_dir="/dataset/",
        hdf_dir="/dataset/hdf/",
        celltypes=['AST', 'Neur', 'OPC'],
        filter_intergenic=False,
        add_pure=False,
        binarize_input=False,
        normalize=False,
        normalizeMax=False,
        ratio_input=False,
        offset_input=1,
        only_training=False,
        cut=False,
        celltype_to_use=None,
        custom_testset=None,
        crop_func=None, annot=None,
        use_train=False,
        pure=False,
        limit=None,**kwargs):
    print("sample test :" +  str(sample_id_test))
    np.random.seed(seed=0)
    SP_test = sample_id_test
    if only_training:
        SP_test += "trainonly"
    if not os.path.isfile(os.path.join(hdf_dir, partition + ".hdf5")):
    #if True:
        mixture = pd.read_parquet(dataset_dir 
                                + name 
                                + MIXTUREFIX
                                )
        portion = pd.read_csv(dataset_dir
                                + name
                                + PORTIONFIX,
                            index_col=None)

        separate_signals = np.load(dataset_dir + name + SEPARATEFIX)["mat"]
        if "promoter" in name:
            mixture = filter_data(mixture, annot,
                            key="chrom",value="chrX",type="mixture")
            separate_signals = filter_data(separate_signals, annot,
                            key="chrom",value="chrX",type="separate")
            annot = annot[~(annot["chrom"] =="chrX")]
            mixture = filter_data(mixture, annot,
                            key="chrom",value="chrY",type="mixture")
            separate_signals = filter_data(separate_signals, annot,
                            key="chrom",value="chrY",type="separate")
            annot = annot[~(annot["chrom"] =="chrY")]
        if only_training:
            sample_id_train = mixture["Sample_num"].unique().tolist()
            sample_id_val = sample_id_train
            sample_id_test = sample_id_train

        else:

            sample_id = mixture["Sample_num"].unique().tolist()
            if not isinstance(sample_id_test, list):
                sample_id_test =  [sample_id_test]
            if sample_id_train is None:
                if sample_id_val is not None:
                    sample_id_train = [it for it in sample_id if it not in sample_id_test if not it in sample_id_val]
                else:
                    sample_id_train = [it for it in sample_id if it not in sample_id_test]
            print("sample test :" +  str(sample_id_test))
            print("sample train :" +  str(sample_id_train))
        sample_test = sample_id_test
        sample_val = sample_id_val
        sample_train = sample_id_train
        if add_pure:
            mixture_pure = pd.read_csv(dataset_dir 
                                        + name
                        + "__pure__synthsize_bulk_data_with_sparse.csv",
                        index_col=None)
            portion_pure = pd.read_csv(dataset_dir 
                                        + name
                    + "__pure__labels_synthsize_bulk_data_with_sparse.csv",
                    index_col=None)
            separate_signals_pure = np.load(dataset_dir 
                                        + name 
                        + "__pure__concatenat_separate_signal.npz")["mat"]
            mixture_train_tt = mixture[mixture["Sample_num"].isin(
                                        sample_train)]
            portion_train_tt = portion[mixture["Sample_num"].isin(
                                                sample_train)]
            separate_signal_train_tt = separate_signals[
                                            mixture["Sample_num"].isin(
                                                        sample_train),:,:]
        if holdout:
            if not (sample_val is None):
                sample_train = sample_train + sample_val

        mixture_train_tt = mixture[mixture["Sample_num"].isin(sample_train)]
        portion_train_tt = portion[mixture["Sample_num"].isin(sample_train)]
        separate_signal_train_tt = separate_signals[
                                            mixture["Sample_num"].isin(
                                                    sample_train),:,:]
        if holdout:
            n_val = int(cut_val*len(mixture_train_tt))
            n_train = len(mixture_train_tt)-n_val
            print("Hold out nval : %s"%str(n_val))
            print("Hold out ntrain : %s"%str(n_train))
            list_index = list(range(len(mixture_train_tt)))
            random.Random(4).shuffle(list_index)
            index_val = list_index[:n_val]
            index_train = list_index[n_val:]
            mixture_val_tt = mixture_train_tt.iloc[index_val,:] 
            portion_val_tt = portion_train_tt.iloc[index_val,:]
            separate_signal_val_tt = separate_signal_train_tt[
                                                        index_val, :,:]
            mixture_train_tt = mixture_train_tt.iloc[index_train,:] 
            portion_train_tt = portion_train_tt.iloc[index_train,:]
            separate_train_tt = separate_signal_train_tt[index_train, :,:]
        else:

            mixture_val_tt = mixture[mixture["Sample_num"].isin(sample_val)]
            portion_val_tt = portion[mixture["Sample_num"].isin(sample_val)]
            separate_signal_val_tt = separate_signals[
                                    mixture["Sample_num"].isin(
                                                        sample_val),:,:]

        if partition == "val":
            mixture_tt = mixture_val_tt
            separate_signal_tt = separate_signal_val_tt
            portion_tt = portion_val_tt
        elif partition == "test" and not use_train:
            mixture_tt = mixture[mixture["Sample_num"].isin(sample_test)]
            portion_tt = portion[mixture["Sample_num"].isin(sample_test)]
            separate_signal_tt = separate_signals[
                                    mixture["Sample_num"].isin(
                                                        sample_test),:,:]
            if limit is not None:
                idd = np.random.randint(0,len(mixture_test_tt), limit)
                mixture_test_tt = mixture_test_tt.iloc[idd]
                portion_test_tt = portion_test_tt.iloc[idd]
                separate_signal_test_tt = separate_signal_test_tt[idd]
        else:
            mixture_tt = mixture_train_tt
            portion_tt = portion[mixture["Sample_num"].isin(sample_test)]
            separate_signal_tt = separate_signal_train_tt
        len_train = len(mixture_tt)
        print("len %s: "%partition + str(len(mixture_tt)) )
        if filter_intergenic:
            mixture_tt = filter_data(mixture_tt,
                                    annot,
                                key="peak_type",value="Intergenic",
                                    type="mixture")
            separate_signal_tt = filter_data(
                                        separate_signal_tt,
                                        annot,
                                        key="peak_type",
                                        value="Intergenic",
                                        type="separate")

        _data = SeparationDataset(mixture_tt,
                                portion_tt,
                               separate_signal_tt,
                                celltypes,
                                hdf_dir, 
                                partition,
                               data_transform=crop_func,
                               binarize=binarize,
                               binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   offset=offset_input,
                                    celltype_to_use=celltype_to_use,
                                    cut=cut,
                                   ratio=ratio_input)
        return _data, annot#, test_data

    else:
        _data = SeparationDataset(None,
                                    None,
                                    None,
                                    celltypes,
                                    hdf_dir, 
                                    partition,
                                   data_transform=crop_func,
                                   binarize=binarize,
                                   binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   offset=offset_input,
                                    celltype_to_use=celltype_to_use,
                                            cut=cut,
                                           ratio=ratio_input)
        return _data, annot#, test_data


def make_dataloader(partition,
                    annot=None,
                    is_train=True,
                    data_kwargs=None,
                    num_workers=2,
                    ratio=False,
                    batch_size=16,
                    use_train=False,
                    limit=None,
                    pure=False,
                    only_training=False,
                    custom_testset=None,
                    **kwargs,
                    ):
        dataset, annot = prepareData(partition,
                **data_kwargs, 
                annot=annot)
        return DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=is_train,
                        num_workers=num_workers)
    #return CellsDataLoader(dataset,
    #                        is_train=is_train,
    #                      segment_len=chunk_size,
    #                  num_workers=num_workers)

def gatherCelltypesOLD(celltype_to_use, separate_signal, celltypes):
        if celltype_to_use is not None:
            if "Neurons" in celltype_to_use:
                get_index = [i for i,it in enumerate(celltypes) if it in ["EX","INH"]]
                tmp = separate_signal
                for i in range(1, len(get_index)):
                    tmp[:,get_index[0],:] += tmp[:,get_index[i],:] 
                new_ind = [i for i,it in enumerate(celltypes) if it not in ["INH"]]
                celltypes[get_index[0]] = "Neurons"
                celltypes.remove("INH")
                #print(celltypes)
                separate_signal = tmp[:,new_ind,:]
            if "OPCs-Oligo" in celltype_to_use:
                get_index = [i for i,it in enumerate(celltypes) if it in ["OPCs","OLD"]]
                tmp = separate_signal
                for i in range(1, len(get_index)):
                    tmp[:,get_index[0],:] += tmp[:,get_index[i],:] 
                new_ind = [i for i,it in enumerate(celltypes) if it not in ["OLD"]]
                celltypes[get_index[0]] = "OPCs-Oligo"
                celltypes.remove("OLD")
                #print(celltypes)
                separate_signal = tmp[:,new_ind,:]
            if "AST-OPCs-OLD" in celltype_to_use:
                get_index = [i for i,it in enumerate(celltypes) if it in ["AST","OPCs","OLD"]]
                tmp = separate_signal
                for i in range(1, len(get_index)):
                    tmp[:,get_index[0],:] += tmp[:,get_index[i],:] 
                new_ind = [i for i,it in enumerate(celltypes) if it not in ["OLD"]]
                celltypes[get_index[0]] = "AST-OPCs-OLD"
                celltypes.remove("OLD")
                celltypes.remove("OPCs")
                #print(celltypes)
                separate_signal = tmp[:,new_ind,:]
            if "Glia" in celltype_to_use:
                get_index = [i for i,it in enumerate(celltypes) if it in [ "AST","MIC", "OPCs","OLD"]]
                tmp = separate_signal
                for i in range(1, len(get_index)):
                    tmp[:,get_index[0],:] += tmp[:,get_index[i],:] 
                new_ind = [i for i,it in enumerate(celltypes) if it not in ["MIC", "OPCs","OLD"]]
                celltypes[get_index[0]] = "Glia"
                celltypes.remove("OLD")
                celltypes.remove("MIC")
                celltypes.remove("OPCs")
                #print(celltypes)
                separate_signal = tmp[:,new_ind,:]
                #print(new_ind)
                #print(separate_signal.shape)


            celltypes_to_use = [i for i,it in enumerate(celltypes) if it in celltype_to_use]
            print("celltypes used ")
            print(celltypes_to_use)
        return celltypes_to_use, separate_signal, celltypes

def gatherCelltypes(celltype_to_use, separate_signal, celltypes):
    #print(separate_signal.max())
    if celltype_to_use is not None:
        new_separate = np.zeros((separate_signal.shape[0],
                            len(celltype_to_use),
                             separate_signal.shape[2]))
        for ind,ct in enumerate(celltype_to_use):
            #print(ct)
            if ct =="Neurons" :
                get_index = [i for i,it in enumerate(celltypes) if it in ["EX","INH"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:] 
                #print(celltypes)
                new_separate[:,ind,:] = tmp.copy()
                #print(new_separate.max())
            elif ct=="OPCs-Oligo" :
                get_index = [i for i,it in enumerate(celltypes) if it in ["OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:] 
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="MIC-OPCs-OLD":
                get_index = [i for i,it in enumerate(celltypes) if it in ["MIC", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:] 
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="AST-OPCs-OLD":
                get_index = [i for i,it in enumerate(celltypes) if it in ["AST", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:] 
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="Glia":
                get_index = [i for i,it in enumerate(celltypes) if it in [ "AST","MIC", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:] 
                ##print(celltypes)
                new_separate[:,ind,:] = tmp.copy()
                #print(new_separate.max())
            else:
                get_index = [i for i,it in enumerate(celltypes) if it in [ct]]
                new_separate[:,ind,:] = separate_signal[:,get_index[0],:]
    else:
            new_separate = separate_signal
    return celltype_to_use, new_separate

def OLDprepareData(partition, 
        sample_id_test=SAMPLE_ID_TEST,
        sample_id_val=SAMPLE_ID_VAL,
        holdout=True,
        cut_val=0.2,
        binarize=False,
        name="filter_promoter_ctrl",
        dataset_dir="/dataset/",
        hdf_dir="/dataset/hdf/",
        celltypes=['AST', 'Neur', 'OPC'],
        filter_intergenic=False,
        add_pure=False,
        binarize_input=False,
        normalize=False,
        normalizeMax=False,
        ratio_input=False,
        offset_input=1,
        only_training=False,
        cut=False,
        celltype_to_use=None,
        custom_testset=None,
        crop_func=None, annot=None,
        use_train=False,
        pure=False,
        limit=None,**kwargs):
    
    print("sample test :" +  str(sample_id_test))
    #print("sample val :" +  str(sample_id_val))
    np.random.seed(seed=0)
    SP_test = sample_id_test
    if only_training:
        SP_test += "trainonly"
    if not os.path.isfile(dataset_dir + name + str(SP_test) + "n_mixture_test.h5"):
    #if True:
        mixture = pd.read_csv(dataset_dir + name + MIXTUREFIX, index_col=None)
                #"_only_sparse__synthsize_bulk_data_with_sparse.csv", index_col=None)
        portion = pd.read_csv(dataset_dir + name + PORTIONFIX, index_col=None)
        # "_only_sparse__labels_synthsize_bulk_data_with_sparse.csv", index_col=None)

        separate_signals = np.load(dataset_dir + name + SEPARATEFIX)["mat"]
        #"_only_sparse__concatenat_separate_signal.npz")["mat"]
        if "promoter" in name:
            mixture = filter_data(mixture, annot,
                            key="chrom",value="chrX",type="mixture")
            separate_signals = filter_data(separate_signals, annot,
                            key="chrom",value="chrX",type="separate")
            annot = annot[~(annot["chrom"] =="chrX")]
            mixture = filter_data(mixture, annot,
                            key="chrom",value="chrY",type="mixture")
            separate_signals = filter_data(separate_signals, annot,
                            key="chrom",value="chrY",type="separate")
            annot = annot[~(annot["chrom"] =="chrY")]
        
        if only_training:
            sample_id_train = mixture["Sample_num"].unique()
            sample_id_val = sample_id_train
            sample_id_test = sample_id_train

        else:

            sample_id = mixture["Sample_num"].unique().tolist()
            sample_id_test = [it for it in sample_id if (str(sample_id_test) in it)]
            sample_id_val = [it for it in sample_id if (str(sample_id_val) in it)]
            #assert sample_id_test in sample_id
            sample_id_train = [it for it in sample_id if it not in sample_id_test if not it in sample_id_val]
        sample_test = sample_id_test
        sample_val = sample_id_val
        sample_train = sample_id_train
        if add_pure:
            mixture_pure = pd.read_csv(dataset_dir + name + "__pure__synthsize_bulk_data_with_sparse.csv", index_col=None)
            portion_pure = pd.read_csv(dataset_dir + name + "__pure__labels_synthsize_bulk_data_with_sparse.csv", index_col=None)
            separate_signals_pure = np.load(dataset_dir + name + "__pure__concatenat_separate_signal.npz")["mat"]
            mixture_train_tt = mixture[mixture["Sample_num"].isin(sample_train)]
            portion_train_tt = portion[mixture["Sample_num"].isin(sample_train)]
            separate_signal_train_tt = separate_signals[
                                                    mixture["Sample_num"].isin(
                                                                sample_train),:,:]


        if holdout:
            sample_train = sample_train + sample_val

        mixture_train_tt = mixture[mixture["Sample_num"].isin(sample_train)]
        portion_train_tt = portion[mixture["Sample_num"].isin(sample_train)]
        separate_signal_train_tt = separate_signals[
                                                mixture["Sample_num"].isin(
                                                            sample_train),:,:]
        if holdout:
            n_val = int(cut_val*len(mixture_train_tt))
            n_train = len(mixture_train_tt)-n_val
            print("Hold out nval : %s"%str(n_val))
            print("Hold out ntrain : %s"%str(n_train))
            list_index = list(range(len(mixture_train_tt)))
            random.shuffle(list_index)
            index_val = list_index[:n_val]
            index_train = list_index[n_val:]
            mixture_val_tt = mixture_train_tt.iloc[index_val,:] 
            portion_val_tt = portion_train_tt.iloc[index_val,:]
            separate_signal_val_tt = separate_signal_train_tt[index_val, :,:]
            mixture_train_tt = mixture_train_tt.iloc[index_train,:] 
            portion_train_tt = portion_train_tt.iloc[index_train,:]
            separate_train_tt = separate_signal_train_tt[index_train, :,:]
        else:

            mixture_val_tt = mixture[mixture["Sample_num"].isin(sample_val)]
            portion_val_tt = portion[mixture["Sample_num"].isin(sample_val)]
            separate_signal_val_tt = separate_signals[mixture["Sample_num"].isin(sample_val),:,:]

        mixture_train_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_mixture_train.h5", 'df')
        portion_train_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_proportion_train.h5", 'df')
        np.savez_compressed(dataset_dir + name + str(SP_test)+ "n_separate_train.npz", mat=separate_signal_train_tt)

        mixture_val_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_mixture_val.h5", 'df')
        portion_val_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_proportion_val.h5", 'df')
        np.savez_compressed(dataset_dir + name + str(SP_test)+ "n_separate_val.npz", mat=separate_signal_val_tt)
        mixture_test_tt = mixture[mixture["Sample_num"].isin(sample_test)]
        portion_test_tt = portion[mixture["Sample_num"].isin(sample_test)]
        separate_signal_test_tt = separate_signals[mixture["Sample_num"].isin(sample_test),:,:]
        mixture_test_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_mixture_test.h5", 'df')
        portion_test_tt.to_hdf(dataset_dir + name + str(SP_test)+ "n_proportion_test.h5", 'df')
        np.savez_compressed(dataset_dir + name + str(SP_test)+ "n_separate_test.npz", mat=separate_signal_test_tt)
        if partition == "val":
            mixture_tt = mixture_val_tt
            separate_signal_tt = separate_signal_val_tt
            portion_tt = portion_val_tt
        else:
            mixture_tt = mixture_train_tt
            separate_signal_tt = separate_signal_train_tt
            portion_tt = portion_train_tt
    else:

        if partition=="test":
            
            if pure:
                mixture_test_tt = pd.read_csv(args.dataset_dir + name + "__pure_pure__synthsize_bulk_data_with_sparse.csv", index_col=None)
                portion_test_tt = pd.read_csv(args.dataset_dir + name + "__pure_pure__labels_synthsize_bulk_data_with_sparse.csv", index_col=None)
                separate_signal_test_tt = np.load(args.dataset_dir + name + "__pure_pure__concatenat_separate_signal.npz")["mat"]
            elif custom_testset is not None:
                mixture_test_tt = pd.read_csv(dataset_dir + custom_testset + "_synthsize_bulk_data_with_sparse.csv", index_col=None)
                portion_test_tt = pd.read_csv(dataset_dir + custom_testset + "_labels_synthsize_bulk_data_with_sparse.csv", index_col=None)
                separate_signal_test_tt = np.load(dataset_dir + custom_testset + "_concatenat_separate_signal.npz")["mat"]
                if use_train:
                    sample_id = mixture_test_tt["Sample_num"].unique()
                    sample_id_test = [it for it in sample_id if (str(sample_id_test) in it)]
                    sample_id_val = [it for it in sample_id if (str(sample_id_val) in it)]
                    sample_test = [it for it in sample_id if it!=sample_id_test if it !=sample_id_val]
                else:
                    #sample_test = [sample_id_test]
                    sample_test = [it for it in sample_id if (str(sample_id_test) in it)]
                portion_test_tt = portion_test_tt[mixture_test_tt["Sample_num"].isin(sample_test)]
                separate_signal_test_tt = separate_signal_test_tt[
                                        mixture_test_tt["Sample_num"].isin(
                                                sample_test),:,:]
                mixture_test_tt = mixture_test_tt[
                                mixture_test_tt["Sample_num"].isin(sample_test)]
            elif use_train and (custom_testset is None):
                mixture_test_tt = pd.read_hdf(dataset_dir + 
                                            name + str(SP_test) 
                                            +"n_mixture_%s.h5"%"train",
                                            'df')
                df_s = mixture_test_tt.reset_index().groupby("Sample_num").sample(
                                                            200,
                                                            replace=False,
                                                            random_state=1)
                portion_test_tt= pd.read_hdf(dataset_dir +
                        name + str(SP_test)+ "n_proportion_%s.h5"%"train", 'df')
                separate_signal_test_tt = np.load(dataset_dir + 
                        name + str(SP_test)+ "n_separate_%s.npz"%"train")["mat"]
                mixture_test_tt = mixture_test_tt.iloc[df_s.index.tolist(),:]
                portion_test_tt = portion_test_tt.iloc[df_s.index.tolist(),:]
                separate_signal_test_tt = separate_signal_test_tt[df_s.index.tolist(),:]
            else:
                mixture_test_tt = pd.read_hdf(dataset_dir + 
                            name + str(SP_test)+ "n_mixture_%s.h5"%partition, 'df')
                portion_test_tt= pd.read_hdf(dataset_dir +
                        name + str(SP_test)+ "n_proportion_%s.h5"%partition, 'df')
                separate_signal_test_tt = np.load(dataset_dir + 
                        name + str(SP_test)+ "n_separate_%s.npz"%partition)["mat"]
            portion_test_tt.rename(columns={"PER.END": "PEREND"}, inplace=True)
            if "promoter" in name:
                annot = annot[~(annot["chrom"] =="chrX")]
                annot = annot[~(annot["chrom"] =="chrY")]
            if "Unnamed: 0" in mixture_test_tt.columns.tolist():
                mixture_test_tt.drop("Unnamed: 0",axis=1, inplace=True)
            if filter_intergenic:
                mixture_test_tt = filter_data(mixture_test_tt,
                                                    annot,
                        key="peak_type",value="Intergenic",
                                                    type="mixture")
                separate_signal_test_tt = filter_data(
                                                    separate_signal_test_tt,
                                                    annot,
                                        key="peak_type",value="Intergenic",
                                                    type="separate")
            
            if limit is not None:
                idd = np.random.randint(0,len(mixture_test_tt), limit)
                mixture_test_tt = mixture_test_tt.iloc[idd]
                portion_test_tt = portion_test_tt.iloc[idd]
                separate_signal_test_tt = separate_signal_test_tt[idd]

            test_data = SeparationDataset(mixture_test_tt, 
                     portion_test_tt, 
                            separate_signal_test_tt,
                            celltypes,
                            hdf_dir, 
                                partition,
                                   data_transform=crop_func,
                                   binarize=binarize,
                                   binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   pure=pure,
                                    celltype_to_use=celltype_to_use,
                                   offset=offset_input,
                                   ratio=ratio_input)
            
            return test_data, mixture_test_tt, separate_signal_test_tt, portion_test_tt, annot
        else:
            mixture_tt = pd.read_hdf(dataset_dir + name + str(SP_test)+ "n_mixture_%s.h5"%partition, 'df')
            portion_tt= pd.read_hdf(dataset_dir + name + str(SP_test)+ "n_proportion_%s.h5"%partition, 'df')
            separate_signal_tt = np.load(dataset_dir + name + str(SP_test)+ "n_separate_%s.npz"%partition)["mat"]


    len_train = len(mixture_tt)
    print("len %s: "%partition + str(len(mixture_tt)) )
    portion_tt.rename(columns={"PER.END": "PEREND"}, inplace=True)
    
    if "Unnamed: 0" in mixture_tt.columns.tolist():
        mixture_tt.drop("Unnamed: 0",axis=1, inplace=True)
    if filter_intergenic:
        mixture_tt = filter_data(mixture_tt,
                                            annot,
                                        key="peak_type",value="Intergenic",
                                            type="mixture")
        separate_signal_tt = filter_data(
                                            separate_signal_tt,
                                            annot,
                                        key="peak_type",value="Intergenic",
                                            type="separate")

    _data = SeparationDataset(mixture_tt,
                                portion_tt,
                                   separate_signal_tt,
                            celltypes,
                            hdf_dir, 
                                partition,
                                   data_transform=crop_func,
                                   binarize=binarize,
                                   binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   offset=offset_input,
                                    celltype_to_use=celltype_to_use,
                                    cut=cut,
                                   ratio=ratio_input)
    return _data, annot#, test_data
