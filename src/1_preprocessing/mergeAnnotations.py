import os 
import pandas as pd
import numpy as np
import argparse
import episcanpy as es
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--path', default="./data/",
                    help='Location to save pseudobulks data')

parser.add_argument('--name', default="pseudobulks",
                    help='Name pseudobulks data')


def parseHomerAnnotation(path):
    ann_h = pd.read_csv(path, sep="\t")
    ann_h["chrm"] = ann_h["Chr"].str.split("chr", expand=True)[1].values
    ann_h.sort_values(["chrm", "Start"], inplace=True)
    ann_h["peakType"] = ann_h["Annotation"].str.split("(", expand=True)[0].str.split(" ", expand=True)[0]
    ann_h["peakID"] = ann_h["Chr"] + "_" + (ann_h["Start"]).astype(str) + "_" +ann_h["End"].astype(str)
    return ann_h

def parseAndMergePavisannot(annot_pavis, annot):
    annot_pavis["peakID"] = (annot_pavis["Chromosome"].astype(str).values
                            + "_" 
                            + annot_pavis["Loci Start"].astype(str).values
                            + "_" 
                            + annot_pavis["Loci End"].astype(str).values )
    annot_pavis = annot_pavis[annot_pavis.peakID.isin(annot.peakID.values.tolist())]
    annot.loc[annot.peakID.isin(annot_pavis.peakID.values.tolist()),
                "peakPavis"] = annot_pavis.Category.values

    annot.loc[annot.peakID.isin(annot_pavis.peakID.values.tolist()),"nearestGenePavis"] = annot_pavis["Gene Symbol"].values
    annot.loc[annot.peakID.isin(annot_pavis.peakID.values.tolist()),"distanceToTSSPavis"] = annot_pavis["Distance to TSS"].values
    return annot

def mergeAnnots(annot, annot_h):
    annot["peakTypeHomer"] = annot_h[annot_h.peakID.isin(annot.peakID.values.tolist())].peakType.values
    annot["GeneType"] = annot_h[annot_h.peakID.isin(annot.peakID.values.tolist())]["Gene Type"].values
    return annot

def parseAndMergeChipannot(annot_chip, annot):
    annot_chip["peakID"] = (annot_chip["seqnames"].astype(str).values
                            + "_" 
                            + (annot_chip["start"] -1).astype(str).values
                            + "_" 
                            + annot_chip["end"].astype(str).values )
    annot_chip = annot_chip[annot_chip.peakID.isin(annot.peakID.values.tolist())]
    annot.loc[annot.peakID.isin(annot_chip.peakID.values.tolist()),"nearestGeneChip"] = annot_chip["SYMBOL"].values
    annot.loc[annot.peakID.isin(annot_chip.peakID.values.tolist()),"peakAnnotationChip"] = annot_chip["annotation"].values
    annot["shortAnnotChip"] = annot_chip["annotation"].values
    annot.loc[annot["shortAnnotChip"].str.contains("Exon"),"shortAnnotChip"] = "Exon"
    annot.loc[annot["shortAnnotChip"].str.contains("Intron"),"shortAnnotChip"] = "Intron"
    

    annot.loc[annot.peakID.isin(annot_chip.peakID.values.tolist()),
                    "distanceToTSSChip"] = annot_chip["distanceToTSS"].values
    return annot


def annotatePeak(adata):
    #!wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz 
    #!gunzip gencode.v38.annotation.gtf
    
    subprocess.run(["wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz"]) 
    subprocess.run(["gunzip gencode.v38.annotation.gtf"]) 
    
    adata.var_names = (adata.var.chrm.astype(str).values 
                       + "_" + adata.var.start.astype(str).values 
                       + "_" + adata.var.end.astype(str).values)

    epi.tl.find_genes(adata,
           gtf_file='gencode.v38.annotation.gtf',
           key_added='gene',
           upstream=2000,
           feature_type='gene',
           annotation='HAVANA',
           raw=False)

def main():
    args = parser.parse_args()
    path = args.path
    annot_pa = os.path.join(path, args.name + "_annotations.csv")
    annot = pd.read_csv(annot_pa)
    annot_chipseeker = pd.read_csv(os.path.join(path, "annot_chipseeker.tsv"),header=0, sep="\t")
    annot = parseAndMergeChipannot(annot_chipseeker, annot)
    #annot_homer = os.path.join(path, "annot_homer.tsv")
    #annot_h = parseHomerAnnotation(annot_homer)
    #annot = mergeAnnots(annot, annot_h)
    #annot_pavis = pd.read_csv(os.path.join(path, "pavis_annotation.txt"), sep="\t")
    #annot = parseAndMergePavisannot(annot_pavis, annot)
    
    annot.to_csv(os.path.join(path, "mergeAnnot.csv"))


if __name__ == "__main__":
    main()
