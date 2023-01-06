library(ChIPseeker)
library(ChIPpeakAnno)
library(GenomicRanges)
library(genomation)                    # 
library(data.table)
suppressPackageStartupMessages(library("argparse"))
require(TxDb.Hsapiens.UCSC.hg38.knownGene)
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
parser <- ArgumentParser()
parser$add_argument( "--path_data",
                        default="./data/",
                           help="Directory with peaks_with_header.tsv")
args <- parser$parse_args()
mypath = args$path_data
peakfile = paste0(mypath,"peaks_with_header.tsv")
peakAnno <- annotatePeak(peakfile, tssRegion=c(-1000, 1000), 
                         TxDb=txdb, annoDb="org.Hs.eg.db")

x = as.data.frame(peakAnno@anno)
write.table( x=x, 
                file = paste0(mypath, "annot_chipseeker.tsv"), 
                sep="\t", col.names=TRUE,
                na="",
                row.names=FALSE, quote=FALSE )

