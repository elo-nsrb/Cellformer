library(ChIPseeker)
library(ChIPpeakAnno)
library(GenomicRanges)
library(genomation)                    # 
library(data.table)
require(TxDb.Hsapiens.UCSC.hg38.knownGene)
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
mypath = "./data/"
peakfile = paste0(mypath,"peaks_with_header.tsv")
peakAnno <- annotatePeak(peakfile, tssRegion=c(-1000, 1000), 
                         TxDb=txdb, annoDb="org.Hs.eg.db")

x = as.data.frame(peakAnno@anno)
write.table( x=x, 
                file = paste0(mypath, "annot_chipseeker.tsv"), 
                sep="\t", col.names=TRUE,
                na="",
                row.names=FALSE, quote=FALSE )

###
#data(TSS.human.GRCh38)
#peakfile = paste0(mypath,"peaks.tsv")
#gr1 <- toGRanges(peakfile, format="BED", header=FALSE)
#gr1.anno <- annotatePeakInBatch(gr1, AnnotationData=TSS.human.GRCh38)
#
#library(org.Hs.eg.db)
#gr1.anno <- addGeneIDs(annotatedPeak=gr1.anno, 
#                        orgAnn="org.Hs.eg.db", 
#                        IDs2Add="symbol")
#library(GenomicFeatures)
#library("biomaRt")
#library(TxDb.Hsapiens.UCSC.hg38.knownGene)
#ucsc.hg38.knownGene <- genes(TxDb.Hsapiens.UCSC.hg38.knownGene)
#gr1.anno2 <- annotatePeakInBatch(gr1, 
#                              AnnotationData=ucsc.hg38.knownGene)
#gr1.anno2 <- addGeneIDs(annotatedPeak=gr1.anno2, 
#                         orgAnn="org.Hs.eg.db", 
#                         feature_id_type="entrez_id",
#                         IDs2Add="symbol")
#ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl")
#
#aCR<-assignChromosomeRegion(gr1, nucleotideLevel=FALSE, 
#                            precedence=c("Promoters", 
#                                         "immediateDownstream", 
#                                         "fiveUTRs", "threeUTRs", 
#                                         "Exons", "Introns"), 
#                                TxDb=TxDb.Hsapiens.UCSC.hg38.knownGene)
#png(paste0(mypath, 'upsetplot_peakannot_chipseeker.jpg'))
#upsetplot(peakAnno, vennpie=TRUE)
#dev.off()
