library(ArchR)
#getGeneAnnotation("hg38"#)
addArchRGenome("hg38")                 # 

pathDATA = "./scDATA/"
pathSC = "./data/"
metadata = read.csv("./data/annot_all_scATAC_detailed.csv")
Allfiles <- Sys.glob(file.path(pathDATA,"ArrowFiles/*.arrow"))
proj <- ArchRProject( ArrowFiles = Allfiles, 
                      outputDirectory = "scATAC-seq_detailed_6_ct",
                      copyArrows = TRUE 
                              )
saveArchRProject(proj, "all_scatac_detailed_6_ct")
#149011 cells
proj = loadArchRProject("all_scatac_detailed_6_ct")

idxSample <- BiocGenerics::which(proj$cellNames %in% metadata$cellNames)
celname = proj$cellNames[idxSample]    # 
proj = proj[celname,]
proj #125183


row.names(metadata) = metadata$cellNames
##125343
metadata = metadata[proj$cellNames,]
##125184
proj = addCellColData(proj, data=metadata$celltype2, cell=proj$cellNames, name="celltype")
proj = addCellColData(proj, data=metadata$Donor_ID, cell=proj$cellNames, name="DonorID")
proj = addCellColData(proj, data=metadata$Region, cell=proj$cellNames, name="Region") # 
proj = addCellColData(proj, data=metadata$barcode, cell=proj$cellNames, name="Barcode")

#remove Doublets
idxSample <- BiocGenerics::which(! proj$celltype %in% "Doublets")
nocel = proj$cellNames[idxSample]
proj = proj[nocel,]#124767
proj

##remove UnknownNeurons and Nigral Neurons
idxSample <- BiocGenerics::which(! proj$celltype %in% "UnknownNeurons")
nocel = proj$cellNames[idxSample]
proj = proj[nocel,]#12346
idxSample <- BiocGenerics::which(! proj$celltype %in% "NigralNeurons")
nocel = proj$cellNames[idxSample]
proj = proj[nocel,]#122524
proj

# group cell per replicates
proj@cellColData$SampleGroupby = paste0(proj$DonorID, proj$Region, proj$celltype)
saveArchRProject(proj, "6_ct_no_doublets_no_nn_un")
proj = loadArchRProject("6_ct_no_doublets_no_nn_un")



addArchRThreads(threads = 1)
proj$Clusters = proj$SampleGroupby
proj <- addGroupCoverages(ArchRProj = proj, groupBy = "Clusters", force=TRUE)
saveArchRProject(proj, "6_ct_no_doublets_no_nn_un")
proj = loadArchRProject("6_ct_no_doublets_no_nn_un")
pathToMacs2 <- findMacs2()
proj <- addReproduciblePeakSet(
    ArchRProj = proj, groupby="Clusters", 
    pathToMacs2 = pathToMacs2,
    force=TRUE)
proj <- addPeakMatrix(proj, force=TRUE)
addArchRThreads(threads = 30)
proj <- addMotifAnnotations(ArchRProj = proj, motifSet = "cisbp", name = "Motif")
proj <- addMotifAnnotations(ArchRProj = proj, motifSet = "JASPAR2020", name = "MotifJaspar")
saveArchRProject(proj, "6_ct_no_doublets_no_nn_un")
proj = loadArchRProject("6_ct_no_doublets_no_nn_un")
markersPeaks <- getMarkerFeatures(
    ArchRProj = proj, 
    useMatrix = "PeakMatrix", 
    groupBy = "celltype",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon",
  threads=20
)
markerList <- getMarkers(markersPeaks, cutOff = "FDR <= 0.05 & Log2FC >= 0.5", returnGR=TRUE)

savepath = pathSC #paste0(pathSC,"region_donor_cluster/")

heatmapPeaks <- plotMarkerHeatmap(
  seMarker = markersPeaks, 
  cutOff = "FDR <= 0.05 & Log2FC >= 1",
  transpose = TRUE, #labelMarkers=markerGenes,
)
plotPDF(heatmapPeaks, name = "peak-label-Marker-Heatmap_log2_1_fdr_0.05_region_donor_cluster", width = 8, height = 6, ArchRProj = proj, addDOC = FALSE)
draw(heatmapPeaks, heatmap_legend_side = "bot", annotation_legend_side = "bot")
write.csv(proj@geneAnnotation$genes, paste0(savepath,"gene_annot_genes.csv"))
write.csv(proj@geneAnnotation$TSS, paste0(savepath,"gene_annot_TSS.csv"))
write.csv(proj@geneAnnotation$exons, paste0(savepath,"gene_annot_exons.csv"))
motifpos = readRDS(proj@peakAnnotation$Motif$Positions)
write.csv(proj@peakAnnotation$Motif$MotifSummary, paste0(savepath,"peak_annot_cisbp_summary.csv"))
write.csv(motifpos, paste0(savepath,"peak_annot_cisbp.csv"))
write.csv(proj@peakAnnotation$MotifJaspar$MotifSummary, paste0(savepath,"peak_annot_motifJaspar_Summary.csv"))
motifposJas = readRDS(proj@peakAnnotation$MotifJaspar$Positions)
write.csv(motifposJas, paste0(savepath,"peak_annot_jaspar.csv"))
write.csv(assays(markersPeaks)$AUC, paste0(savepath,"markerspeak_AUC.csv"))
write.csv(assays(markersPeaks)$Log2FC, paste0(savepath,"markerspeak_log2fc.csv"))
write.csv(assays(markersPeaks)$FDR, paste0(savepath,"markerspeak_FDR.csv"))
write.csv(assays(markersPeaks)$MeanBGD, paste0(savepath,"markerspeak_meanBG.csv"))
write.csv(assays(markersPeaks)$Mean, paste0(savepath,"markerspeak_mean.csv"))
write.csv(assays(markersPeaks)$MeanDiff, paste0(savepath,"markerspeak_MeanDiff.csv"))
write.csv(assays(markersPeaks)$Pval, paste0(savepath,"markerspeak_Pval.csv"))
assays(markersPeaks)$Pval
write.csv(rowData(markersPeaks), paste0(savepath,"markerspeak_rowdata.csv"))
rowData(markersPeaks)
write.csv(colData(markersPeaks), paste0(savepath,"markerspeak_coldata.csv"))
pkmtx = getMatrixFromProject(proj, "PeakMatrix")
assay(pkmtx)
pkmtx
library(Matrix)
writeMM(assay(pkmtx), paste0(savepath,"my_mat.mtx"))
colData(pkmtx)
write.csv(colData(pkmtx), paste0(savepath,"coldata.csv"))
write.csv(rowData(pkmtx), paste0(savepath,"rowdata.csv"))
write.csv(proj@peakSet, paste0(savepath,"peakset.csv"))

