#!/usr/bin/env Rscript
# Reproduces the 10x Genomics workflow:
# https://www.10xgenomics.com/analysis-guides/xenium-cell-type-annotation
#
# Dependencies (see guide): Seurat (v5+), SeuratObject, BPCells (optional but
# recommended), jsonlite, tidyverse. Plotting packages (ggplot2, ggpmisc, etc.)
# are only needed if you run plotting sections interactively.
#
# On HPC systems, load R with the site module first (names vary; use `module spider R`):
#   module load R
#   # or: module load gcc/12 R/4.4.0
#
# Example:
#   Rscript scripts/xenium_cell_type_annotation.R \
#     --xenium-dir /path/to/Xenium_outs \
#     --flex-h5 /path/to/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5 \
#     --flex-annotations /path/to/FLEX_Ovarian_Barcode_Cluster_Annotation.csv \
#     --output /path/to/xenium_cell_labels.csv

options(future.globals.maxSize = 1e9)

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(jsonlite)
  library(tidyverse)
})

args <- commandArgs(trailingOnly = TRUE)
parse_args <- function() {
  kv <- list()
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (startsWith(a, "--")) {
      key <- sub("^--", "", a)
      if (i + 1L <= length(args) && !startsWith(args[[i + 1L]], "--")) {
        kv[[key]] <- args[[i + 1L]]
        i <- i + 2L
      } else {
        kv[[key]] <- "TRUE"
        i <- i + 1L
      }
    } else {
      i <- i + 1L
    }
  }
  kv
}
cli <- parse_args()

get_arg <- function(name, default = NULL) {
  v <- cli[[name]]
  if (is.null(v)) default else v
}

xenium_dir <- get_arg("xenium-dir", Sys.getenv("XENIUM_DIR", ""))
flex_h5 <- get_arg("flex-h5", Sys.getenv("FLEX_H5", ""))
flex_annotations <- get_arg("flex-annotations", Sys.getenv("FLEX_ANNOTATIONS_CSV", ""))
output_csv <- get_arg("output", file.path(getwd(), "xenium_cell_labels.csv"))
sketch_ncells <- as.integer(get_arg("sketch-ncells", "100000"))
use_bpcells <- tolower(get_arg("use-bpcells", "true")) %in% c("1", "true", "yes")

if (!nzchar(xenium_dir) || !nzchar(flex_h5) || !nzchar(flex_annotations)) {
  stop(
    "Provide --xenium-dir, --flex-h5, and --flex-annotations (or set ",
    "XENIUM_DIR, FLEX_H5, FLEX_ANNOTATIONS_CSV).",
    call. = FALSE
  )
}

xenium_dir <- normalizePath(xenium_dir, mustWork = TRUE)
flex_h5 <- normalizePath(flex_h5, mustWork = TRUE)
flex_annotations <- normalizePath(flex_annotations, mustWork = TRUE)
output_csv <- normalizePath(output_csv, mustWork = FALSE)
out_dir <- dirname(output_csv)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

has_bpcells <- requireNamespace("BPCells", quietly = TRUE)
if (use_bpcells && !has_bpcells) {
  message("BPCells not installed; continuing without on-disk matrices.")
  use_bpcells <- FALSE
}

custom_hex <- c(
  "#3375AB", "#FFFF00", "#E8280E", "#8D1909", "#E47E11", "#FF75A0", "#19297C",
  "#AC674E", "#805D93", "#13B0F7", "#8980F5", "#595959", "#0BFFFF", "#169873",
  "#8EE525", "#169873", "#52E097", "#8980F5", "#EE00B0"
)
names(custom_hex) <- c(
  "Tumor Associated Fibroblasts", "Malignant Cells Lining Cyst", "Tumor Cells",
  "Inflammatory Tumor Cells", "Proliferative Tumor Cells", "VEGFA+ Tumor Cells",
  "Stromal Associated Fibroblasts", "Endothelial Cells", "Stromal Associated Macrophages",
  "Smooth Muscle Cells", "Tumor Associated Macrophages", "T & NK Cells", "Pericytes",
  "Granulosa and FT Epithelial Cells", "Ciliated Epithelial Cells", "Granulosa Cells",
  "Fallopian Tube Epithelium", "Macrophages", "MT-High, Jun+/Fos+ Tumor Cells"
)

message("Loading Flex reference...")
flex_data.obj <- Read10X_h5(flex_h5)
flex_data.obj <- CreateSeuratObject(counts = flex_data.obj)

if (use_bpcells) {
  flex_counts_dir <- file.path(out_dir, "flex_counts_bpcells")
  dir.create(flex_counts_dir, showWarnings = FALSE, recursive = TRUE)
  BPCells::write_matrix_dir(mat = flex_data.obj[["RNA"]]$counts, dir = flex_counts_dir)
  counts.mat <- BPCells::open_matrix_dir(dir = flex_counts_dir)
  flex_data.obj[["RNA"]]$counts <- counts.mat
  rm(counts.mat)
}

flex_data.obj[["percent.mt"]] <- PercentageFeatureSet(flex_data.obj, pattern = "^MT-")
flex_data.obj <- subset(
  flex_data.obj,
  subset = nCount_RNA > 200 & nCount_RNA < 10000 & percent.mt < 10
)

flex_annotation_file <- read.csv(flex_annotations, stringsAsFactors = FALSE)
flex_annotations_vec <- flex_annotation_file$Cell.Annotation
names(flex_annotations_vec) <- flex_annotation_file$Barcode
flex_data.obj <- AddMetaData(
  object = flex_data.obj,
  metadata = flex_annotations_vec,
  col.name = "cell_type"
)

message("Loading Xenium...")
xenium.obj <- LoadXenium(xenium_dir, fov = "fov", molecule.coordinates = FALSE)
DefaultAssay(xenium.obj) <- "Xenium"

if (use_bpcells) {
  xen_counts_dir <- file.path(out_dir, "xenium_counts_bpcells")
  dir.create(xen_counts_dir, showWarnings = FALSE, recursive = TRUE)
  BPCells::write_matrix_dir(mat = xenium.obj[["Xenium"]]$counts, dir = xen_counts_dir)
  counts.mat <- BPCells::open_matrix_dir(dir = xen_counts_dir)
  xenium.obj[["Xenium"]]$counts <- counts.mat
  rm(counts.mat)
}

xenium.obj@meta.data$nCount_Xenium_log <- log1p(xenium.obj@meta.data$nCount_Xenium)
xenium.obj@meta.data$nFeature_Xenium_log <- log1p(xenium.obj@meta.data$nFeature_Xenium)
xenium.obj <- subset(xenium.obj, subset = nCount_Xenium > 0)

gene_panel_path <- file.path(xenium_dir, "gene_panel.json")
panel_source <- NULL
if (file.exists(gene_panel_path)) {
  gene_panel <- jsonlite::fromJSON(gene_panel_path)
  targets <- gene_panel$payload$targets
  panel_source <- setNames(
    data.frame(
      gene_panel = targets$source$identity$name,
      gene = targets$type$data$name,
      stringsAsFactors = FALSE
    ),
    c("gene_panel", "gene")
  )
}

get_gex_means <- function(xenium_obj, flex_obj) {
  xen_means <- data.frame(
    mean_counts = Matrix::rowMeans(xenium_obj[["Xenium"]]$counts),
    gene = rownames(xenium_obj[["Xenium"]]$counts),
    stringsAsFactors = FALSE
  ) %>%
    arrange(desc(mean_counts)) %>%
    mutate(Rank = dplyr::row_number())

  flex_means <- data.frame(
    mean_counts = Matrix::rowMeans(flex_obj[["RNA"]]$counts),
    gene = rownames(flex_obj[["RNA"]]$counts),
    stringsAsFactors = FALSE
  ) %>%
    arrange(desc(mean_counts)) %>%
    mutate(Rank = dplyr::row_number())

  merge(xen_means, flex_means, by = "gene", all.x = TRUE)
}

merged_means <- get_gex_means(xenium.obj, flex_data.obj)
if (!is.null(panel_source)) {
  merged_means <- merge(merged_means, panel_source, by.x = "gene", by.y = "gene", all.x = TRUE)
  merged_means <- na.omit(merged_means) %>% arrange(gene_panel)
  message(
    "Per-gene Flex vs Xenium correlation (log means) available in merged_means; ",
    nrow(merged_means), " genes with panel metadata."
  )
}

message("Reference: normalize, PCA, neighbors, clusters...")
DefaultAssay(flex_data.obj) <- "RNA"
flex_data.obj <- NormalizeData(flex_data.obj) %>%
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA(verbose = FALSE) %>%
  RunUMAP(dims = 1:15) %>%
  FindNeighbors(dims = 1:15) %>%
  FindClusters(resolution = 0.5)

message("Xenium: sketch + sketch assay workflow...")
DefaultAssay(xenium.obj) <- "Xenium"
xenium.obj <- NormalizeData(xenium.obj)
xenium.obj <- FindVariableFeatures(xenium.obj)
xenium.obj <- SketchData(
  object = xenium.obj,
  ncells = sketch_ncells,
  method = "LeverageScore",
  sketched.assay = "sketch"
)
DefaultAssay(xenium.obj) <- "sketch"
xenium.obj <- FindVariableFeatures(xenium.obj) %>%
  ScaleData() %>%
  RunPCA(npcs = 20) %>%
  RunUMAP(dims = 1:16, return.model = TRUE) %>%
  FindNeighbors(reduction = "pca", dims = 1:16) %>%
  FindClusters(resolution = 0.6)

flex_xen_common_genes <- intersect(rownames(xenium.obj), rownames(flex_data.obj))
message(length(flex_xen_common_genes), " shared genes for label transfer.")

flex_subset <- CreateSeuratObject(
  counts = flex_data.obj[["RNA"]]$counts[flex_xen_common_genes, ],
  meta.data = flex_data.obj@meta.data
) %>%
  NormalizeData() %>%
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA(verbose = FALSE)

message("Materializing counts for FindTransferAnchors (can be memory-heavy)...")
flex_data.obj[["RNA"]]$counts <- as(object = flex_data.obj[["RNA"]]$counts, Class = "dgCMatrix")
xenium.obj[["Xenium"]]$counts <- as(object = xenium.obj[["Xenium"]]$counts, Class = "dgCMatrix")

anchors_from_flex <- FindTransferAnchors(
  reference = flex_subset,
  query = xenium.obj,
  query.assay = "sketch",
  features = flex_xen_common_genes,
  dims = 1:20,
  reference.reduction = "pca"
)

label_transfer <- TransferData(
  anchorset = anchors_from_flex,
  refdata = flex_subset$cell_type,
  dims = 1:20
)
xenium.obj <- AddMetaData(object = xenium.obj, metadata = label_transfer, col.name = "predicted.id")

message("ProjectData + TransferSketchLabels to full dataset...")
DefaultAssay(xenium.obj) <- "sketch"
xenium.obj <- ProjectData(
  object = xenium.obj,
  assay = "Xenium",
  full.reduction = "pca.full",
  sketched.assay = "sketch",
  sketched.reduction = "pca",
  umap.model = "umap",
  dims = 1:16,
  refdata = list(cluster_full = "sketch_snn_res.0.6")
)
DefaultAssay(xenium.obj) <- "Xenium"

xenium.obj <- TransferSketchLabels(
  xenium.obj,
  sketched.assay = "sketch",
  reduction = "pca.full",
  dims = 1:16,
  refdata = list(predicted.id_full = "predicted.id"),
  k = 50,
  reduction.model = "umap",
  recompute.neighbors = FALSE,
  recompute.weights = FALSE,
  verbose = TRUE
)

xenium.obj@meta.data$cell_type <- xenium.obj@meta.data$predicted.id_full

cell_id <- colnames(xenium.obj)
group <- as.character(xenium.obj$cell_type)
color <- unname(custom_hex[group])
color[is.na(color)] <- ""

pred_score <- rep(NA_real_, length(cell_id))
if ("predicted.id_full.score" %in% colnames(xenium.obj@meta.data)) {
  pred_score <- xenium.obj@meta.data$predicted.id_full.score
} else if ("prediction.score.max" %in% colnames(xenium.obj@meta.data)) {
  pred_score <- xenium.obj@meta.data$prediction.score.max
}

out_df <- tibble::tibble(
  cell_id = cell_id,
  group = group,
  color = color,
  prediction_score = pred_score
)

readr::write_csv(out_df, output_csv, na = "")
message("Wrote ", nrow(out_df), " rows to ", output_csv)
