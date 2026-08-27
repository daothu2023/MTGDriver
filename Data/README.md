## Data

1. PPI_CPDB
- File format: contains undirected gene–gene edges:
  ```csv
  gene1,gene2
  TP53,BRCA1
  EGFR,KRAS
  ...
  Usage: These graphs define the structure (edge_index) for the GCN model.
2. Node Features
Each node is assigned a feature vector that integrates multiple biological signals. Specifically, we adopt four multi-omics features: somatic mutation rate, copy number alteration (CNA), DNA methylation, and gene expression, following the design of EMOGI. In addition, we incorporate pathway-derived features encoded as binary indicators, reflecting gene membership in curated pathways from databases such as KEGG.


