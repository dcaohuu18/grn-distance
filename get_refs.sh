#!/bin/bash

mkdir -p scenic_ref
cd scenic_ref

# 1. Mouse Transcription Factor List (Your updated link)
wget https://raw.githubusercontent.com/aertslab/pySCENIC/master/resources/mm_mgi_tfs.txt -O mm_tfs.txt

# 2. Motif Annotation Table (m0.001-o0.0 threshold)
wget https://resources.aertslab.org/cistarget/motif2tf/motifs-v9-nr.mgi-m0.001-o0.0.tbl

# 3. cisTarget Ranking Database (mm10, 10kb TSS, Note the double underscores "__")
wget https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc9nr/gene_based/mm10__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather

cd ..