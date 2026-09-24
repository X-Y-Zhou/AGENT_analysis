# Cell annotation forked from https://github.com/pachterlab/GSP_2019/blob/master/ECCITE%20CTCL.ipynb

import numpy as np
import matplotlib.pyplot as plt
import velocyto as vcy
import protaccel.protaccel as pa

[prot_count_array,prot_cells,adt_names] = pa.import_prot_data('dataset/real_data/GSM3596101_CTCL-ADT-count.csv')
vlm = vcy.VelocytoLoom("dataset/real_data/190426_CTCL.loom")

vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=100, min_expr_cells=20)
vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
plt.show()

gene_dict = {'B220 (CD45R)':['PTPRC'], 'B7-H1 (PD-L1)':['CD274'],  'C-kit (CD117)':['KIT'], 'CCR7':['CCR7'], 'CD11b':['ITGAM'], 'CD11c':['ITGAX'], 'CD138':['SDC1'], 'CD14':['CD14'], 'CD16':['FCGR3A'], 'CD19':['CD19'], 'CD1a':['CD1A'], 'CD2':['CD2'], 'CD223 (lag3)':['LAG3'],  'CD26 (Adenosine)':['DPP4'], 'CD27':['CD27'], 'CD28':['CD28'], 'CD3':['CD3E'], 'CD34':['CD34'], 'CD366 (tim3)':['HAVCR2'], 'CD4':['CD4'], 'CD44':['CD44'], 'CD45':['PTPRC'], 'CD45RA':['PTPRC'], 'CD45RO':['PTPRC'], 'CD5':['CD5'], 'CD56':['NCAM1'], 'CD62L':['SELL'], 'CD66b':['CEACAM8'], 'CD69':['CD69'], 'CD7':['CD7'],  'CD8':['CD8A'], 'CTLA4':['CTLA4'], 'EpCAM (CD326)':['EPCAM'], 'HLA-A,B,C':['HLA-A'], 'IL7Ralpha (CD127)':['IL7R'],   'LAMP1':['LAMP1'], 'MHCII (HLA-DR)':['HLA-DRA'], 'Ox40 (CD134)':['TNFRSF4'], 'PD-1 (CD279)':['PDCD1'], 'PD-L1 (CD274)':['CD274'], 'PD1 (CD279)':['PDCD1'],  'Siglec-8':['SIGLEC8']}
InvertDict = lambda d: dict( (v,k) for k in d for v in d[k] )
prot_dict = InvertDict(gene_dict)
mrna_targets = list(prot_dict.keys())
pa.enforce_protein_filter(vlm,mrna_targets,adt_names)

vlm.filter_genes(by_cv_vs_mean=True,by_detection_levels=True)

first_char = vlm.ca['CellID'][0].find(':')+1
last_char = -1

[prot_count_array, shared_cells, prot_cells] = pa.shared_cells_filter(vlm, prot_cells, prot_count_array, first_char, last_char)

pa.impute(vlm, prot_count_array, k=800, impute_in_prot_space=True, size_norm=False, impute_in_pca_space=False)

t_cl =  [0,0,1,2,1]
[cluster_ID, num_clusters] = pa.identify_clusters(vlm,vlm.connectivity,
                                                  correct_tags=True,tag_correction_list=t_cl,
                                                  method_name='ModularityVertexPartition')
