path='D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/'
setwd(path)
#MS-BERT
Fdrcontrol <- function(dataset){
  #dataset='HYC_GUI';
  S1='MS_BERT';
  ptr1='_quant_sum6_precursor_area_diann_combine.csv'
  S1prs=dir(paste0(path,S1,'/',dataset), paste0('(.*)',ptr1))
  if(length(S1prs)==0){
    ptr1='.mzML_precursor.csv'
    S1prs=dir(paste0(path,S1,'/',dataset), paste0('(.*)',ptr1))
  }
  files=gsub(ptr1,'\\1',S1prs)
  res=list()
  for(i in c(1:length(files))){
    #i=2
    Tr=read.csv(paste0(path,S1,'/',dataset,'/',files[i],ptr1),header = T)
    colnames(Tr)[which(colnames(Tr)=='UniprotID')]='Protein_id'
    colnames(Tr)[which(colnames(Tr)=='Protein')]='Protein_id'
    colnames(Tr)[which(colnames(Tr)=='transition_group_id')]='Precursor'
    Tr$Peptides= gsub('[0-9]','',Tr$Precursor)
    if(grepl('HYC',dataset)){
      Tr=Tr[!grepl('_ECOLI',Tr$ProteinName), ]
    }
    Tr1=Tr[!grepl('^DECOY',Tr$Protein_id), ]
    Tr1$Peptidesc=gsub("\\(UniMod:[0-9]+\\)", "", Tr1$Precursor)
    #files[i]=gsub('.mzML','',files[i])
    res[[files[i]]]=Tr1[c('Precursor', 'Protein_id')]
  }
  res
}
MS_BERT_HYC_GUI <- Fdrcontrol('HYC_GUI')
MS_BERT_Tumor15_IPX0001981000 <- Fdrcontrol('Tumor15_IPX0001981000')
#output MS_BERT_Human_proteome
MS_BERT_Human_proteome <- data.frame()
for(i in names(MS_BERT_Tumor15_IPX0001981000)){
  #i=names(MS_BERT_Tumor15_IPX0001981000)[1]
  T1=MS_BERT_Tumor15_IPX0001981000[[i]][c('Precursor', 'Protein_id')]
  T1$Filename=i
  T1$Dataset='Human_tissue'
  MS_BERT_Human_proteome=rbind(MS_BERT_Human_proteome,T1)
}
#output MS_BERT_3_species_proteome
MS_BERT_3_species_proteome <- data.frame()
for(i in names(MS_BERT_HYC_GUI)){
  #i=names(MS_BERT_HYC_GUI)[1]
  T1=MS_BERT_HYC_GUI[[i]][c('Precursor', 'Protein_id')]
  T1$Filename=i
  T1$Dataset='HYC'
  MS_BERT_3_species_proteome=rbind(MS_BERT_3_species_proteome,T1)
}
writexl::write_xlsx(list(Human_proteome=MS_BERT_Human_proteome,Three_species_proteome=MS_BERT_3_species_proteome),'MS_BERT_proteome.xlsx')

#DIA-NN
res_DIANN <- function(S1,dataset){
  #dataset='HYC_GUI';
  #S1='DIA_NN_19_lib';
  if(S1=='DIA_NN_19_lib'){
    hz='_lib_report.pr_matrix.tsv'
  }
  if(S1=='DIA_NN_19_fas'){
    hz='_fas_report.pr_matrix.tsv'
  }
  ptr1=paste0('(.*)',hz)
  S1prs=dir(paste0(path,S1,'/',dataset), ptr1)
  files=gsub(ptr1,'\\1',S1prs)
  res=list()
  for(i in c(1:length(files))){
    #i=2
    Tr=read.csv(paste0(path,S1,'/',dataset,'/',files[i],hz),sep='\t',header = T)
    if(grepl('HYC',dataset)){
      Tr1=Tr[!grepl('_ECOLI',Tr$Protein.Names), ]
    }else{Tr1=Tr}
    res[[files[i]]]=data.frame(Precursor=Tr1$Precursor.Id, Peptidesc=paste0(Tr1$Stripped.Sequence,Tr1$Precursor.Charge), Peptides=Tr1$Stripped.Sequence, Protein_id=Tr1$Protein.Group)
  }
  res
}
DIA_NN_19_lib_HYC_GUI <- res_DIANN(S1='DIA_NN_19_lib','HYC_GUI')
DIA_NN_19_lib_Tumor15_IPX0001981000 <- res_DIANN(S1='DIA_NN_19_lib','Tumor15_IPX0001981000')

DIA_NN_19_fas_HYC_GUI <- res_DIANN(S1='DIA_NN_19_fas','HYC_GUI')
DIA_NN_19_fas_Tumor15_IPX0001981000 <- res_DIANN(S1='DIA_NN_19_fas','Tumor15_IPX0001981000')


#output DIA_NN_19_lib_human_proteome
DIA_NN_19_lib_human_proteome <- data.frame()
for(i in names(DIA_NN_19_lib_Tumor15_IPX0001981000)){
  #i=names(DIA_NN_19_lib_human)[1]
  T1=DIA_NN_19_lib_Tumor15_IPX0001981000[[i]][c('Precursor', 'Protein_id')]
  T1$Filename=i
  T1$Dataset='Human_tissue'
  DIA_NN_19_lib_human_proteome=rbind(DIA_NN_19_lib_human_proteome,T1)
}
#output DIA_NN_19_lib_3_species_proteome
DIA_NN_19_lib_3_species_proteome <- data.frame()
for(i in names(DIA_NN_19_lib_HYC_GUI)){
  #i=names(DIA_NN_19_lib_HYC_GUI)[1]
  T1=DIA_NN_19_lib_HYC_GUI[[i]][c('Precursor', 'Protein_id')]
  T1$Filename=i
  T1$Dataset='HYC'
  DIA_NN_19_lib_3_species_proteome=rbind(DIA_NN_19_lib_3_species_proteome,T1)
}
writexl::write_xlsx(list(Human_proteome=DIA_NN_19_lib_human_proteome,Three_species_proteome=DIA_NN_19_lib_3_species_proteome),'DIA_NN_19_lib_proteome.xlsx')


#Overlap
compare1 <- function(D1,D2,title){
  #D1=MS_BERT_HYC_GUI;D2=DIA_NN_19_lib_HYC_GUI;title='MS_BERT_vs_DIA_NN_19_lib_human'
  prs=t(sapply(names(D1),function(x){
    peps1=unique(D1[[x]]$Precursor)
    peps2=unique(D2[[x]]$Precursor)
    c(D1=length(peps1), D2=length(peps2), Rate=length(peps1)/length(peps2),
      Overlap=length(intersect(peps1,peps2)),Overlap_rate=length(intersect(peps1,peps2))/length(peps2))
  }))
  pgs=t(sapply(names(D1),function(x){
    pros1=unique(D1[[x]]$Protein_id)
    pros2=unique(D2[[x]]$Protein_id)
    c(D1=length(pros1), D2=length(pros2), Rate=length(pros1)/length(pros2),
      Overlap=length(intersect(pros1,pros2)),Overlap_rate=length(intersect(pros1,pros2))/length(pros2))
  }))
  res=list(prs=data.frame(Files=rownames(prs),prs),
           pgs=data.frame(Files=rownames(pgs),pgs))
  writexl::write_xlsx(res,paste0(title,'.xlsx'))
  res
}

MS_BERT_vs_DIA_NN_19_lib_HYC_GUI       = compare1(D1=MS_BERT_HYC_GUI,       D2=DIA_NN_19_lib_HYC_GUI,       title='1MS_BERT_vs_DIA_NN_19_lib_HYC_GUI')
MS_BERT_vs_DIA_NN_19_lib_Tumor15_IPX0001981000         = compare1(D1=MS_BERT_Tumor15_IPX0001981000,         D2=DIA_NN_19_lib_Tumor15_IPX0001981000,         title='1MS_BERT_vs_DIA_NN_19_lib_Tumor15_IPX0001981000')

MS_BERT_vs_DIA_NN_19_fas_HYC_GUI     = compare1(D1=MS_BERT_HYC_GUI, D2=DIA_NN_19_fas_HYC_GUI, title='1MS_BERT_vs_DIA_NN_19_fas_HYC_GUI')
MS_BERT_vs_DIA_NN_19_fas_Tumor15_IPX0001981000       = compare1(D1=MS_BERT_Tumor15_IPX0001981000,   D2=DIA_NN_19_fas_Tumor15_IPX0001981000,   title='1MS_BERT_vs_DIA_NN_19_fas_Tumor15_IPX0001981000')

