path='D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/MS_BERT'
setwd(path)

Produce_combined_quant_without_LFQ <- function(path, dataset){
  #dataset='HYC_GUI'
  pathMS_BERT=paste0(path, '/', dataset)
  prt1='_quant_sum6_precursor_area_diann_combine.csv'
  pgt1='_quant_sum6_protein_area_diann_combine.csv'
  #precursors
  MS_BERT_prs <- dir(pathMS_BERT, paste0('.*',prt1))
  if(length(MS_BERT_prs)==0){
    prt1='.mzML_precursor.csv'
    pgt1='.mzML_protein.csv'
    MS_BERT_prs = dir(pathMS_BERT, paste0('.*',prt1))
  }
    
  MS_BERT_precursor <- data.frame()
  for(i in MS_BERT_prs){
    T1 = read.csv(paste0(pathMS_BERT,'/',i))
    colnames(T1)[which(colnames(T1)=='transition_group_id')]='Precursor'
    colnames(T1)[which(colnames(T1)=='ProteinID')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='Protein')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='UniprotID')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='ProteinName')]='Protein_name'
    colnames(T1)[which(colnames(T1)=='precursor_quant')]='quant'
    T1=T1[T1$Protein_id!='nan', ]
    T1=T1[!grepl('_ECOLI',T1$Protein_name) & !grepl('^DECOY',T1$Protein_id), ]
    MS_BERT_precursor = rbind(MS_BERT_precursor,T1)
  }
  MS_BERT_precursor$file_name=gsub('.mzML','',MS_BERT_precursor$file_name)

  #Proteins
  MS_BERT_pgs <- dir(pathMS_BERT, paste0('.*',pgt1))
  MS_BERT_protein <- data.frame()
  for(i in MS_BERT_pgs){
    #i=MS_BERT_pgs[1]
    T1 = read.csv(paste0(pathMS_BERT,'/',i))
    colnames(T1)[which(colnames(T1)=='transition_group_id')]='Precursor'
    colnames(T1)[which(colnames(T1)=='ProteinID')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='Protein')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='UniprotID')]='Protein_id'
    colnames(T1)[which(colnames(T1)=='ProteinName')]='Protein_name'
    colnames(T1)[which(colnames(T1)=='Protein_quant')]='quant'
    T1=T1[T1$Protein_id!='nan', ]
    T1=T1[!grepl('_ECOLI',T1$Protein_name) & !grepl('^DECOY',T1$Protein_id), ]
    fn = gsub('_Protein.csv','',i)
    T1$file_name = fn
    MS_BERT_protein = rbind(MS_BERT_protein,T1)
  }
  MS_BERT_protein$file_name=gsub(pgt1,'',MS_BERT_protein$file_name)

  ltw_precursor <- function(data){
    #data=MS_BERT_precursor
    rn=unique(data$Precursor)
    cn=c('Protein_id', 'Protein_name', sort(unique(data$file_name)))
    D1=as.data.frame(matrix(ncol=length(cn), nrow=length(rn)))
    colnames(D1)=cn
    rownames(D1)=rn
    D1$Protein_id=''
    D1$Protein_name=''
    for(i in cn[-c(1,2)]){
      #i=cn[3]
      T1=data[data$file_name==i, ]
      tmap=T1$Protein_id
      names(tmap)=T1$Precursor
      tmap=tmap[tmap!='']
      T2=tapply(T1$quant,factor(T1$Precursor),mean,na.rm=T)
      D1[names(T2),i]=T2
      D1[names(tmap),'Protein_id']=tmap
      if(sum(!is.na(T1$Protein_name))>0){
        tmap2=T1$Protein_name
        names(tmap2)=T1$Precursor
        tmap2=tmap2[tmap2!='']
        D1[names(tmap2),'Protein_name']=tmap2
      }
    }
    D1
  }
  
  ltw_Protein <- function(data){
    #  data=MS_BERT_protein
    rn=unique(data$Protein_id)
    cn=c('Protein_id', 'Protein_name', sort(unique(data$file_name)))
    D1=as.data.frame(matrix(ncol=length(cn), nrow=length(rn)))
    colnames(D1)=cn
    rownames(D1)=rn
    D1$Protein_id=rn
    D1$Protein_name=''
    for(i in cn[-c(1,2)]){
      #i=cn[3]
      T1=data[data$file_name==i, ]
      T2=tapply(T1$quant,factor(T1$Protein_id),mean,na.rm=T)
      D1[names(T2),i]=T2
      if(sum(!is.na(T1$Protein_name))>0){
        tmap2=T1$Protein_name
        names(tmap2)=T1$Protein_id
        tmap2=tmap2[tmap2!='']
        D1[names(tmap2),'Protein_name']=tmap2
      }
    }
    D1
  }
  
  MS_BERT_precursor1=ltw_precursor(MS_BERT_precursor)
  MS_BERT_protein1  =ltw_Protein(MS_BERT_protein)
  write.table(cbind(Precursor=rownames(MS_BERT_precursor1),MS_BERT_precursor1),paste0(pathMS_BERT,'/MS_BERT_precursor1_',dataset,'.tsv'), sep='\t',row.names = F)
  write.table(MS_BERT_protein1,paste0(pathMS_BERT,'/MS_BERT_protein1_',dataset,'.tsv'), sep='\t',row.names = F)
}
###HYC set
Produce_combined_quant_without_LFQ(path, dataset='HYC_GUI')
###human set
Produce_combined_quant_without_LFQ(path, dataset='Tumor15_IPX0001981000')


###DIA-NN_lib res
path='D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/DIA_NN_19_lib'
setwd(path)
Produce_combined_quant_without_LFQ_DIA_NN <- function(path, dataset){
  #dataset='HYC_GUI'
  pathDIA_NN=paste0(path, '/', dataset)
  #precursors
  DIA_NN_prs <- dir(pathDIA_NN, '.*_lib_report.pr_matrix.tsv')
  DIA_NN_precursor <- data.frame()
  for(i in DIA_NN_prs){
    #i=DIA_NN_prs[1]
    T1 = read.csv(paste0(pathDIA_NN,'/',i),sep='\t')
    colnames(T1)[ncol(T1)]='quant'
    T1$Precursor=T1$Precursor.Id
    T1$Protein_id=T1$Protein.Group
    T1$Protein_name=T1$Protein.Names
    T1=T1[!grepl('_ECOLI',T1$Protein_name) & !grepl('^DECOY',T1$Protein_id), ]
    T1$file_name=gsub('_lib_report.pr_matrix.tsv','',i)
    T1=T1[,c('Precursor','Protein_id','Protein_name','quant','file_name')]
    DIA_NN_precursor = rbind(DIA_NN_precursor,T1)
  }
  #Proteins
  DIA_NN_pgs <- dir(pathDIA_NN, '.*_lib_report.pg_matrix.tsv')
  DIA_NN_Protein <- data.frame()
  for(i in DIA_NN_pgs){
    #i=DIA_NN_pgs[1]
    T1 = read.csv(paste0(pathDIA_NN,'/',i),sep='\t')
    colnames(T1)[ncol(T1)]='quant'
    T1$Protein_id=T1$Protein.Group
    T1$Protein_name=T1$Protein.Names
    T1=T1[!grepl('_ECOLI',T1$Protein_name) & !grepl('^DECOY',T1$Protein_id), ]
    T1$file_name=gsub('_lib_report.pg_matrix.tsv','',i)
    T1=T1[,c('Protein_id','Protein_name','quant','file_name')]
    DIA_NN_Protein = rbind(DIA_NN_Protein,T1)
  }
  
  ltw_precursor <- function(data){
    #data=DIA_NN_precursor
    rn=unique(data$Precursor)
    cn=c('Protein_id', 'Protein_name', sort(unique(data$file_name)))
    D1=as.data.frame(matrix(ncol=length(cn), nrow=length(rn)))
    colnames(D1)=cn
    rownames(D1)=rn
    D1$Protein_id=''
    D1$Protein_name=''
    for(i in cn[-c(1,2)]){
      T1=data[data$file_name==i, ]
      tmap=T1$Protein_id
      names(tmap)=T1$Precursor
      tmap=tmap[tmap!='']
      T2=tapply(T1$quant,factor(T1$Precursor),mean,na.rm=T)
      D1[names(T2),i]=T2
      D1[names(tmap),'Protein_id']=tmap
      if(sum(!is.na(T1$Protein_name))>0){
        tmap2=T1$Protein_name
        names(tmap2)=T1$Precursor
        tmap2=tmap2[tmap2!='']
        D1[names(tmap2),'Protein_name']=tmap2
      }
    }
    D1
  }
  
  ltw_Protein <- function(data){
    #  data=DIA_NN_Protein
    rn=unique(data$Protein_id)
    cn=c('Protein_id', 'Protein_name', sort(unique(data$file_name)))
    D1=as.data.frame(matrix(ncol=length(cn), nrow=length(rn)))
    colnames(D1)=cn
    rownames(D1)=rn
    D1$Protein_id=rn
    D1$Protein_name=''
    for(i in cn[-c(1,2)]){
      T1=data[data$file_name==i, ]
      T2=tapply(T1$quant,factor(T1$Protein_id),mean,na.rm=T)
      D1[names(T2),i]=T2
      if(sum(!is.na(T1$Protein_name))>0){
        tmap2=T1$Protein_name
        names(tmap2)=T1$Protein_id
        tmap2=tmap2[tmap2!='']
        D1[names(tmap2),'Protein_name']=tmap2
      }
      
    }
    D1
  }
  
  DIA_NN_precursor1=ltw_precursor(DIA_NN_precursor)
  DIA_NN_protein1  =ltw_Protein(DIA_NN_Protein)
  write.table(cbind(Precursor=rownames(DIA_NN_precursor1),DIA_NN_precursor1),paste0(pathDIA_NN,'/DIA_NN_precursor1_',dataset,'.tsv'), sep='\t',row.names = F)
  write.table(DIA_NN_protein1,paste0(pathDIA_NN,'/DIA_NN_protein1_',dataset,'.tsv'), sep='\t',row.names = F)
}
###HYC set
Produce_combined_quant_without_LFQ_DIA_NN(path, dataset='HYC_GUI')
###human set
Produce_combined_quant_without_LFQ_DIA_NN(path, dataset='Tumor15_IPX0001981000')

