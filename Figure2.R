#visualization function
LFQPlot_sper <- function(data, title, type='HYC', pdf1='',xlim1,xlim2){
  #data=pgs1_ori_HYC_GUI; title='Protein_intensities'; type='HYC'; pdf1=''
  LFQ=list()
  library('ggpointdensity')
  library('ggplot2')
  library('viridis')
  library('cowplot')
  #library(patchwork)
  #data=pr1;title='DIANN';type='HYC'
  if(sum(colnames(data)=='Protein_name')>0){
    id=sapply(data$Protein_name, function(x){strsplit(x, ';')[[1]][1]})
  }else{
    id=sapply(data$Protein, function(x){strsplit(x, ';')[[1]][1]})
  }
  sp=gsub('(.*)_(.*)', '\\2', id)
  D1=data[(ncol(data)-5): ncol(data)]
  A=apply(D1[,c(1:3)],1,function(x){mean(x,na.rm=T)})
  B=apply(D1[,c(4:6)],1,function(x){mean(x,na.rm=T)})
  D2=data.frame(log2B=log2(B),log2ABratio=log2(A/B), sp=sp)
  #print(summary(D2$log2B))
  #print(summary(D2$log2ABratio))
  p1 <- ggplot(data = D2[D2$sp=="YEAST",], mapping = aes(x = log2B, y = log2ABratio)) +
    geom_pointdensity(adjust = 4,na.rm=T,show.legend=F, size = 0.2) +
    scale_color_viridis(option = "B",direction = -1,begin=0,end=0.8,alpha = 0.3,discrete = F) + 
    geom_hline(yintercept=log2(2),color=c('orange'))+
    labs(tag = title, title = 'YEAST') +
    ylim(-10,10) +
    xlim(xlim1,xlim2) +
    theme_classic()
  
  p2 <- ggplot(data = D2[D2$sp=="HUMAN",], mapping = aes(x = log2B, y = log2ABratio)) +
    geom_pointdensity(adjust = 4,na.rm=T,show.legend=F, size = 0.2) +
    scale_color_viridis(option = "D",direction = -1,begin=0,end=0.8,alpha = 0.3) + 
    geom_hline(yintercept=log2(1),color=c('green'))+
    labs(tag = title, title = 'HUMAN') +
    ylim(-10,10) +
    xlim(xlim1,xlim2) +
    theme_classic()
  
  p3 <- ggplot(data = D2[D2$sp=="CAEEL",], mapping = aes(x = log2B, y = log2ABratio)) +
    geom_pointdensity(adjust = 4,na.rm=T,show.legend=F, size = 0.2) +
    scale_color_viridis(option = "G",direction = -1,begin=0,end=0.8,alpha = 0.3) + 
    geom_hline(yintercept=log2(1/1.3),color=c('skyblue'))+
    labs(tag = title, title = 'CAEEL') +
    ylim(-10,10) +
    xlim(xlim1,xlim2) +
    theme_classic()
  if(pdf1==''){
    plot_grid(p1,p2,p3, nrow = 1)
  }else{
    pdf(paste0(pdf1, '.pdf'), width = 15, height = 4)
    print(plot_grid(p1,p2,p3, nrow = 1))
    dev.off()
  }
  
}

#plot_grid(p1+p2+p3, nrow = 1)
LFQbox <- function(data, title, type='HYC', ylim1){
  #data=pr1;title='DIANN';type='HYC'
  if(sum(colnames(data)=='Protein_name')>0){
    id=sapply(data$Protein_name, function(x){strsplit(x, ';')[[1]][1]})
  }else{
    id=sapply(data$Protein, function(x){strsplit(x, ';')[[1]][1]})
  }
  sp=gsub('(.*)_(.*)', '\\2', id)
  D1=data[(ncol(data)-5): ncol(data)]
  A=apply(D1[,c(1:3)],1,function(x){mean(x,na.rm=T)})
  B=apply(D1[,c(4:6)],1,function(x){mean(x,na.rm=T)})
  if(type=='HYC'){
    boxplot(log2(A[sp!='ECOLI']/B[sp!='ECOLI'])~sp[sp!='ECOLI'], cex=0.5, pch=19, border=c('skyblue', 'green', 'orange'), col='white',
            ylab='log2ratios', xlab='', main=title, ylim=ylim1)
    abline(h=c(log2(2),0,log2(1/1.3)),col=c('orange', 'green', 'skyblue'),lty=2)
  }
}

###
cclat <- function(data){
  #data=pgs1_ori_HYC_GUI
  if(sum(colnames(data)=='Protein_name')>0){
    id=sapply(data$Protein_name, function(x){strsplit(x, ';')[[1]][1]})
  }else{
    id=sapply(data$Protein, function(x){strsplit(x, ';')[[1]][1]})
  }
  sp=gsub('(.*)_(.*)', '\\2', id)
  D1=data[(ncol(data)-5): ncol(data)]
  CV1=median(apply(D1[,c(1:3)],1,function(x){sd(x,na.rm=T)/mean(x,na.rm=T)}), na.rm=T)
  CV2=median(apply(D1[,c(4:6)],1,function(x){sd(x,na.rm=T)/mean(x,na.rm=T)}), na.rm=T)
  A=apply(D1[,c(1:3)],1,function(x){mean(x,na.rm=T)})
  B=apply(D1[,c(4:6)],1,function(x){mean(x,na.rm=T)})
  HUMAN=median(log2(A[sp=='HUMAN']/B[sp=='HUMAN']),na.rm=T)
  YEAST=median(log2(A[sp=='YEAST']/B[sp=='YEAST']),na.rm=T)
  CAEEL=median(log2(A[sp=='CAEEL']/B[sp=='CAEEL']),na.rm=T)
  cat('\nCVs:','\n',CV1,'\n',CV2,'\nmedian of CV', '\n',median(c(CV1, CV2)))
  cat('\nHUMAN: ',HUMAN, '\nYEAST: ',YEAST, '\nCAEEL: ',CAEEL,'\n')
}
path='D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/MS_BERT'
setwd(path)

###HYC set
#MS-BERT 
dataset='HYC_GUI'
pathMS_BERT_HYC_GUI=paste0(path, '/', dataset)
prs1_ori_HYC_GUI <- read.table('D:/LZW/Guomics/MS-BERT/Benchmark/MaxLFQ/model_astral0_decoy_240705/res_240705/prs6_TIC_RT_dependent_occupancynormal.tsv', sep='\t',header = T)
prs1_ori_HYC_GUI <- prs1_ori_HYC_GUI[!grepl('_ECOLI',prs1_ori_HYC_GUI$Protein_name), ]
rownames(prs1_ori_HYC_GUI) <- prs1_ori_HYC_GUI$Precursor
pgs1_ori_HYC_GUI <- read.table('D:/LZW/Guomics/MS-BERT/Benchmark/MaxLFQ/model_astral0_decoy_240705/res_240705/pgs1_TIC_RT_dependent_MaxLFQ.tsv', sep='\t', header=T)
pgs1_ori_HYC_GUI <- pgs1_ori_HYC_GUI[pgs1_ori_HYC_GUI$Protein!='', ]
pgs1_ori_HYC_GUI <- pgs1_ori_HYC_GUI[!grepl('_ECOLI',pgs1_ori_HYC_GUI$Protein_name), ]
rownames(pgs1_ori_HYC_GUI) <- pgs1_ori_HYC_GUI$Protein
pgs1_ori_HYC_GUI[pgs1_ori_HYC_GUI==0]=NA

###MS-BERT 
#prs
cclat(prs1_ori_HYC_GUI)
#pgs
cclat(pgs1_ori_HYC_GUI)
#
LFQPlot_sper(data=pgs1_ori_HYC_GUI,title='',type='HYC', pdf1=paste0('..','/pgs1_ori_HYC_GUI'),xlim1=14,xlim2=33) ###
LFQPlot_sper(data=prs1_ori_HYC_GUI,title='',type='HYC', pdf1=paste0('..','/prs1_ori_HYC_GUI'),xlim1=14,xlim2=33) ###
pdf('D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/Boxplot_HYC_MS_BERT.pdf', width = 5, height = 5)
LFQbox(data=prs1_ori_HYC_GUI,title='MS_BERT_prs',type='HYC', c(-5, 5))
LFQbox(data=pgs1_ori_HYC_GUI,title='MS_BERT_pgs',type='HYC', c(-5, 5))
dev.off()

#DIANN set for comparison
DIANN_prs_HYC_GUI <- read.csv('D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/DIA_NN_19_lib/HYC_GUI_lib_combine_report.pr_matrix.tsv',sep='\t',header=T)
rownames(DIANN_prs_HYC_GUI) <- DIANN_prs_HYC_GUI$Precursor.Id
DIANN_prs_HYC_GUI1 <- cbind(Protein_name=DIANN_prs_HYC_GUI$Protein.Names, DIANN_prs_HYC_GUI)
DIANN_pgs_HYC_GUI <- read.csv('D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/DIA_NN_19_lib/HYC_GUI_lib_combine_report.pg_matrix.tsv',sep='\t',header=T)
rownames(DIANN_pgs_HYC_GUI) <- DIANN_pgs_HYC_GUI$Protein.Group
DIANN_pgs_HYC_GUI1 <- cbind(Protein_name=DIANN_pgs_HYC_GUI$Protein.Names, DIANN_pgs_HYC_GUI)

###DIANN
cclat(DIANN_prs_HYC_GUI1)
cclat(DIANN_pgs_HYC_GUI1)

###plot for DIANN 
LFQPlot_sper(data=DIANN_prs_HYC_GUI1,title='',type='HYC', pdf1=paste0(pathMS_BERT_HYC_GUI,'DIANN_prs_HYC_GUI1'),xlim1=14,xlim2=33)
LFQPlot_sper(data=DIANN_pgs_HYC_GUI1,title='',type='HYC', pdf1=paste0(pathMS_BERT_HYC_GUI,'DIANN_pgs_HYC_GUI1'),xlim1=14,xlim2=33)
LFQbox(data=DIANN_prs_HYC_GUI1,title='DIANN_prs',type='HYC', c(-10, 10))
LFQbox(data=DIANN_pgs_HYC_GUI1,title='DIANN_pgs',type='HYC', c(-10, 10))


#Protein comparison
commp_HYC_GUI=intersect(rownames(pgs1_ori_HYC_GUI), rownames(DIANN_pgs_HYC_GUI1))
pesp_HYC_GUI = sapply(5:0, function(x){
  cor.test(log2(unlist(pgs1_ori_HYC_GUI[commp_HYC_GUI, ncol(pgs1_ori_HYC_GUI)-x])), 
           log2(unlist(DIANN_pgs_HYC_GUI1[commp_HYC_GUI, ncol(DIANN_pgs_HYC_GUI1)-x])), 
           method='pearson', use='pairwise.complete.obs')$estimate
})
names(pesp_HYC_GUI) = colnames(pgs1_ori_HYC_GUI[ncol(pgs1_ori_HYC_GUI)-c(5:0)])
mean(pesp_HYC_GUI) #0.9456043

####precursor_ori comparison
comm_HYC_GUI=intersect(gsub('^_','',rownames(prs1_ori_HYC_GUI)), rownames(DIANN_prs_HYC_GUI1))
pes_HYC_GUI = sapply(5:0, function(x){
  cor.test(log2(unlist(prs1_ori_HYC_GUI[comm_HYC_GUI, ncol(prs1_ori_HYC_GUI)-x])), 
           log2(unlist(DIANN_prs_HYC_GUI1[comm_HYC_GUI, ncol(DIANN_prs_HYC_GUI1)-x])), method='pearson')$estimate
})
names(pes_HYC_GUI) = colnames(prs1_ori_HYC_GUI[ncol(prs1_ori_HYC_GUI)-c(5:0)])
mean(pes_HYC_GUI) #0.7885133

###
###MS-BERT 
cclat(prs1_ori_HYC_GUI[comm_HYC_GUI,])
cclat(pgs1_ori_HYC_GUI[commp_HYC_GUI,])

###DIANN
cclat(DIANN_prs_HYC_GUI1[comm_HYC_GUI,])
cclat(DIANN_pgs_HYC_GUI1[commp_HYC_GUI,])


