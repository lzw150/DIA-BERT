path='D:/LZW/Guomics/DIA-BERT/Benchmark/fdrs_DIANN19/'
setwd(path)

#DIANN proteome
DIA_NN_Human_proteome         <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Human_proteome')
DIA_NN_Three_species_proteome <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Three_species_proteome')
#DIA_BERT proteome
DIA_BERT_Human_proteome         <- readxl::read_xlsx('DIA_BERT_proteome.xlsx',sheet = 'Human_proteome')
DIA_BERT_Three_species_proteome <- readxl::read_xlsx('DIA_BERT_proteome.xlsx',sheet = 'Three_species_proteome')

pgs=readxl::read_xlsx('Figure1.xlsx', sheet = 'pgs_lib')
pgs=pgs[order(pgs$`DIA-BERT`,decreasing = T),]
rownames(pgs)=pgs$Files


###quantification comparison
boxplot2 <- function(data,main,lab1="Common",lab2="Unique",ylab1="Log2intensity",addpoints=TRUE){
  #data=prs_Human_proteome_long; main="Human proteome (precursors)"
  par(mar=c(3,4,3,1))
  myplot <- boxplot(quant ~ type*var , data=data, 
                    boxwex=0.4 , ylab=ylab1, 
                    main=main, 
                    col=c( "slateblue1" , "tomato"),  
                    xaxt="n", cex=0.1,xlab = '')
  # To add the label of x axis
  my_names <- sapply(strsplit(myplot$names , '\\.') , function(x) x[[2]] )
  my_names <- my_names[seq(1 , length(my_names) , 2)]
  axis(1, 
       at = seq(1.5 , length(my_names)*2 , 2), 
       labels = my_names , las=2,
       tick=FALSE , cex=0.3)
  # Add the grey vertical lines
  for(i in seq(0.5 , length(my_names)*2+1 , 2)){ 
    abline(v=i,lty=1, col="grey")
  }
  if(addpoints==TRUE){
    # Add data points
    data$type2=paste(data$var,data$type,sep='_')
    mylevels <- sort(unique(data$type2))
    levelProportions <- table(data$type2)/nrow(data)
    for(i in 1:length(mylevels)){
      #i=2
      thislevel <- mylevels[i]
      thisvalues <- data[data$type2==thislevel, "quant"]
      # take the x-axis indices and add a jitter, proportional to the N in each level
      myjitter <- jitter(rep(i, length(thisvalues)), amount=levelProportions[mylevels[i]]/2)
      points(myjitter, thisvalues, pch=20, col=rgb(0,0,0,.9)) 
    } 
  }
  
  # Add a legend
  legend("topright", legend = c(lab1, lab2), 
         col=c("slateblue1" , "tomato"),
         pch = 15, bty = "n", pt.cex = 3, cex = 1.2,  horiz = F, inset = c(0.1, 0.1))
}

##comparison of quantification
comparison_of_quantification <- function(dataset,filenames,prs_file,pgs_file,rename,type,proteome1,proteome2,pdfwidth=8,subdir='DIA_BERT/',softaware='In DIA-BERT '){
  #dataset='HYC_GUI';filenames=rownames(pgs)[1:6]
  #prs_file='DIA_BERT_precursor1_HYC_GUI.tsv';pgs_file='DIA_BERT_protein1_HYC_GUI.tsv'
  #rename='Sample_';type='HYC';
  #proteome1=DIA_BERT_Three_species_proteome;proteome2=DIA_NN_Three_species_proteome
  #pdfwidth=8;subdir='DIA_BERT/';softaware='In DIA-BERT '
  cat(dataset,'\n',
      filenames,'\n',
      prs_file,'\n',
      pgs_file,'\n',
      rename,'\n',
      type,'\n')
  path_1=paste0(path,subdir,dataset)
  #precursors
  prs_quant_ <- read.table(paste0(path_1,'/',prs_file), sep='\t',header = T)
  Precursor_ <- prs_quant_$Precursor
  prs_quant_1 <- prs_quant_[-c(1:2)]
  rownames(prs_quant_1) <- Precursor_
  colnames(prs_quant_1) <- gsub('\\.','-', colnames(prs_quant_1))
  colnames(prs_quant_1) <- gsub('^X','', colnames(prs_quant_1))
  prs_quant_1=prs_quant_1[, filenames]
  colnames(prs_quant_1) <- paste0(rep(rename,length(filenames)), 1:length(filenames))
  prs_quant_1[prs_quant_1==0]=NA
  #proteins
  pgs_quant_ <- read.table(paste0(path_1,'/',pgs_file), sep='\t',header = T)
  pgs_quant_ <- pgs_quant_[pgs_quant_$Protein_id!='',]
  rownames(pgs_quant_) <- pgs_quant_$Protein_id
  colnames(pgs_quant_) <- gsub('\\.','-', colnames(pgs_quant_))
  colnames(pgs_quant_) <- gsub('^X','', colnames(pgs_quant_))
  pgs_quant_=pgs_quant_[, filenames]
  colnames(pgs_quant_) <- paste0(rep(rename,length(filenames)), 1:length(filenames))
  pgs_quant_[pgs_quant_==0]=NA
  
  #comparison of precursors
  colnames(proteome1)[colnames(proteome1)=='transition_group_id']='Precursor'
  colnames(proteome1)[colnames(proteome1)=='Protein']='Protein_id'
  colnames(proteome2)[colnames(proteome2)=='transition_group_id']='Precursor'
  colnames(proteome2)[colnames(proteome2)=='Protein']='Protein_id'
  set1=proteome1$Precursor[proteome1$Dataset==type]
  set2=proteome2$Precursor[proteome2$Dataset==type]
  set1=unique(set1[set1!='' & !is.na(set1)]); cat('precursors set1:',length(unique(set1)),'\n') #
  set2=unique(set2[set2!='' & !is.na(set2)]); cat('precursors set2:',length(unique(set2)),'\n') #
  prs_quant_1_unique=prs_quant_1[setdiff(set1, set2),]
  prs_quant_1_common=prs_quant_1[intersect(set1, set2),]
  prs_quant_1_long=data.frame()
  for(i in colnames(prs_quant_1)){
    #i=colnames(prs_quant_1)[1]
    T1=data.frame(var=i, type='Unique precursors', quant=log(prs_quant_1_unique[,i],2))
    T2=data.frame(var=i, type='Common precursors', quant=log(prs_quant_1_common[,i],2))
    prs_quant_1_long=rbind(prs_quant_1_long, T1, T2)
  }
  prs_quant_1_long=prs_quant_1_long[!is.na(prs_quant_1_long$quant) & prs_quant_1_long$quant!='', ]
  #ttest for precursor quant comparison
  cat('ttest for precursor quant comparison\n')
  print(sapply(colnames(prs_quant_1),function(x){
    t.test(log(prs_quant_1_unique[,x],2), 
           log(prs_quant_1_common[,x],2))$p.value
  }))
  
  #comparison of proteins
  set1=proteome1$Protein_id[proteome1$Dataset==type]
  set2=proteome2$Protein_id[proteome2$Dataset==type]
  set1=unique(set1[set1!='' & !is.na(set1)]); cat('proteins set1:',length(unique(set1)),'\n') #
  set2=unique(set2[set2!='' & !is.na(set2)]); cat('proteins set2:',length(unique(set2)),'\n') #
  pgs_quant__unique=pgs_quant_[setdiff(set1, set2),]
  pgs_quant__common=pgs_quant_[intersect(set1, set2),]
  pgs_quant__long=data.frame()
  for(i in colnames(pgs_quant_)){
    #i=colnames(pgs_quant_)[1]
    T1=data.frame(var=i, type='Unique proteins', quant=log(pgs_quant__unique[,i],2))
    T2=data.frame(var=i, type='Common proteins', quant=log(pgs_quant__common[,i],2))
    pgs_quant__long=rbind(pgs_quant__long, T1, T2)
  }
  pgs_quant__long=pgs_quant__long[!is.na(pgs_quant__long$quant) & pgs_quant__long$quant!='', ]
  #ttest for protein quant comparison
  cat('ttest for protein quant comparison\n')
  print(sapply(colnames(pgs_quant_),function(x){
    t.test(log(pgs_quant__unique[,x],2), 
           log(pgs_quant__common[,x],2))$p.value
  }))
  #boxplot for comparison between unique and common prs or pgs
  pdf(paste0(path,softaware,type,' proteome boxplot for comparison between unique and common prs or pgs.pdf'), width=pdfwidth,height = 5)
  boxplot2(prs_quant_1_long, main=paste0(type," proteome (precursors)"),addpoints=FALSE)
  boxplot2(pgs_quant__long, main=paste0(type," proteome (proteins)"),addpoints=FALSE)
  dev.off()
  return(list(prs_quant_1_long, pgs_quant__long))
}

#In DIA-BERT
quant_long_HYC_GUI_DIA_BERT       = comparison_of_quantification(dataset='HYC_GUI',filenames=rownames(pgs)[1:6],
                                                                prs_file='DIA_BERT_precursor1_HYC_GUI.tsv',pgs_file='DIA_BERT_protein1_HYC_GUI.tsv',
                                                                rename='Sample_',type='HYC',
                                                                proteome1=DIA_BERT_Three_species_proteome,proteome2=DIA_NN_Three_species_proteome)


#In DIA-NN
quant_long_HYC_GUI_DIA_NN       = comparison_of_quantification(dataset='HYC_GUI',filenames=rownames(pgs)[1:6],
                                                               prs_file='DIA_NN_precursor1_HYC_GUI.tsv',pgs_file='DIA_NN_protein1_HYC_GUI.tsv',
                                                               rename='Sample_',type='HYC',subdir='DIA_NN_19_lib/',softaware='In DIA-NN ',
                                                               proteome1=DIA_NN_Three_species_proteome,proteome2=DIA_BERT_Three_species_proteome)

