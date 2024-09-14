setwd('D:/LZW/Guomics/DIA-BERT/Benchmark/fdrs_DIANN19')
library('cowplot')
pgs=readxl::read_xlsx('Figure1.xlsx', sheet = 'pgs_lib')
rks=sort(tapply(pgs$`DIA-BERT`,factor(pgs$Disease),mean,na.rm=T),decreasing = T)
rks_id=c()
ct=1
for(i in names(rks)){
  T1=pgs[pgs$Disease==i, ]
  T2=T1[order(T1$`DIA-BERT`, decreasing = T), ]
  id=paste(ct, c(1:nrow(T2)),sep='_')
  names(id)=T2$Files
  rks_id=c(rks_id, id)
  ct=ct+1
}
pgs$rks_id=rks_id[pgs$Files]
pgs=pgs[order(pgs$rks_id),]
rownames(pgs)=pgs$Files
write.csv(pgs,'pgs.csv')

prs=readxl::read_xlsx('Figure1.xlsx', sheet = 'prs_lib')
rownames(prs)=prs$Files
prs=prs[rownames(pgs),]

###comparison with DIA-NN library-free mode
prs_fas=readxl::read_xlsx('Figure1.xlsx', sheet = 'prs_fas')
rownames(prs_fas)=prs_fas$Files
prs_fas=prs_fas[rownames(pgs),]

pgs_fas=readxl::read_xlsx('Figure1.xlsx', sheet = 'pgs_fas')
rownames(pgs_fas)=pgs_fas$Files
pgs_fas=pgs_fas[rownames(pgs),]

#pgs_Human_proteome
pgs_Human_proteome=pgs[-c(1:6),c('DIA-NN', 'DIA-BERT')]
rownames(pgs_Human_proteome)=paste(pgs$rks_id[-c(1:6)], pgs$Disease[-c(1:6)], sep='_')

pgs_fas_Human_proteome=pgs_fas[-c(1:6),c('DIA-NN', 'DIA-BERT')]
rownames(pgs_fas_Human_proteome)=rownames(pgs_Human_proteome)


#prs_Human_proteome
prs_Human_proteome=prs[-c(1:6),c('DIA-NN', 'DIA-BERT')]
rownames(prs_Human_proteome)=rownames(pgs_Human_proteome)

prs_fas_Human_proteome=prs_fas[-c(1:6),c('DIA-NN', 'DIA-BERT')]
rownames(prs_fas_Human_proteome)=rownames(pgs_Human_proteome)


###
boxplot2 <- function(data,main,lab1="Common",lab2="Unique",ylab1="Log2intensity",addpoints=TRUE,ylim1=NULL){
  #data=prs_Human_proteome_long; main="Human proteome (precursors)"
  par(mar=c(3,4,3,1))
  myplot <- boxplot(quant ~ type*var , data=data, 
                    boxwex=0.4 , ylab=ylab1, ylim=ylim1,
                    main=main, 
                    col=c( "#8FBFA4", "#D78B61"),  
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
      points(myjitter, thisvalues, pch=20, col=rgb(0,0,0,.9),cex=0.6) 
    } 
  }
  
  # Add a legend
  legend("topright", legend = c(lab1, lab2), 
         col=c( "#8FBFA4", "#D78B61"),
         pch = 15, bty = "n", pt.cex = 3, cex = 1.2,  horiz = F, inset = c(0.1, 0.1))
}


#precursor number for tissue
T1=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(prs_Human_proteome)), 
              type='DIA-NN', quant=c(prs_Human_proteome$`DIA-NN`))
T2=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(prs_Human_proteome)), 
              type='DIA-BERT', quant=c(prs_Human_proteome$`DIA-BERT`))
prs_Human_proteome_long=rbind(T1,T2)
cat('ttest for precursor identification comparison\n')
print(sapply(unique(prs_Human_proteome_long$var),function(x){
  #x=unique(prs_Human_proteome_long$var)[2]
  t.test(c(prs_Human_proteome_long[prs_Human_proteome_long$var==x & prs_Human_proteome_long$type=='DIA-BERT','quant']), 
         c(prs_Human_proteome_long[prs_Human_proteome_long$var==x & prs_Human_proteome_long$type=='DIA-NN','quant']),paired = T)$p.value
}))
# 2_cervical cancer        3_gallbladder cancer                4_myosarcoma 5_pancreatic adenocarcinoma 
# 0.0007327964                0.0003632959                0.0060382473                0.0058489910 
#6_gastric carcinoma 
# 0.0372826035 
T1=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(pgs_Human_proteome)), 
              type='DIA-NN', quant=c(pgs_Human_proteome$`DIA-NN`))
T2=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(pgs_Human_proteome)), 
              type='DIA-BERT', quant=c(pgs_Human_proteome$`DIA-BERT`))
pgs_Human_proteome_long=rbind(T1,T2)
cat('ttest for protein identification comparison\n')
print(sapply(unique(pgs_Human_proteome_long$var),function(x){
  #x=unique(pgs_Human_proteome_long$var)[2]
  t.test(c(pgs_Human_proteome_long[pgs_Human_proteome_long$var==x & pgs_Human_proteome_long$type=='DIA-BERT','quant']), 
         c(pgs_Human_proteome_long[pgs_Human_proteome_long$var==x & pgs_Human_proteome_long$type=='DIA-NN','quant']),paired = T)$p.value
}))
#  2_cervical cancer        3_gallbladder cancer                4_myosarcoma 5_pancreatic adenocarcinoma 
#   0.0023270504                0.0040181348                0.0005859145                0.0014723309 
#6_gastric carcinoma 
#   0.0194088939 

#boxplot precursor and protein number for tissue
pdf('Boxplot precursor and protein number for tissue.pdf', width=5,height = 4)
boxplot2(prs_Human_proteome_long, main="Human proteome (precursors)",lab1='DIA-NN',lab2='DIA-BERT',ylab1='Number of precursors',ylim1=c(0,80000))
boxplot2(pgs_Human_proteome_long, main="Human proteome (proteins)",lab1='DIA-NN',lab2='DIA-BERT',ylab1='Number of proteins',ylim1=c(0,10000))
dev.off()


###comparison with DIA-NN library-free mode
#precursor number for tissue
T1=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(prs_fas_Human_proteome)), 
              type='DIA-NN', quant=c(prs_fas_Human_proteome$`DIA-NN`))
T2=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(prs_fas_Human_proteome)), 
              type='DIA-BERT', quant=c(prs_fas_Human_proteome$`DIA-BERT`))
prs_fas_Human_proteome_long=rbind(T1,T2)
cat('ttest for precursor identification comparison with DIA-NN library-free mode\n')
print(sapply(unique(prs_fas_Human_proteome_long$var),function(x){
  #x=unique(prs_fas_Human_proteome_long$var)[2]
  t.test(c(prs_fas_Human_proteome_long[prs_fas_Human_proteome_long$var==x & prs_fas_Human_proteome_long$type=='DIA-BERT','quant']), 
         c(prs_fas_Human_proteome_long[prs_fas_Human_proteome_long$var==x & prs_fas_Human_proteome_long$type=='DIA-NN','quant']),paired = T)$p.value
}))
# 2_cervical cancer        3_gallbladder cancer                4_myosarcoma 5_pancreatic adenocarcinoma 
#  0.0015391425                0.0015964291                0.0025446320                0.0003144746 
#6_gastric carcinoma 
#  0.0017492984 
T1=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(pgs_fas_Human_proteome)), 
              type='DIA-NN', quant=c(pgs_fas_Human_proteome$`DIA-NN`))
T2=data.frame(var=gsub('(.*)_(.*)_(.*)','\\1_\\3',rownames(pgs_fas_Human_proteome)), 
              type='DIA-BERT', quant=c(pgs_fas_Human_proteome$`DIA-BERT`))
pgs_fas_Human_proteome_long=rbind(T1,T2)
cat('ttest for protein identification comparison with DIA-NN library-free mode\n')
print(sapply(unique(pgs_fas_Human_proteome_long$var),function(x){
  #x=unique(pgs_fas_Human_proteome_long$var)[2]
  t.test(c(pgs_fas_Human_proteome_long[pgs_fas_Human_proteome_long$var==x & pgs_fas_Human_proteome_long$type=='DIA-BERT','quant']), 
         c(pgs_fas_Human_proteome_long[pgs_fas_Human_proteome_long$var==x & pgs_fas_Human_proteome_long$type=='DIA-NN','quant']),paired = T)$p.value
}))
#  2_cervical cancer        3_gallbladder cancer                4_myosarcoma 5_pancreatic adenocarcinoma 
#   0.0016186430                0.0023900677                0.0003279147                0.0004883134 
#6_gastric carcinoma 
#   0.0052893210 

#boxplot precursor and protein number for tissue
pdf('Boxplot precursor and protein number for tissue comparison with DIA-NN library-free mode.pdf', width=5,height = 4)
boxplot2(prs_fas_Human_proteome_long, main="Human proteome (precursors)",lab1='DIA-NN',lab2='DIA-BERT',ylab1='Number of precursors',ylim1=c(0,80000))
boxplot2(pgs_fas_Human_proteome_long, main="Human proteome (proteins)",lab1='DIA-NN',lab2='DIA-BERT',ylab1='Number of proteins',ylim1=c(0,10000))
dev.off()


#######################overlapping with DIA-NN library-base mode
#DIANN proteome
DIA_NN_Human_proteome         <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Human_proteome')
DIA_NN_Three_species_proteome <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Three_species_proteome')
#DIA_BERT proteome
DIA_BERT_Human_proteome         <- readxl::read_xlsx('DIA_BERT_proteome.xlsx',sheet = 'Human_proteome')
DIA_BERT_Three_species_proteome <- readxl::read_xlsx('DIA_BERT_proteome.xlsx',sheet = 'Three_species_proteome')


library(VennDiagram)
###comparison between DIA-BERT and DIA-NN
#Human_proteome precursors
set1=DIA_BERT_Human_proteome$Precursor
set2=DIA_NN_Human_proteome$Precursor
set1=set1[set1!='' & !is.na(set1)]; length(unique(set1)) #149926
set2=set2[set2!='' & !is.na(set2)]; length(unique(set2)) #120130
venn_Human_proteome_prs <- venn.diagram(
  x = list(set1, set2),
  category.names = c("DIA-BERT", "DIA-NN"),
  filename = NULL,  # Don't save the plot to a file, display it directly
  fill = c("#D78B61" , "#8FBFA4"),
  alpha = 0.75,
  main='Human proteome (precursors)',
  main.cex=1.5
)

#Human_proteome proteins
set1=DIA_BERT_Human_proteome$Protein_id
set2=DIA_NN_Human_proteome$Protein_id
set1=set1[set1!='' & !is.na(set1)]; length(unique(set1)) #11141
set2=set2[set2!='' & !is.na(set2)]; length(unique(set2)) #9014
venn_Human_proteome_pgs <- venn.diagram(
  x = list(set1, set2),
  category.names = c("DIA-BERT", "DIA-NN"),
  filename = NULL,  # Don't save the plot to a file, display it directly
  fill = c("#D78B61" , "#8FBFA4"),
  alpha = 0.75,
  main='Human proteome (proteins)',
  main.cex=1.5
)

#out to pdf
pdf('Venn plot comparison between DIA-BERT and DIA-NN (Human_proteome).pdf', width=6,height = 3)
plot_grid(venn_Human_proteome_prs, venn_Human_proteome_pgs,  nrow = 1)
dev.off()
