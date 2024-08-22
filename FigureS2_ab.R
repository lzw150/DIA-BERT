setwd('D:/LZW/Guomics/MS-BERT/Benchmark/fdrs_DIANN19/')
library('cowplot')
pgs=read.csv('pgs.csv',row.names = 1)
colnames(pgs)=gsub('\\.','-',colnames(pgs))
prs=readxl::read_xlsx('Figure1.xlsx', sheet = 'prs_lib')
rownames(prs)=prs$Files
prs=prs[rownames(pgs),]

#pgs_Three_species_proteome
pgs_Three_species_proteome=pgs[1:6,c('DIA-NN', 'MS-BERT')]
rownames(pgs_Three_species_proteome)=paste0(rep(c('Three_species_proteome_'),each=6),c(1:6))
#Boxplot
lim2 <- 1.2*max(pgs_Three_species_proteome,na.rm=T)
pgs_Three_species_proteome_barplot <- barplot(t(pgs_Three_species_proteome) , beside=T , las=2, legend.text=T,col=c("#8FBFA4", "#D78B61") , ylim=c(0,lim2) , ylab="Number of proteins")

#prs_Three_species_proteome
prs_Three_species_proteome=prs[1:6,c('DIA-NN', 'MS-BERT')]
rownames(prs_Three_species_proteome)=paste0(rep(c('Three_species_proteome_'),each=6),c(1:6))
#Boxplot
lim4 <- 1.2*max(prs_Three_species_proteome,na.rm=T)
prs_Three_species_proteome_barplot <- barplot(t(prs_Three_species_proteome) , beside=T , las=2, legend.text=T,col=c("#8FBFA4", "#D78B61") , ylim=c(0,lim4) , ylab="Number of precursors")

#out to pdf
pdf('Barplot_three_species.pdf',height = 4, width = 5)
barplot(t(pgs_Three_species_proteome) , beside=T , las=2, legend.text=F,col=c("#8FBFA4", "#D78B61") , ylim=c(0,lim2) , ylab="Number of proteins")
barplot(t(prs_Three_species_proteome) , beside=T , las=2, legend.text=F,col=c("#8FBFA4", "#D78B61") , ylim=c(0,lim4) , ylab="Number of precursors")
dev.off()

###
boxplot2 <- function(data,main,lab1="Common",lab2="Unique",ylab1="Log2intensity",addpoints=TRUE,ylim1=NULL){
  #data=prs_Three_species_proteome_long; main="Human proteome (precursors)"
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
      myjitter <- jitter(rep(i, length(thisvalues)), amount=levelProportions[mylevels[i]]/3)
      points(myjitter, thisvalues, pch=20, col=rgb(0,0,0,.9)) 
    } 
  }
  
  # Add a legend
  legend("topright", legend = c(lab1, lab2), 
         col=c( "#8FBFA4", "#D78B61"),
         pch = 15, bty = "n", pt.cex = 3, cex = 1.2,  horiz = F, inset = c(0.1, 0.1))
}


#precursor and protein number for three species
T1=data.frame(var='Precursor', 
              type='MS-BERT', quant=c(prs_Three_species_proteome$`MS-BERT`))
T2=data.frame(var='Precursor', 
              type='DIA-NN', quant=c(prs_Three_species_proteome$`DIA-NN`))
T3=data.frame(var='Protein', 
              type='MS-BERT', quant=c(pgs_Three_species_proteome$`MS-BERT`))
T4=data.frame(var='Protein', 
              type='DIA-NN', quant=c(pgs_Three_species_proteome$`DIA-NN`))
prgs_Three_species_proteome_long=rbind(T1,T2,T3,T4)
prs_Three_species_proteome_long=rbind(T1,T2)
pgs_Three_species_proteome_long=rbind(T3,T4)
cat('ttest for Three_species_proteome precursor and protein quant comparison\n')
print(sapply(unique(prgs_Three_species_proteome_long$var),function(x){
  #x=unique(prgs_Three_species_proteome_long$var)[2]
  t.test(c(prgs_Three_species_proteome_long[prgs_Three_species_proteome_long$var==x & prgs_Three_species_proteome_long$type=='MS-BERT','quant']), 
         c(prgs_Three_species_proteome_long[prgs_Three_species_proteome_long$var==x & prgs_Three_species_proteome_long$type=='DIA-NN','quant']),paired = T)$p.value
}))
#  Precursor      Protein 
# 1.356084e-07 1.393667e-05 
#boxplot precursor and protein number for three species
pdf('Boxplot precursor and protein number for three species.pdf', width=3,height = 3)
boxplot2(prs_Three_species_proteome_long, main="Three_species proteome (precursors)",lab1='DIA-NN',lab2='MS-BERT',ylab1='Number of precursors')
boxplot2(pgs_Three_species_proteome_long, main="Three_species proteome (proteins)",lab1='DIA-NN',lab2='MS-BERT',ylab1='Number of proteins')
dev.off()


#DIANN proteome
DIA_NN_Human_proteome         <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Human_proteome')
DIA_NN_Three_species_proteome <- readxl::read_xlsx('DIA_NN_19_lib_proteome.xlsx',sheet = 'Three_species_proteome')
#MS_BERT proteome
MS_BERT_Human_proteome         <- readxl::read_xlsx('MS_BERT_proteome.xlsx',sheet = 'Human_proteome')
MS_BERT_Three_species_proteome <- readxl::read_xlsx('MS_BERT_proteome.xlsx',sheet = 'Three_species_proteome')

library(VennDiagram)
###comparison between MS-BERT and DIA-NN
#Three_species_proteome precursors
set1=MS_BERT_Three_species_proteome$Precursor
set2=DIA_NN_Three_species_proteome$Precursor
set1=set1[set1!='' & !is.na(set1)]; length(unique(set1)) #157179
set2=set2[set2!='' & !is.na(set2)]; length(unique(set2)) #129803
venn_Three_species_proteome_prs <- venn.diagram(
  x = list(set1, set2),
  category.names = c("MS-BERT", "DIA-NN"),
  filename = NULL,  # Don't save the plot to a file, display it directly
  fill = c("#D78B61" , "#8FBFA4"),
  alpha = 0.75,
  main='Three species proteome (precursors)',
  main.cex=1.5
)

#Three_species_proteome proteins
set1=MS_BERT_Three_species_proteome$Protein_id
set2=DIA_NN_Three_species_proteome$Protein_id
set1=set1[set1!='' & !is.na(set1)]; length(unique(set1)) #11656
set2=set2[set2!='' & !is.na(set2)]; length(unique(set2)) #10534
venn_Three_species_proteome_pgs <- venn.diagram(
  x = list(set1, set2),
  category.names = c("MS-BERT", "DIA-NN"),
  filename = NULL,  # Don't save the plot to a file, display it directly
  fill = c("#D78B61" , "#8FBFA4"),
  alpha = 0.75,
  main='Three species proteome (proteins)',
  main.cex=1.5
)
#out to pdf
pdf('Venn plot comparison between MS-BERT and DIA-NN (Three_species_proteome).pdf', width=6,height = 3)
plot_grid(venn_Three_species_proteome_prs, venn_Three_species_proteome_pgs,  nrow = 1)
dev.off()


