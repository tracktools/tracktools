setwd('E:/Mhyqadeau/article/synmodel/sig')
#---- loading library (required)
library(sp)
library(gstat)

#---- set variogram model
vario <- vgm(psill = 0.30, model = 'Exp', range = 500, nugget = 0)

#--- build grid
xrange = c(0, 3750)
yrange = c(0, 5000)
res = 50

grd <- expand.grid(
  x=seq(from=xrange[1],to=xrange[2], by=res),
  y=seq(from=yrange[1],to=yrange[2],by=res)
)

coordinates(grd)<-~x+y 
gridded(grd)<-TRUE


# --- initialize gstat model
gparam <- gstat(
  formula = logK ~1,
  locations=~x+y,
  dummy =T,
  beta = 0,
  model = vario,
  nmax=30,
)

# --- perform non conditionnal simulation 
logK_pred <- predict(gparam,newdata=grd,nsim=6)
#spplot(obj=logK_pred)

logK_pred@data$logK <- logK_pred@data$sim4 - 5
logK_pred@data$K <- 10^(logK_pred@data$logK)

# visualize simulations
spplot(obj=logK_pred['logK'],main='log10(K [m/s])')


# export data
k = logK_pred@data$K[seq(1,100*75)]
mat = matrix(k,nrow=100,byrow=TRUE)

write.table(mat,'K.dat',sep=',',
          row.names=FALSE,
          col.names=FALSE)


