function [cost,U,V] = costSimpleMKKM_svd(PH,StepSigma,DirSigma,Sigma,numclass)

global nbcall
nbcall=nbcall+1;

Sigma = Sigma+ StepSigma * DirSigma;

Kmatrix = sumKbeta(PH,(Sigma.*Sigma));
[U,V,cost]= mykernelsvd(Kmatrix,numclass);