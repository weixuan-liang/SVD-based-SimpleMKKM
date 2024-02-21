function [grad] = simpleMKKMGrad_svd(PH,U,V,Sigma)

d=size(PH,3);
grad=zeros(d,1);
for k=1:d
     grad(k) = 2*Sigma(k)*trace(U'*PH(:,:,k)*V);  
end
grad = grad / (d-1);