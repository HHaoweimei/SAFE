function Outputs = updateP(G,Y,alpha,gamma,H)
%Update label confidence Y

[p,q]=size(Y);

options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
tic
para_t = gamma/alpha;  %gp 
%G=eye(p);
T = 2*(eye(p)-G)'*(eye(p)-G)+2/para_t*eye(p);

toc
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});
lb=sparse(p*q,1);
ub=reshape(Y,p*q,1);
II = sparse(eye(p));
A = repmat(II,1,q);
b=ones(p,1);
tr = H;
f = reshape(tr, p*q, 1);
Outputs= quadprog(M, -2*(1/para_t)*f, [],[], A, b, lb, ub,[], options);
% Outputs= linprog( -2*(1/para_t)*f, [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);
end