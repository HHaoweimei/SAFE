function G = obtain_G(train_data, Y, k,beta,kdtree)
 
[p,~]=size(Y);
train_data = normr(train_data);
%kdtree = KDTreeSearcher(train_data);
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
G = zeros(p,p);

for i = 1:p
	
	y1 = Y(neighbor(i,:),:);                     
	Dy = repmat(Y(i,:),k,1)-y1;                   
	DyDy = Dy*Dy';                                                    
    DDDD = beta*DyDy ; 

    lb = sparse(k,1);
	ub = ones(k,1);
	Aeq = ub';
	beq = 1;
    
    if all(DDDD(:) == 0)
       
       f = zeros(k, 1);  
       
        options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'interior-point');
       w = linprog(f, [], [], Aeq, beq, lb, ub, options);
    else
        options = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex' );
        w = quadprog(2*DDDD, [], [],[], Aeq, beq, lb, ub,[], options);
    end
   
	G(i,neighbor(i,:)) = w';
end
fprintf('\n')
end

