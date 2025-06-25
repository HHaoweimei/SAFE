function Outputs = build_label_manifold(train_data, train_p_target, k,kdtree)
% 构建标签流形并生成标签置信度矩阵。
[p,q]=size(train_p_target);
train_data = normr(train_data);
%kdtree = KDTreeSearcher(train_data);%构建训练数据的k-d树，加速最近邻搜索。
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
options = optimoptions('quadprog','Display', 'off','Algorithm','interior-point-convex' );
W = zeros(p,p);
%fprintf('Obtain graph matrix W...\n');
for i = 1:p
	train_data1 = train_data(neighbor(i,:),:);%取其k个邻居样本特征：train_data1。
	D = repmat(train_data(i,:),k,1)-train_data1;%计算样本与其邻居的差：D，并构建距离矩阵DD = D * D'。
	DD = D*D';
	lb = sparse(k,1);                          %lb: 权重的下界（0）。
	ub = ones(k,1);                             %ub: 权重的上界（1）。      
	Aeq = ub';                                %Aeq: 线性等式约束，保证权重和为1。
	beq = 1;                                  %beq: 等式约束的右端值，固定为1。
	w = quadprog(2*DD, [], [],[], Aeq, beq, lb, ub,[], options);
	W(i,neighbor(i,:)) = w';
end
fprintf('\n')
%fprintf('Generate the labeling confidence...\n');
M = sparse(p,p);
%fprintf('Obtain Hessian matrix...\n');
WT = W';
T =WT*W+ W*ones(p,p)*WT.*eye(p,p)-2*WT;     %权重矩阵的自乘+对角线加权约束减去对偶项 得到Hessian矩阵T。
T1 = repmat({T},1,q);                      %T1: 将Hessian矩阵扩展到多标签情景。
M = spblkdiag(T1{:});                      %spblkdiag: 构建稀疏块对角矩阵，用于多标签优化。
lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);         %lb和ub: 定义目标变量的下界和上界，ub为标签候选集train_p_target展开的列向量。
II = sparse(eye(p));                     %创建一个 p×p 的稀疏单位矩阵 II。
A = repmat(II,1,q);                       %II 被重复 q 次构造矩阵 A，用于约束每个样本的标签置信度和为1：
b=ones(p,1);
M = (M+M');

%fprintf('quadprog...\n');
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
Outputs= quadprog(M, [], [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);                       %使用quadprog求解优化问题，目标是最小化M定义的二次项，同时满足线性约束。
                                                 % 将优化结果Outputs从向量形式还原为矩阵p x q，表示每个样本的多标签置信度。
end
