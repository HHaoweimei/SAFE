function M = updateM(train_data, y, k, lambda, mu)
    [n, d] = size(train_data);
    S = y * y'; 
    U = zeros(n, n);
    for i = 1:n
        
        sim_row = S(i, :);
        sim_row(i) = -inf; 
        
        
        [~, idx] = sort(sim_row, 'descend');
        k_actual = min(k, length(idx));
        U(i, idx(1:k_actual)) = 1;
    end
    
    U = max(U, U'); 
    
    M = zeros(n, n);
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

    for i = 1:n
        neighbor_idx = find(U(i,:));
        num_neighbors = length(neighbor_idx);
        
        if num_neighbors == 0
            M(i,i) = 1; 
            continue; 
        end
        
        
        D = train_data(neighbor_idx, :) - train_data(i, :);
        H = D * D' + 1e-6 * eye(num_neighbors); 
        
       
        lb = zeros(num_neighbors, 1);
        ub = ones(num_neighbors, 1);
        Aeq = ones(1, num_neighbors);
        beq = 1;
        
       
        try
            w = quadprog(2*lambda*H, [], [], [], Aeq, beq, lb, ub, [], options);
            M(i, neighbor_idx) = w';
        catch
            
            M(i, neighbor_idx) = 1/num_neighbors;
        end
    end

end