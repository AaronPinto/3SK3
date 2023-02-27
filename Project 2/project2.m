%% Project 2
load("data for student\pts_R5_L40_N100_K21.mat");
load("data for student\observation_R5_L40_N100_K21.mat");
load("data for student\dist_R5_L40_N100_K21.mat");

N = size(pts_markers, 2); % number of markers
K = size(pts_o, 1); % number of observations

% initialize marker coordinates randomly
% p_est = repmat(squeeze(pts_markers(1, 1, :))', N, 1);
% p_est = rand(N, 3);
p_est = squeeze(mean(pts_markers));

% set convergence threshold
epsilon = 1e-6;

% set maximum number of iterations
max_iter = 10e10;

% set step size
alpha = 0.1;

% set scaling factor for error term
lambda = 0.0001;

% initialize iteration counter
iter = 0;

for i = 1:N
    while iter <= max_iter
        % compute gradient, and Hessian
        r = zeros(K, 1);
        J = zeros(K, 3);
        sum_p = zeros(3, 1);

        for k = 1:K
            p_iq_k = p_est(i, :)' - pts_o(k, :)'; % delta between estimated p and measured distance

            sqrt_pq = sqrt(p_iq_k' * p_iq_k); 

            r(k, :) = sqrt_pq - dist(i, k); % compute residual vector element

            J(k, :) = p_iq_k' / sqrt_pq; % compute Jacobian row

            sum_p = sum_p + (2 * p_est(i, :)' - reshape(pts_markers(k, i, :), 3, 1)); % continue adding gradient sum term
        end

        g = J' * r + lambda * sum_p; % compute gradient of the loss function

        H_J = J' * J + 2 * lambda * K * eye(3, 3); % compute Hessian of the loss function

        % update marker coordinates using Newton's method
        delta_p = -H_J \ g;

        % check for convergence
        if delta_p < epsilon
            break;
        end

        p_est(i, :) = p_est(i, :) + (alpha * delta_p)';

        % increment iteration counter
        iter = iter + 1;
    end
end

% Computer RMSE error
load("data for student\gt_R5_L40_N100_K21.mat");
mean_rmse = sqrt((1 / N) * sum(norm(squeeze(mean(pts_markers)) - pts_marks_gt) ^ 2))
rmse = sqrt((1 / N) * sum(norm(p_est - pts_marks_gt) ^ 2))