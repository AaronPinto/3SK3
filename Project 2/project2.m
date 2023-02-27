%% Project 2
load("data for student\pts_R5_L40_N100_K21.mat");
load("data for student\observation_R5_L40_N100_K21.mat");
load("data for student\dist_R5_L40_N100_K21.mat");
load("data for student\gt_R5_L40_N100_K21.mat");

p_est_init = squeeze(mean(pts_markers));
N = size(pts_markers, 2); % number of markers
lambda = 0.0001;

p_est = newton(p_est_init, pts_markers, pts_o, dist, lambda, 1e4);
rmse = sqrt((1 / N) * sum(norm(p_est - pts_marks_gt) ^ 2));
disp(rmse);

%% Fine tune lambda
% N = size(pts_markers, 2); % number of markers
% p_est_init = squeeze(mean(pts_markers));
% mean_rmse = sqrt((1 / N) * sum(norm(squeeze(mean(pts_markers)) - pts_marks_gt) ^ 2));

% set scaling factor for error term
% lambda_op = 0.0001;
% lambda_range = 0.01 * lambda_op:0.01 * lambda_op:100 * lambda_op;
% rmse_vals = zeros(1, length(lambda_range));

% for i = 1:length(lambda_range)
%     p_est = newton(p_est_init, pts_markers, pts_o, dist, lambda_range(i), 1e4);

    % Compute RMSE error
%     rmse_vals(i) = sqrt((1 / N) * sum(norm(p_est - pts_marks_gt) ^ 2));

%     disp(i);
% end

% plot(lambda_range, rmse_vals);
% set(gca, 'YScale', 'log');
% xlabel('Value of lambda');
% ylabel('RMSE Value');
% title('Converged RMSE against Lambda');

%% Fine tune initialization
% lambda = 0.0001; % equal to lambda_op
% 
% % set maximum number of iterations
% max_iter_range = 1e3:1e3:1e5;
% rmse_vals = zeros(3, length(max_iter_range));
% 
% % initialize marker coordinates 1 of 3 ways
% p_est_init_1 = repmat(squeeze(pts_markers(1, 1, :))', N, 1);
% p_est_init_2 = rand(N, 3);
% p_est_init_3 = squeeze(mean(pts_markers));
% 
% for i = 1:length(rmse_vals)
%     p_est_1 = newton(p_est_init_1, pts_markers, pts_o, dist, lambda, max_iter_range(i));
%     p_est_2 = newton(p_est_init_2, pts_markers, pts_o, dist, lambda, max_iter_range(i));
%     p_est_3 = newton(p_est_init_3, pts_markers, pts_o, dist, lambda, max_iter_range(i));
% 
%     rmse_vals(1, i) = sqrt((1 / N) * sum(norm(p_est_1 - pts_marks_gt) ^ 2));
%     rmse_vals(2, i) = sqrt((1 / N) * sum(norm(p_est_2 - pts_marks_gt) ^ 2));
%     rmse_vals(3, i) = sqrt((1 / N) * sum(norm(p_est_3 - pts_marks_gt) ^ 2));
% 
%     disp(i);
% end
% 
% plot(max_iter_range, rmse_vals);
% set(gca, 'YScale', 'log');
% xlabel('Max iterations');
% ylabel('RMSE Value');
% title('Max number of iterations needed to converge for RMSE');

%% Implement Newton's method
function p_est = newton(p_est_init, pts_markers, pts_o, dist, lambda, max_iter)
N = size(pts_markers, 2); % number of markers
K = size(pts_o, 1); % number of observations

p_est = p_est_init;

% set convergence threshold
epsilon = 1e-6;

% set step size
alpha = 0.1;

% initialize iteration counter
iter = 0;

for i = 1:N
    % update marker coordinates using Newton's method
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

        delta_p = -H_J \ g; % use left divide to compute inverse of H_j and divide g by it

        % check for convergence
        if delta_p < epsilon
            break;
        end

        p_est(i, :) = p_est(i, :) + (alpha * delta_p)'; % update the coordinate

        iter = iter + 1;
    end
end
end
