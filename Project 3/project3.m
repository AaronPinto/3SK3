%% Project 3

% Load the raw mosaic image
mosaic = imread('raw_mosaic.png');

% Extract the R, G, and B channels
R = mosaic(1:2:end, 1:2:end);
G = mosaic(1:2:end, 2:2:end);
B = mosaic(2:2:end, 2:2:end);

% Create a set of training patches
patch_size = 5; % patch size
step_size = 1; % step size
patches_R = im2col(R, [patch_size, patch_size], 'distinct');
patches_G = im2col(G, [patch_size, patch_size], 'distinct');
patches_B = im2col(B, [patch_size, patch_size], 'distinct');

% Solve the linear system for each patch
coefficients_R = zeros(patch_size^2, 3);
coefficients_G = zeros(patch_size^2, 3);
coefficients_B = zeros(patch_size^2, 3);

for i = 1:size(patches_R, 2)
    % Form the linear system
    A = [G(patches_R(:,i)), B(patches_R(:,i))];
    b = R(patches_R(:,i));

    % Solve the linear system using least squares
    x = pinv(A)*b;
    coefficients_R(:, i) = x;
    
    % Repeat the process for G and B channels
    % ...
end

% Apply the coefficients to the mosaic image
demosaiced_R = zeros(size(R));
demosaiced_G = zeros(size(G));
demosaiced_B = zeros(size(B));

for i = 1:size(patches_R, 2)
    % Form the linear system
    A = [G(patches_R(:,i)), B(patches_R(:,i))];
    b = R(patches_R(:,i));

    % Apply the coefficients to obtain the missing values
    demosaiced_R(patches_R(:,i)) = A*coefficients_R(:,i);
    
    % Repeat the process for G and B channels
    % ...
end

% Combine the demosaiced channels to obtain the final image
demosaiced = cat(3, demosaiced_R, demosaiced_G, demosaiced_B);

% Evaluate the performance of the algorithm
rmse = sqrt(mean((demosaiced - ground_truth).^2, 'all'));

