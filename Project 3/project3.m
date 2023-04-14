%% Project 3

% Load the raw mosaic image
mosaic = imread('test1.png');

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
    % Form the linear system for the R channel
    A_R = double([G(patches_R(:,i)), B(patches_R(:,i))]);
    b_R = double(R(patches_R(:,i)));

    % Solve the linear system using least squares for the R channel
    x_R = pinv(A_R) * b_R;
    coefficients_R(:, i) = x_R;
    
    % Form the linear system for the G channel
    A_G = [R(patches_G(:,i)), B(patches_G(:,i))];
    b_G = G(patches_G(:,i));

    % Solve the linear system using least squares for the G channel
    x_G = pinv(A_G) * b_G;
    coefficients_G(:, i) = x_G;
    
    % Form the linear system for the B channel
    A_B = [R(patches_B(:,i)), G(patches_B(:,i))];
    b_B = B(patches_B(:,i));

    % Solve the linear system using least squares for the B channel
    x_B = pinv(A_B) * b_B;
    coefficients_B(:, i) = x_B;
end

% Apply the coefficients to the mosaic image
demosaiced_R = zeros(size(R));
demosaiced_G = zeros(size(G));
demosaiced_B = zeros(size(B));

for i = 1:size(patches_R, 2)
    % Form the linear system for the R channel
    A_R = [G(patches_R(:,i)), B(patches_R(:,i))];
    b_R = R(patches_R(:,i));

    % Apply the coefficients to obtain the missing values for the R channel
    demosaiced_R(patches_R(:,i)) = A_R*coefficients_R(:,i);
    
    % Form the linear system for the G channel
    A_G = [R(patches_G(:,i)), B(patches_G(:,i))];
    b_G = G(patches_G(:,i));

    % Apply the coefficients to obtain the missing values for the G channel
    demosaiced_G(patches_G(:,i)) = A_G*coefficients_G(:,i);
    
    % Form the linear system for the B channel
    A_B = [R(patches_B(:,i)), G(patches_B(:,i))];
    b_B = B(patches_B(:,i));

    % Apply the coefficients to obtain the missing values for the B channel
    demosaiced_B(patches_B(:,i)) = A_B*coefficients_B(:,i);
end

% Combine the demosaiced channels to obtain the final image
demosaiced = cat(3, demosaiced_R, demosaiced_G, demosaiced_B);

% Evaluate the performance of the algorithm
matlab_demosaic = demosaic(mosaic, 'rggb');

rmse = sqrt(mean((demosaiced - matlab_demosaic).^2, 'all'));

