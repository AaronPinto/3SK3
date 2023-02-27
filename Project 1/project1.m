%% Part 1 (Pseudocode)
% float global_sum = 0; // Initialize the sum to 0
% float[] memory = [4 elements]; // allocate memory to store each local sum per processor
% int N = number of terms in the series;
% 
% // Divide the series into 4 equal parts (4 parallel processors), with each part containing N / 4 terms
% // calculate local sums, assign each part to a different processor
% for i = 0 to 4 do // create new thread per iteration to run on separate processors
%   // runs on separate processors
%   float local_sum = 0;
%   
%   for j = i to N do
%     if j & 1 == 0 then
%       local_sum += 1.0 / (j + 1);
%     else
%       local_sum -= 1.0 / (j + 1);
%     end if
%   next j += 4
%
%   // store local sums, each thread accesses a different part of memeory so that's safe
%   memory[i] = local_sum;
% next i
%
% waitAll(); // wait for all threads to finish
% 
% // calculate global sum on processor 1
% for (int i = 0; i < 4; i++) {
%   global_sum += memory[i];
% }
% 
% // return result
% return global_sum;

%% Part c
N = 1:2^10;
reference = log(2);
error = zeros(1, length(N));

for i = 1:length(N)
  sum = kahan_sum(N(i));
  error(i) = abs(reference - sum);
end

plot(N, error);
set(gca, 'YScale', 'log');
xlabel('Number of terms, N');
ylabel('Numerical error');
title('Numerical error of the Kahan sum algorithm for the alternating harmonic series');

abs(reference - kahan_sum(2^32))

%% Part 2, use Kahan summation algorithm to increase precision https://en.wikipedia.org/wiki/Kahan_summation_algorithm#The_algorithm
function sum = kahan_sum(N)
  sum = single(0);
  c = single(0);
  sign = single(1);

  for i = 1:N
    y = single(sign / i - c);
    t = sum + y;
    c = single((t - sum) - y);
    sum = t;
    sign = -sign;
  end
end
