clear;clc;

%load dataset (lost)
load('split_lost.mat');

% hyper-parameters1
Maxiter = 20;
k = 10;
alpha = 30;
beta = 0.1;
gamma = 7.5;
lambda = 0.03;

% please be careful
% all the data must be [number_of_samples, feature/label]
train_p_target = train_p_target';
test_target = test_target';
train_target=train_target';

[ accuracy_test] = SAFE(train_data, train_p_target, ...
    test_data, test_target,train_target, k,  Maxiter, ...
    gamma, lambda, alpha, beta);
