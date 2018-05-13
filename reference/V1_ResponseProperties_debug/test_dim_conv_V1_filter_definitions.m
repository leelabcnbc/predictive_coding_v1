% generate those gabor filters from dim_conv_V1_filter_definitions
clear all;
close all;
[wFFon,wFFoff]=dim_conv_V1_filter_definitions();
save('test_dim_conv_V1_filter_definitions.mat', 'wFFon', 'wFFoff');
