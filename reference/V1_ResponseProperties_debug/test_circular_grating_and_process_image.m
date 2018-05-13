% relevant parts from V1_orientation_tuning_contrast.m
clear all;
close all;

crop=0;
grating_wavel=6;
grating_angles=-22.5:7.5:22.5;
%grating_angles=[-50:12.5:50];
contrasts=[0.05,0.2,0.8];

iterations=12;
patch_diam=15;
phase=0;

node=1;

[wFFon,wFFoff]=dim_conv_V1_filter_definitions;

j=0;
Ion_all = cell(numel(contrasts), numel(grating_angles));
Ioff_all = cell(numel(contrasts), numel(grating_angles));
I_all = cell(numel(contrasts), numel(grating_angles));
for contrast=contrasts
  j=j+1;
  i=0;
  for ga=grating_angles
	i=i+1;
	fprintf(1,'.%i.\n',i); 
	I=image_circular_grating(patch_diam,20,grating_wavel,ga,phase,contrast); 
	[Ion,Ioff]=preprocess_image(I);
    I_all{j,i} = I;
    Ion_all{j,i} = Ion;
    Ioff_all{j,i} = Ioff;
  end
end

save('test_circular_grating_and_process_image.mat', 'I_all', 'Ion_all', ...
    'Ioff_all');
