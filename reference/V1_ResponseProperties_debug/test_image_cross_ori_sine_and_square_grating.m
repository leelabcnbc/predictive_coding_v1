function test_image_cross_ori_sine_and_square_grating
% adapted from V1_cross_orient_orient.m

grating_wavel=6;
patch_diam=51;

contrast=0.5;
phase=0;

grating_angles=[-60:20:60];

I_all = cell(2, numel(grating_angles));

for test=1:2
  i=0;
  for angle=grating_angles
	i=i+1;
	fprintf(1,'.%i.',i); 
	if test==1 %cross-orientation simulus
	  I=image_cross_orientation_sine(patch_diam,grating_wavel,grating_wavel,...
									 0,angle,phase,phase,contrast,contrast);
	else %orientation tuning stimulus
	  I=image_square_grating(patch_diam,0,grating_wavel,angle, ...
							 phase,contrast*2); 
	end
	I_all{test, i} = I;
  end
end


save('test_image_cross_ori_sine_and_square_grating.mat',...
    'I_all');


