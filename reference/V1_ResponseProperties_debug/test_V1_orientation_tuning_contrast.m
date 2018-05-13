function test_V1_orientation_tuning_contrast
crop=0;
grating_wavel=6;

grating_angles=[-22.5:7.5:22.5];
%grating_angles=[-50:12.5:50];
contrasts=[0.05,0.2,0.8];

iterations=12;
patch_diam=15;
phase=0;

node=1;

[wFFon,wFFoff]=dim_conv_V1_filter_definitions;

clf
j=0;
Y_init_all = cell(numel(contrasts), numel(grating_angles));
Y_full_all = cell(numel(contrasts), numel(grating_angles));
for contrast=contrasts
  j=j+1;
  i=0;
  for ga=grating_angles
	i=i+1;
	fprintf(1,'.%i.',i); 
	I=image_circular_grating(patch_diam,20,grating_wavel,ga,phase,contrast); 
	[Ion,Ioff]=preprocess_image(I);
	[a,b]=size(I);

	%plot original image
	maxsubplot(3,length(grating_angles),i),
	imagesc(I(:,:,1),[0,1]);
	axis('equal','tight'), set(gca,'XTick',[],'YTick',[],'FontSize',11);
	drawnow;
	
	%initial response without competition
	[y,ron,roff,eon,eoff,Y]=dim_conv_on_and_off(wFFon,wFFoff,Ion,Ioff,1);
	y=mean(Y,4);
	maxsubplot(3,length(grating_angles),i+length(grating_angles)),
	imagesc(y(crop+1:a-crop,crop+1:b-crop,node),[0,0.01]), 
	axis('equal','tight'), set(gca,'XTick',[],'YTick',[],'FontSize',11);
	drawnow;
    Y_init_all{j, i} = Y;
	
	%perform competition
	[y,ron,roff,eon,eoff,Y]=dim_conv_on_and_off(wFFon,wFFoff,Ion,Ioff,iterations);
	y=mean(Y,4);
	maxsubplot(3,length(grating_angles),i+2*length(grating_angles)),
	imagesc(y(crop+1:a-crop,crop+1:b-crop,node),[0,0.01]), 
	axis('equal','tight'), set(gca,'XTick',[],'YTick',[],'FontSize',11);
	drawnow;
    
    Y_full_all{j, i} = Y;
	sc(j,i)=y(ceil(a/2),ceil(b/2),node);
  end
end

% save sc
save('test_V1_orientation_tuning_contrast.mat', ...
    'sc', 'Y_full_all', 'Y_init_all');

