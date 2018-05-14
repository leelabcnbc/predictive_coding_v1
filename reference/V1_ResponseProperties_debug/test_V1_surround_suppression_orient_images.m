function test_V1_surround_suppression_orient_images()
crop=0;
grating_wavel=6;
phase=0;
node=1;
levitt=0

if levitt
    %for Levitt and Lund experiment
    contrast_hi=0.75;
    contrast_lo=0.25;
    tests=4;
    context_angles=[-90:30:90];
    patch_diam=15;iterations=20;
else
    %for Jones and Sillito experiment
    contrast_hi=0.5;
    tests=2;
    context_angles=[-90:22.5:90];
    patch_diam=7;iterations=12;%gives orientation contrast suppression
    %patch_diam=11;iterations=8;%gives non-orientation specific suppression
    %patch_diam=13;iterations=12;%gives mixed general suppression
    %patch_diam=17;iterations=16;%gives orientation alignment suppression
    %patch_diam=19;iterations=16;%gives orientation contrast facilitation
    %   patch_diam=15;iterations=20;
end
maxdiam=24;

I_all = cell(2, numel(context_angles));

for test=1:tests
    i=0;
    clear Ion Ioff;
    for ca=context_angles
        i=i+1;
        fprintf(1,'.%i.',i);
        if test==1 %contextual surround
            Ig=image_contextual_surround(patch_diam,0,patch_diam,grating_wavel,...
                grating_wavel,ca,phase,phase,...
                contrast_hi,contrast_hi);
        elseif test==2 %orientation tuning
            Ig=image_circular_grating(patch_diam,patch_diam,grating_wavel,ca,phase,...
                contrast_hi);
        end
        I=zeros(3*maxdiam)+0.5;
        [a,b]=size(I);
        [ag,bg]=size(Ig);
        I(ceil(a/2)-floor(ag/2):ceil(a/2)+floor(ag/2),ceil(b/2)-floor(bg/2):ceil(b/2)+floor(bg/2))=Ig;
        
        I_all{test, i} = I;
    end
end

save('test_V1_surround_suppression_orient_images.mat', 'I_all');
