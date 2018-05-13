clear all;
close all;

grating_wavel=6;

diams=[3:4:31];

iterations=8;
contrast=0.25;
phase=0;

node=1;
I_all = cell(2, numel(diams));
for test=1:2
    i=0;
    for diam=diams
        i=i+1;
        if test==1 %a grating
            I=image_contextual_surround(diam,max(diams)-diam/2,0,grating_wavel,grating_wavel,0,phase,0,contrast,0);
        else %an annulus
            assert(test==2);
            I=image_contextual_surround(0,diam/2,max(diams)-diam/2,grating_wavel,grating_wavel,0,0,phase,0,contrast);
        end
        I_all{test, i} = I;
    end
end

save('test_image_contextual_surround.mat', 'I_all');