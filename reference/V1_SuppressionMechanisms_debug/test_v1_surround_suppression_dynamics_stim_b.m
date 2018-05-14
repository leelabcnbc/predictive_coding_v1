function test_v1_surround_suppression_dynamics_stim_b

add_v1_path();

grating_wavel=6;
patch_diam=11;
gap=2;
image_size=3*patch_diam+2*gap;
contrast=1;

I_all = cell(2,2);

for test=1:2
    phase=0;
    for t=1:2
        if t == 1
            I=image_centre_surround(patch_diam,gap,0.5*(image_size-patch_diam),...
                grating_wavel,grating_wavel,90,90,phase,phase,...
                contrast,contrast);
        elseif t==2
            if test==1
                I=image_centre_surround(patch_diam,gap,0.5*(image_size-patch_diam),...
                    grating_wavel,grating_wavel,90,90,phase,phase,...
                    contrast,contrast);
            else
                I=image_centre_surround(patch_diam,gap,0.5*(image_size-patch_diam),...
                    grating_wavel,grating_wavel,0,90,phase,phase,...
                    contrast,contrast);
            end
        end
        
        
        I_all{test, t} = I;
        
    end
end

save('test_v1_surround_suppression_dynamics_stim_b.mat', 'I_all');