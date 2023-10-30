clear
close all

load("thanos_processed_one_person_1_WETLAB.mat");
max_val = max(max(max(I_values)));
bright_scale = 4;

for i=1:size(I_values, 3)
    frame_image = I_values(:,:,i)*bright_scale;
    frame_image(frame_image > 255) = 255;
    imshow(frame_image,[0 255])
end

