clear
close all

load("thanos_processed_2023-10-10_14-17-18.mat");
max_val = max(max(max(I_values)));
bright_scale = 

for i=1:size(I_values, 3)
    frame_image = I_values(:,:,i)*bright_scale;
    frame_image(frame_image > 255) = 255;
    imshow(frame_image)
end

