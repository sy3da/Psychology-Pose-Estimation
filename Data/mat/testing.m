clear
close all

load("thanos_processed_2023-10-10_12-59-47.mat");
max_val = max(max(max(I_values)));

for i=1:size(I_values, 3)
    frame = I_values(:,:,i);
    frame_image = frame*255/max_val;
    imshow(frame_image)
end

