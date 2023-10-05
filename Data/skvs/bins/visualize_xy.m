clear
close all

pose_data = readtable("pose_data.csv");

hip = [pose_data.Hip_Center_RawX, pose_data.Hip_Center_RawY];
spine = [pose_data.Spine_RawX, pose_data.Spine_RawY];
shoulder_center = [pose_data.Shoulder_Center_RawX, pose_data.Shoulder_Center_RawY];
head = [pose_data.Head_RawX, pose_data.Head_RawY];
shoulder_right = [pose_data.Shoulder_Right_RawX, pose_data.Shoulder_Right_RawY];
elbow_right = [pose_data.Elbow_Right_RawX, pose_data.Elbow_Right_RawY];
wrist_right = [pose_data.Wrist_Right_RawX, pose_data.Wrist_Right_RawY];
hand_right = [pose_data.Hand_Right_RawX, pose_data.Hand_Right_RawY];
shoulder_left = [pose_data.Shoulder_Left_RawX, pose_data.Shoulder_Left_RawY];
elbow_left = [pose_data.Elbow_Left_RawX, pose_data.Elbow_Left_RawY];
wrist_left = [pose_data.Wrist_Left_RawX, pose_data.Wrist_Left_RawY];
hand_left = [pose_data.Hand_Left_RawX, pose_data.Hand_Left_RawY];
hip_right = [pose_data.Hip_Right_RawX, pose_data.Hip_Right_RawY];
knee_right = [pose_data.Knee_Right_RawX, pose_data.Knee_Right_RawY];
ankle_right = [pose_data.Ankle_Right_RawX, pose_data.Ankle_Right_RawY];
foot_right = [pose_data.Foot_Right_RawX, pose_data.Foot_Right_RawY];
hip_left = [pose_data.Hip_Left_RawX, pose_data.Hip_Left_RawY];
knee_left = [pose_data.Knee_Left_RawX, pose_data.Knee_Left_RawY];
ankle_left = [pose_data.Ankle_Left_RawX, pose_data.Ankle_Left_RawY];
foot_left = [pose_data.Foot_Left_RawX, pose_data.Foot_Left_RawY];



for i = 1:length(hip)
    hold on
    scatter(hip(i, 1), hip(i, 2))
    scatter(spine(i, 1), spine(i, 2))
    scatter(shoulder_center(i, 1), shoulder_center(i, 2))
    scatter(head(i, 1), head(i, 2))
    scatter(shoulder_right(i, 1), shoulder_right(i, 2))
    scatter(elbow_right(i, 1), elbow_right(i, 2))
    scatter(wrist_right(i, 1), wrist_right(i, 2))
    scatter(hand_right(i, 1), hand_right(i, 2))
    scatter(shoulder_left(i, 1), shoulder_left(i, 2))
    scatter(elbow_left(i, 1), elbow_left(i, 2))
    scatter(wrist_left(i, 1), wrist_left(i, 2))
    scatter(hand_left(i, 1), hand_left(i, 2))
    scatter(hip_right(i, 1), hip_right(i, 2))
    scatter(knee_right(i, 1), knee_right(i, 2))
    scatter(ankle_right(i, 1), ankle_right(i, 2))
    scatter(foot_right(i, 1), foot_right(i, 2))
    scatter(hip_left(i, 1), hip_left(i, 2))
    scatter(knee_left(i, 1), knee_left(i, 2))
    scatter(ankle_left(i, 1), ankle_left(i, 2))
    scatter(foot_left(i, 1), foot_left(i, 2))
    hold off
    
    xlim([0 640])
    ylim([0 480])
    drawnow
end