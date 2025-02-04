% Time settings matching Python
tF = 1;
n = 100;  % number of time steps
tlist = linspace(0,tF,n);

load("./output_files/mesh_nodes.mat")
meshPoints = nodes;  % N x 2 matrix of x,y coordinates
meshPoints_ext = [meshPoints zeros(size(meshPoints, 1), 1)];
meshPoints_copy = meshPoints_ext;

for i = 2:length(tlist)
    meshPoints_copy(:, 3) = tlist(i);
    meshPoints_ext = [meshPoints_ext; meshPoints_copy];
end

% CSV generation
headers = {'x', 'y', 't'};
spaceTimeTable = array2table(meshPoints_ext, 'VariableNames', headers);
% Save to CSV
csvFileName = './output_files/space_time_points.csv';
writetable(spaceTimeTable, csvFileName);

% Display confirmation message
fprintf('Space-time points saved to %s\n', csvFileName);