tF = 1;
n = 100;  % number of time steps
tlist = linspace(0,tF,n);

meshPoints = mesh';  % N x 2 matrix of x,y coordinates
meshPoints_ext = [meshPoints zeros(size(meshPoints, 1), 1)];
meshPoints_copy = meshPoints_ext;

for i = 2:length(tlist)
    meshPoints_copy(:, 3) = tlist(i);
    meshPoints_ext = [meshPoints_ext; meshPoints_copy];
end