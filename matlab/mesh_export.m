mesh = load("mesh_nodes.mat");
nodes = mesh.mesh_nodes.';

nodes_ext = [nodes zeros(size(nodes,1), 1)];

% Time settings matching Python
tF = 10;
n = 100;  % number of time steps
tlist = linspace(0,tF,n);

index = 1;

for i=1:n
    for j=index:index+n-1
        nodes(j, 3) = tlist(i);
        disp(j)
    end
    index = index + n;
end