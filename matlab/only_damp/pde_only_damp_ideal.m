%------- PDE coefficients matching Python implementation -------%

T = 1;    % Tension
mu = 1;   % Mass density
k = 1;    % Damping coefficient

t_f = 10;  % Final time scaling
u_min = -0.21;
u_max = 0.0;
x_min = 0.0;
x_max = 1.0;
y_min = 0.0;
y_max = 1.0;
f_min = -3.0;
f_max = 0.0;

delta_u = u_max - u_min; % 0.21
delta_x = x_max - x_min; % 1
delta_y = y_max - y_min; % 1
delta_f = f_max - f_min; % 3

alpha_2 = (T/mu) * (t_f^2) / (delta_x^2);  % 100
beta_2 = (T/mu) * (t_f^2) / (delta_y^2);   % 100
gamma = (t_f^2) / delta_u;  % 4761.9

numberOfPDE = 1;
model = createpde(numberOfPDE);

% Specify the geometry

S = [3, 4, x_min, x_max, x_max, x_min, y_max, y_max, y_min, y_min]';
g = decsg(S);
geometryFromEdges(model,g);

% Plot geometry
pdegplot(model,"EdgeLabels","on");
title("Square Membrane Geometry With Edge Labels")
xlabel("x")
ylabel("y")

% First specify the coefficients for the undamped model,
% otherwise Matlab raises an error

specifyCoefficients(model, ...
    "m", 1, ... % Inertia term
    "d", 0, ... % Damping term
    "c", alpha_2, ... % Wave propagation (laplacian coefficient)
    "a", 0, ... % No reaction term
    "f", @(location,state) gamma * externalForce(location, state) ...
    );

% Apply zero Dirichlet boundary conditions on all edges (fixed boundaries)

applyBoundaryCondition(model,"dirichlet","Edge",[1,2,3,4],"u",-u_min/delta_u);

% Generate mesh: value chosen to avoid Matlab crash

mesh = generateMesh(model, 'Hmax', 0.05);
nodes = mesh.Nodes.';
save("mesh_nodes.mat", "nodes", "-mat");

% Set initial conditions (zero displacement and velocity as in Python)

setInitialConditions(model,-u_min/delta_u,0);

% Compute damping coefficient according to
% https://it.mathworks.com/help/pde/ug/pde.pdemodel.specifycoefficients.html#mw_90c0a86d-26a6-4abf-9c79-c89129275bf2
% Setting it manually raises an error

fem = assembleFEMatrices(model);
d = 1*t_f*fem.M + 0*fem.K;

% Specify coefficients for the final damped model

specifyCoefficients(model, ...
    "m", 1, ... % Inertia term
    "d", d, ... % Damping term
    "c", alpha_2, ... % Wave propagation (laplacian coefficient)
    "a", 0, ... % No reaction term
    "f", @(location,state) gamma * externalForce(location, state) ...
    );


% Time settings matching Python
tF = 1;
n = 100;  % number of time steps
tlist = linspace(0,tF,n);

% File to save/load results
% To avoid solving the pde (computationally expensive)
% every time if conditions don't change

resultsFile = 'pde_ideal_with_damping.mat';

if isfile(resultsFile)
    disp('Loading results from file...');
    load(resultsFile, 'result', 'tlist', 'u');
else
    % Solve PDE and save results
    model.SolverOptions.ReportStatistics = 'on';
    disp('Solving PDE with damping...');
    result = solvepde(model, tlist);
    u = result.NodalSolution;
    u = (u*delta_u)+u_min;

    % Save results to file
    save(resultsFile, 'result', 'tlist', 'u');
    disp('Results saved to file.');
end

% Create animation
figure
umax = max(max(u));
umin = min(min(u));

% Animation gif
gifFilename = 'membrane_ideal_with_damping.gif';
frameDelay = 0.1;

t_line = linspace(0, 10, n);

disp("Starting for loop")
for i = 1:n
    pdeplot(model,"XYData",u(:,i),"ZData",u(:,i), ...
        "ZStyle","continuous","Mesh","off");
    zlim([umin umax]);
    xlabel('x')
    ylabel('y')
    zlabel('u')
    title(sprintf('Time: %.2f s', t_line(i)))
    colorbar

    % Capture the current frame
    drawnow;
    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    
    % Write to the GIF file
    if i == 1
        imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', frameDelay);
    else
        imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', frameDelay);
    end

    M(i) = frame;

end

% External force function (Gaussian as in Python)
function f = externalForce(location,~)
    x = location.x;
    y = location.y;
    x_f = 0.2;
    y_f = 0.2;
    h = -3.0; % Minimum force value from Python
    f = h * exp(-400 * ((x - x_f).^2 + (y - y_f).^2));
end