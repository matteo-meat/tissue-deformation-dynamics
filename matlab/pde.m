% PDE coefficients matching Python implementation
T = 1;    % Tension
mu = 1;   % Mass density
k = 1;    % Damping coefficient

numberOfPDE = 1;
model = createpde(numberOfPDE);
geometryFromEdges(model,@squareg);

% % Plot geometry
% pdegplot(model,"EdgeLabels","on");
% ylim([-.1 1.1]);
% xlim([-.1 1.1]);
% axis equal
% title("Square Membrane Geometry With Edge Labels")
% xlabel("x")
% ylabel("y")

% Specify PDE coefficients for wave equation
% mu*∂²u/∂t² + k*∂u/∂t - T*∇²u = f
specifyCoefficients(model,"m",mu,"d",0,"c",T,"a",k,"f",@externalForce);

% Apply zero Dirichlet boundary conditions on all edges (fixed boundaries)
applyBoundaryCondition(model,"dirichlet","Edge",[1,2,3,4],"u",0);

% Generate mesh
mesh = generateMesh(model,'Hmax',0.02);
figure
pdemesh(model);
ylim([-.1 1.1]);
xlim([-.1 1.1]);
axis equal
xlabel('x')
ylabel('y')
title('Mesh Visualization')

% Set initial conditions (zero displacement and velocity as in Python)
setInitialConditions(model,0,0);

% Time settings matching Python
tF = 10;
n = 100;  % number of time steps
tlist = linspace(0,tF,n);

% File to save/load results
resultsFile = 'pde_results.mat';

if isfile(resultsFile)
    % Load results from file
    disp('Loading results from file...');
    load(resultsFile, 'result', 'tlist', 'u');
else
    % Solve PDE and save results
    model.SolverOptions.ReportStatistics = 'on';
    disp('Solving PDE...');
    result = solvepde(model, tlist);
    u = result.NodalSolution;

    % Save results to file
    save(resultsFile, 'result', 'tlist', 'u');
    disp('Results saved to file.');
end

% Create animation
figure
umax = max(max(u));
umin = min(min(u));
disp("Starting for loop")
for i = 1:n
    pdeplot(model,"XYData",u(:,i),"ZData",u(:,i), ...
        "ZStyle","continuous","Mesh","off");
    axis([0 1 0 1 umin umax]);  % Set axis to match domain [0,1]x[0,1]
    xlabel('x')
    ylabel('y')
    zlabel('u')
    title(sprintf('Time: %.2f s', tlist(i)))
    colorbar
    M(i) = getframe;
end

% Play animation
movie(M)

% Calculate the maximum displacement at each time step
maxDisplacement = max(abs(u), [], 1);

% Calculate the L2 norm of the displacement at each time step
l2Norm = sqrt(sum(u.^2, 1));

% Plot the maximum displacement over time
figure;
plot(tlist, maxDisplacement, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Maximum Displacement');
title('Maximum Displacement Over Time');
grid on;

% Plot the L2 norm over time
figure;
plot(tlist, l2Norm, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('L2 Norm of Displacement');
title('L2 Norm of Displacement Over Time');
grid on;

% External force function (Gaussian as in Python)
function f = externalForce(location,~)
    x = location.x;
    y = location.y;
    x_f = 0.2;
    y_f = 0.2;
    h = -3.0; % Minimum force value from Python
    f = h * exp(-400 * ((x - x_f).^2 + (y - y_f).^2));
end

% Optional: Save animation
% v = VideoWriter('membrane_oscillation.avi');
% open(v);
% writeVideo(v,M);
% close(v);
