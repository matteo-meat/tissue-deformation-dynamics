%% --- Ideal membrane for error evaluation ---

T = 1;
mu = 1;
k = 1;

t_f = 10;
u_min = -0.21;
u_max = 0.0;
x_min = 0.0;
x_max = 1.0;
y_min = 0.0;
y_max = 1.0;
f_min = -3.0;
f_max = 0.0;

delta_u = u_max - u_min;
delta_x = x_max - x_min;
delta_y = y_max - y_min;
delta_f = f_max - f_min;

alpha_2 = (T/mu) * (t_f^2) / (delta_x^2);
beta_2 = (T/mu) * (t_f^2) / (delta_y^2);
gamma = (t_f^2) / delta_u;

numberOfPDE = 1;
model = createpde(numberOfPDE);

S = [3, 4, x_min, x_max, x_max, x_min, y_max, y_max, y_min, y_min]';
g = decsg(S);
geometryFromEdges(model,g);

pdegplot(model,"EdgeLabels","on");
title("Square Membrane Geometry With Edge Labels")
xlabel("x")
ylabel("y")

specifyCoefficients(model, ...
    "m", 1, ...
    "d", 0, ...
    "c", alpha_2, ...
    "a", 0, ...
    "f", @(location,state) gamma * externalForce(location, state) ...
    );

applyBoundaryCondition(model,"dirichlet","Edge",[1,2,3,4],"u",0);

mesh = generateMesh(model, 'Hmax', 0.05);
nodes = mesh.Nodes.';
save("mesh_nodes.mat", "nodes", "-mat");

setInitialConditions(model,0,0);

fem = assembleFEMatrices(model);
d = 1*t_f*fem.M + 0*fem.K;

specifyCoefficients(model, ...
    "m", 1, ...
    "d", d, ...
    "c", alpha_2, ...
    "a", 0, ...
    "f", @(location,state) gamma * externalForce(location, state) ...
    );

tF = 1;
n = 100;
tlist = linspace(0,tF,n);

resultsFile = 'pde_ideal_with_damping.mat';

if isfile(resultsFile)
    disp('Loading results from file...');
    load(resultsFile, 'result', 'tlist', 'u_i');
else
    model.SolverOptions.ReportStatistics = 'on';
    disp('Solving PDE with damping...');
    result = solvepde(model, tlist);
    u_i = result.NodalSolution;

    save(resultsFile, 'result', 'tlist', 'u_i');
    disp('Results saved to file.');
end

%% -----------------------------------

%% --- Evaluation of models ---

baseInputFolder  = 'predictions_csv_files';

modelTypes = {'MLP', 'KAN', 'RWF'};
numInstances = 6; 

%% --- Loop Over Each Model Folder ---
for m = 1:length(modelTypes)
    modelType = modelTypes{m};
    for n = 1:numInstances

        % Construct folder name (e.g., 'MLP_1') and determine input/output folders.
        folderName = sprintf('%s_%d', modelType, n);
        inputFolder = fullfile(baseInputFolder, folderName);

        if ~exist(inputFolder, 'dir')
            fprintf('Input folder %s not found. Skipping...\n', inputFolder);
            continue;
        end

        predictionsFile = fullfile(inputFolder, 'predictions.csv');
        if ~exist(predictionsFile, 'file')
            fprintf('File predictions.csv not found in folder %s. Skipping...\n', inputFolder);
            continue;
        end

        outputFolder = folderName;
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end

        fprintf('Processing folder: %s\n', folderName);

        %% --- Load and Save Prediction Data ---
        csv = readtable(predictionsFile);
        u = csv{:,:};
        save(fullfile(outputFolder, 'formatted_predictions.mat'), 'u');
        u = load(fullfile(outputFolder, 'formatted_predictions.mat'));
        u = u.u;

        t = size(u,2);
        tlist = linspace(0, 1, size(u,2));
        
        mesh_full = load("mesh_full.mat");
        mesh_full = mesh_full.mesh;
        
        figure;
        umax = max(max(u));
        umin = min(min(u));
        gifFilename = 'membrane_pred_damp.gif';
        gifFilename = fullfile(outputFolder, gifFilename);
        frameDelay = 0.1;

        for i=1:t
            pdeplot(mesh_full, "XYData",u(:,i),"ZData",u(:,i), ...
                "ZStyle","continuous","Mesh","off");
            
            zlim([umin umax]);
        
            xlabel('x')
            ylabel('y')
            zlabel('u')
            title(sprintf('Time: %.2f s', tlist(i)))
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
        
        close;

        %% --- Compute Simulation Metrics ---
        % Calculate the maximum displacement at each time step
        maxDisplacement = max(abs(u), [], 1);

        % Calculate the L2 norm of the displacement at each time step
        l2Norm = sqrt(sum(u.^2, 1));

        %% --- Plot Maximum Displacement Over Time ---
        figure;
        plot(tlist, maxDisplacement, 'r', 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('Maximum Displacement');
        title('Maximum Displacement Over Time');
        grid on;
        savefig(fullfile(outputFolder, 'displ_damp.fig'));
        close;  % Close the figure to avoid clutter

        %% --- Plot L2 Norm Over Time ---
        figure;
        plot(tlist, l2Norm, 'b', 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('L2 Norm of Displacement');
        title('L2 Norm of Displacement Over Time');
        grid on;
        savefig(fullfile(outputFolder, 'l2_damp.fig'));
        close;

        %% --- Error Membrane Simulation ---

        figure;
        umax = max(max(u_i - u));
        umin = min(min(u_i - u));
        gifFilename = 'error_membrane_pred_damp.gif';
        gifFilename = fullfile(outputFolder, gifFilename);
        frameDelay = 0.1;

        for i=1:t
            pdeplot(model, "XYData",u_i(:,i) - u(:,i),"ZData",u_i(:,i) - u(:,i), ...
                "ZStyle","continuous","Mesh","off");
            
            zlim([umin umax]);
        
            xlabel('x')
            ylabel('y')
            zlabel('u_i - u')
            title(sprintf('Time: %.2f s', tlist(i)))
            colorbar
        
            drawnow;
            frame = getframe(gcf);
            img = frame2im(frame);
            [imind, cm] = rgb2ind(img, 256);
            
            if i == 1
                imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', frameDelay);
            else
                imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', frameDelay);
            end
        
            M(i) = frame;
        
        end
        
        close;

        %% --- Plot Error difference ---
        diff_e = u_i - u;

        figure;
        plot(tlist, diff_e, 'r', 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('Error difference');
        title('Error difference Over Time');
        grid on;
        savefig(fullfile(outputFolder, 'error_difference_damp.fig'));
        close;

        fprintf('Finished processing folder: %s\n', folderName);
    end
end

fprintf('All predictions have been processed.\n');

function f = externalForce(location,~)
    x = location.x;
    y = location.y;
    x_f = 0.2;
    y_f = 0.2;
    h = -3.0;
    f = h * exp(-400 * ((x - x_f).^2 + (y - y_f).^2));
end