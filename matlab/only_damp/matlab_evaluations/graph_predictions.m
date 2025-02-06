tF = 1;
t = 100;  % number of time steps
tlist = linspace(0,tF,t);

modelTypes = {'MLP', 'KAN', 'RWF'};
numInstances = 6; 

for m = 1:length(modelTypes)
    modelType = modelTypes{m};
    for n = 1:numInstances

        folderName = sprintf('%s_%d', modelType, n);

        fprintf('Processing folder: %s\n', folderName);

        u = load(fullfile(folderName, "formatted_predictions.mat"));
        u = u.u;
        mesh_full = load("mesh_full.mat");
        mesh_full = mesh_full.mesh;

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
        savefig(fullfile(folderName, 'pred_damp.fig'));
        
        % Plot the L2 norm over time
        figure;
        plot(tlist, l2Norm, 'b', 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('L2 Norm of Displacement');
        title('L2 Norm of Displacement Over Time');
        grid on;
        savefig(fullfile(folderName,'l2_pred_damp.fig'));

    end
end