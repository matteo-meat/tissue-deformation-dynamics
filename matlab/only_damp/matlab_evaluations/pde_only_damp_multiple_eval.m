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

        %% --- Plot Error difference Over Time ---
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
