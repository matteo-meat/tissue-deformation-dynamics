baseInputFolder  = 'predictions_csv_files';

% Construct folder name (e.g., 'MLP_1') and determine input/output folders.
folderName = "KAN_7";
inputFolder = fullfile(baseInputFolder, folderName);

if ~exist(inputFolder, 'dir')
    fprintf('Input folder %s not found. Skipping...\n', inputFolder);
    return;
end

predictionsFile = fullfile(inputFolder, 'predictions.csv');
if ~exist(predictionsFile, 'file')
    fprintf('File predictions.csv not found in folder %s. Skipping...\n', inputFolder);
    return;
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
t_line = linspace(0, 10, t);

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

close;

fprintf('Finished processing folder: %s\n', folderName);

fprintf('All predictions have been processed.\n');
