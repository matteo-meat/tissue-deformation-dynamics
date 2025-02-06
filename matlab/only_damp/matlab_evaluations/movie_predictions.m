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

        figure
        umax = max(max(u));
        umin = min(min(u));
        gifFilename = 'membrane_pred_damp.gif';
        gifFilename = fullfile(folderName, gifFilename);
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
    end
end