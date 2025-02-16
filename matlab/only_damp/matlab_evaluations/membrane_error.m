tF = 10;
n = 100;
t_line = linspace(0,tF,n);

mesh_full = load("mesh_full.mat");
mesh_full = mesh_full.mesh;   

idealResults = 'ideal_solution.mat';

if isfile(idealResults)
    disp('Loading results from file...');
    load(idealResults);
    u_i = u;
else
    disp("Error loading ideal results!")
end

outputFolder = "KAN_8";
u_eval = load(fullfile(outputFolder, 'formatted_predictions.mat'));
u_eval = u_eval.u;  

figure;
umax = max(max(u_i - u_eval));
umin = min(min(u_i - u_eval));
gifFilename = 'error_membrane_pred_damp.gif';
gifFilename = fullfile(outputFolder, gifFilename);
frameDelay = 0.1;

for i=1:n
    pdeplot(mesh_full, "XYData",u_i(:,i) - u_eval(:,i),"ZData",u_i(:,i) - u_eval(:,i), ...
        "ZStyle","continuous","Mesh","off");

    zlim([umin umax]);

    xlabel('x')
    ylabel('y')
    zlabel('u_i - u')   
    title(sprintf('Time: %.2f s', t_line(i)))
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

% --- Plot Error difference ---
diff_e = u_i - u_eval;

figure;
plot(t_line, diff_e, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Error');
title('Error Over Time');
grid on;
savefig(fullfile(outputFolder, 'error_plot_damp.fig'));
close;

l2_error = sqrt(sum(diff_e.^2));

figure;
plot(t_line, l2_error, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('L2 Norm');
title('L2 Norm of Error');
grid on;
savefig(fullfile(outputFolder, 'l2_plot_damp.fig'));
close;