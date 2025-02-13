u_min = -0.21;
u_max = 0.0;
x_min = 0.0;
x_max = 1.0;
y_min = 0.0;
y_max = 1.0;
f_min = -3.0;
f_max = 0.0;

u_denorm = (u_eval * (u_max - u_min)) + u_min;

t = size(u_denorm,2);
t_line = linspace(0, 10, t);

mesh_full = load("mesh_full.mat");
mesh_full = mesh_full.mesh;

figure;
umax = max(max(u_denorm));
umin = min(min(u_denorm));
frameDelay = 0.1;

for i=1:t
    pdeplot(mesh_full, "XYData",u_denorm(:,i),"ZData",u_denorm(:,i), ...
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

    M(i) = frame;

end

% close;

idealResults = 'ideal_solution.mat';
load(idealResults);
u_i = u;

figure;
umax = max(max(u_i));
umin = min(min(u_i));
frameDelay = 0.1;

for i=1:t
    pdeplot(mesh_full, "XYData",u_i(:,i),"ZData",u_i(:,i), ...
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

    M_i(i) = frame;

end

figure;
umax = max(max(u_i - u_denorm));
umin = min(min(u_i - u_denorm));
frameDelay = 0.1;

for i=1:t
    pdeplot(mesh_full, "XYData",u_i(:,i) - u_denorm(:,i),"ZData",u_i(:,i) - u_denorm(:,i), ...
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

    M_diff(i) = frame;

end

% close;

diff_e = u_i - u_denorm;

figure;
plot(tlist, diff_e, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Error');
title('Error Over Time');
grid on;

l2_error = sqrt(sum(diff_e.^2));

plot(tlist, l2_error, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('L2 Norm');
title('L2 Norm of Error');
grid on;
