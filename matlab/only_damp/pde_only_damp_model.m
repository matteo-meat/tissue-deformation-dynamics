csv = readtable('formatted_predictions.csv');
u = csv{:,:}
save('formatted_predictions.mat', 'u')

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
savefig('displ_damp.fig')

% Plot the L2 norm over time
figure;
plot(tlist, l2Norm, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('L2 Norm of Displacement');
title('L2 Norm of Displacement Over Time');
grid on;
savefig('l2_damp.fig')
