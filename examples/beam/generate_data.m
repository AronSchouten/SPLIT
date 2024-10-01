% Generate data of cantilever beam
% Requires YetAnotherFEcode
clear;
close all;
clc

% Order of finite difference? (1-4)
order = 3;
% Degree of freedom of tip
outdof = 122;
% Create plot of data?
create_plot = true;
% Initial displacement number (1-5)
nr = '3';

%% Prepare model
% Properties of Beam
E       = 104e9;   % [Pa] Young's modulus
rho     = 440;     % [kg/m^3] density
nu      = 0.3;     % [ ] Poissons ratio

l = 1;             % [m] beam's length
h = .02;           % [m] beam's height
b = .05;           % [m] beam's out-of-plane thickness

% Setup material
myMaterial = KirchoffMaterial();
set(myMaterial,'YOUNGS_MODULUS',E,'DENSITY',rho,'POISSONS_RATIO',nu);
myMaterial.PLANE_STRESS = true;	% set "false" for plane_strain

% Setup element
myElementConstructor = @()Quad4Element(b, myMaterial);

% Construct mesh
nx = 40;
ny = 2;

elementType = 'QUAD4';
[nodes, elements, nset] = mesh_2Drectangle(l,h,nx,ny,elementType);
myMesh = Mesh(nodes);
myMesh.create_elements_table(elements,myElementConstructor);

myMesh.set_essential_boundary_condition([nset{1}],1:2,0)

% Assembly
BeamAssembly = Assembly(myMesh);
M = BeamAssembly.mass_matrix();
u0 = zeros(myMesh.nDOFs, 1);
[K,~] = BeamAssembly.tangent_stiffness_and_force(u0);

Mc = BeamAssembly.constrain_matrix(M);
Kc = BeamAssembly.constrain_matrix(K);

omega1_sq = eigs(Kc,Mc,1,'SM');
C = sqrt(omega1_sq) / 50 * M;

BeamAssembly.DATA.K = K;
BeamAssembly.DATA.M = M;
BeamAssembly.DATA.C = C;

%% Modal analysis
Cc = BeamAssembly.constrain_matrix(C);

[phi1,D] = eigs(Kc,Mc,1,'SM');
mu = phi1' * Mc * phi1; % Mass normalize
phi1 = phi1/sqrt(mu);
omega1 = sqrt(D);

lambda = zeros(2);

zeta = 1/(2*omega1) * ( phi1' * Cc * phi1 )/( phi1' * Mc * phi1 );
lambda(1) = (-zeta-sqrt(zeta.^2-1))*omega1;
lambda(2) = (-zeta+sqrt(zeta.^2-1))*omega1;   

eigfreq = imag(lambda(2));
damprat = real(lambda(2));

%% Generate trajectory
% Define variables
T = 2*pi/eigfreq; % time period of forcing
h = T/1000; %5e-4;
tmax = 30*T;
nDofs = myMesh.nDOFs;

% No external forcing
F_ext = @(t) zeros(nDofs, 1);

% Load in initial displacement
name = strcat('data\initial_conditions\q0_',nr,'.mat');
load(name)
qd0 = BeamAssembly.constrain_vector(zeros(nDofs, 1));
qdd0 = BeamAssembly.constrain_vector(zeros(nDofs, 1));

residual = @(q,qd,qdd,t) residual_nonlinear(q,qd,qdd,t,BeamAssembly,F_ext);
TI = ImplicitNewmark('timestep',h,'alpha',0.005,'linear',false);

% Perform and save time integration
TI.Integrate(q0,qd0,qdd0,tmax,residual);

time = TI.Solution.time;
q = TI.Solution.q;
qd = TI.Solution.qd;

%% Calculate and save values
% Linear coefficients
m = phi1' * Mc * phi1;
c = phi1' * Cc * phi1;
k = phi1' * Kc * phi1;

% Copute x and its derivatives
[ddx,dx,x,t] = finiteDifference(q,time,order);

% % Can also be done using SSMLearn's finiteTimeDifference()
% [~,x,t] = finiteTimeDifference(q,time,1);
% [ddx,dx,~] = finiteTimeDifference(qd,time,1);

name = strcat('data\trajectories\non_linear_',nr,'.mat');
save(name,'t','x','dx','ddx','m','c','k','phi1','Mc')

%% Plotting the data
if create_plot
    % Initial displacement
    figure()
    PlotMesh(nodes, elements, 0);
    PlotFieldonDeformedMesh(nodes, elements, reshape(BeamAssembly.unconstrain_vector(q0), 2, []).');
    title('Initial displacement')

    % Power Spectral Density plot to see fast and slow vibratory modes
    figure()
    spectrogram(q(outdof,:),'yaxis')
    xlabel('Time (s)')
    xticks([])
    title('Power Spectral Density plot')
    
    % Displacement, velocity and acceleration of tip of beam over time
    figure()
    plot(t, x(outdof,:));
    hold on
    grid on
    plot(t, dx(outdof,:));
    plot(t, ddx(outdof,:));
    xlabel('Time (s)');
    ylabel('$q_{Tip}$ (m)', Interpreter='latex');
    legend({'x', 'dx/dt', 'ddx/dtt'})
    title('Displacement, velocity and acceleration of tip of beam over time')
end

%% Finite Difference Function
function [ddx,dx,x,t] = finiteDifference(X_in,t_in,halfw)
    % Central finite time difference with uniform monodimensional grid spacing
    % of customized accuracy, equal to 2*halfw. X is a matrix of n_variables 
    % x n_instances. Therefore, the finite difference is implemented columnwise.
    
    if halfw>4
        disp('The cofficients for this accuracy are not present in the current implementation. The finite difference is computed with accuracy O(Dt^8)')
        halfw = 4;
    end
    % Coefficients for the numerical derivative
    coeff_mat = [1/2   2/3   3/4    4/5; ...
                   0 -1/12 -3/20   -1/5; ...
                   0     0  1/60  4/105; ...
                   0     0     0 -1/280];

    % Coefficients for the second derivative
    coeff_mat2 = [1   4/3   3/2    8/5; ...
                  0 -1/12 -3/20   -1/5; ...
                  0     0  1/90  8/315; ...
                  0     0     0 -1/560];
    coeff_mat3 = [-2, -5/2, -49/18, -205/75];

    % Computation
    base_int = halfw+1:size(X_in,2)-halfw;
    x = X_in(:,base_int); t = t_in(base_int); dt = t(2)-t(1);
    dx = zeros(size(x));
    ddx = coeff_mat3(halfw) * X_in(:,base_int);
    for ii = 1:halfw
        dx = dx + coeff_mat(ii,halfw) * ...
                     (X_in(:,base_int+ii) - X_in(:,base_int-ii));

        ddx = ddx + coeff_mat2(ii,halfw) * ...
                     (X_in(:,base_int+ii) + X_in(:,base_int-ii));
    end
    dx = dx/dt;
    ddx = ddx/(dt^2);
end
