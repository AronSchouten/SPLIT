% Test performance of Autoencoder
% Requires YetAnotherFEcode
clear;
close all;
clc

% Degree of freedom (tip = 122)
outdof = 122;
% Redo FOM simulation
redo_FOM = false;

%                     % Small % Large %  Fold %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training trajectory %  '2'  %  '3'  %  '5'  %
%  Testing trajectory %  '1'  %  '2'  %  '4'  %
%  Forcing amplitude  %   1   %  600  %  6000 %
%  Number of periods  %  200  %  200  %  400  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Training trajectory number (1-5)
train = '3';
% Testing trajectory number (1-5)
test = '2';
% Forcing amplitude
epsilon = 600;
% Number of periods to simulate (forced response)
NumPeriods = 200;

%                       % Decoupled AE %   SINDy AE   %    SPLIT     %    SPLIT+    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Method         %     'AE'     %   'SINDyAE'  %    'SPLIT'   %    'SPLIT'   %
% Polynomial regression %     True     %     False    %     False    %     True     %
%    Use linear terms   %     False    %              %              %     True     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Method
method = 'SPLIT';
% Use polynomial regression?
PolReg = false;
% Use linear terms in polynomial regression?
UseLinear = [];

name = strcat('data\trajectories\unforced_non_linear_', train, '.mat');
load(name);
name = strcat('data\learned_parameters\', method, '_', train, '.mat');
load(name);

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
nDofs = myMesh.nDOFs;
u0 = zeros(nDofs, 1);
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

[phi1, D] = eigs(Kc,Mc,1,'SM');
mu = phi1' * Mc * phi1; % Mass normalize
phi1 = phi1/sqrt(mu);
omega1 = sqrt(D);

lambda = zeros(2);

zeta = 1/(2*omega1) * ( phi1' * Cc * phi1 )/( phi1' * Mc * phi1 );
lambda(1) = (-zeta-sqrt(zeta.^2-1))*omega1;
lambda(2) = (-zeta+sqrt(zeta.^2-1))*omega1;   

eigfreq = imag(lambda(2));
damprat = real(lambda(2));

%% Fit dynamics

if PolReg
    z = encode(x, enc_weights);
    [dz, ddz] = encode_d(x, dx, ddx, enc_weights);
    
    dEtadt = [dz; ddz]; Eta = [z; dz]; time = t;
    
    if UseLinear
        phi = @(y) [y(1,:).*y(1,:); y(1,:).*y(2,:); y(2,:).*y(2,:); y(1,:).*y(1,:).*y(1,:); y(1,:).*y(1,:).*y(2,:); y(1,:).*y(2,:).*y(2,:); y(2,:).*y(2,:).*y(2,:)];
        R0 = [0, 1;
              -k, -c];
        R1 = ((phi(Eta) * phi(Eta)')\(phi(Eta) * dEtadt' - phi(Eta)*(R0*Eta)' ))';
        
        R = [R0, R1];
    else
        phi = @(y) [y(1,:); y(2,:); y(1,:).*y(1,:); y(1,:).*y(2,:); y(2,:).*y(2,:); y(1,:).*y(1,:).*y(1,:); y(1,:).*y(1,:).*y(2,:); y(1,:).*y(2,:).*y(2,:); y(2,:).*y(2,:).*y(2,:)];
        X = phi(Eta)';
        Y = dEtadt';
        L = ones(size(time'));
        
        X_normal = max(abs(X),[],1);
        X = X./X_normal;
        XL = L.*X;
        R = (diag(X_normal.^(-1)) * (( XL'*X )\( XL'*Y )))';
    end
end

phi = @(y) [y(1,:); y(2,:); y(1,:).*y(1,:); y(1,:).*y(2,:); y(2,:).*y(2,:); y(1,:).*y(1,:).*y(1,:); y(1,:).*y(1,:).*y(2,:); y(1,:).*y(2,:).*y(2,:); y(2,:).*y(2,:).*y(2,:)];   

%% Test performance
name = strcat('data\trajectories\unforced_non_linear_', test, '.mat');
load(name);

z = encode(x, enc_weights);
[dz, ~] = encode_d(x, dx, ddx, enc_weights);

xRec = decode(z, dec_weights);

% Use learned coefficients when not using polynomial regression
if ~PolReg
    R = [[0, 1, 0, 0, 0, 0, 0, 0, 0]; sindy_coefficients'];
end

tStart = t(1);
tEnd = t(end);
nSamp = length(t);
[~, z_out] = ode78(@(t,y) R*phi(y), linspace(tStart, tEnd, nSamp), [z(1), dz(1)]);

xRep = decode(z_out(:,1)', dec_weights);

%% Backbone curves (Requires SSMLearn's PFF)
% [amp,freq,damp,~] = PFF(t,x(outdof,:));
% [ampRec,freqRec,dampRec,~] = PFF(t,xRec(outdof,:));
% [ampRep,freqRep,dampRep,~] = PFF(t,xRep(outdof,:));
% 
% figure()
% title('Backbone curves (Obtained using Peak Finding & Fitting)')
% plot(freq,amp, 'LineWidth', 3)
% hold on
% plot(freqRec,ampRec, '--', 'LineWidth', 3)
% plot(freqRep,ampRep, '-.', 'LineWidth', 3)
% legend({'Original', 'Reconstructed', 'Reproduced'}, 'fontsize', 30)
% xlabel('Frequency (Hz)', 'fontsize', 30);
% ylabel('$q_{tip}$ (m)', 'Interpreter','latex', 'fontsize', 30);
% set(gca, 'fontsize', 30)
% set(gcf,'position',[50 50 1000 600]);

%% Analyzing performance
% Calculate reconstruction error
normedTrajDist = mean(vecnorm(x - xRec, 2, 1)) / max(vecnorm(x, 2, 1));
NMTE = mean(normedTrajDist)*100;
disp(['Normalized mean trajectory error (Reconstruction) = ' num2str(NMTE) '%'])

% Calculate reproduction error
normedTrajDist = mean(vecnorm(x - xRep, 2, 1)) / max(vecnorm(x, 2, 1));
NMTE = mean(normedTrajDist)*100;
disp(['Normalized mean trajectory error (Reproduction) = ' num2str(NMTE) '%'])

% Dynamics in reduced coordinates
figure()
plot(z, dz, 'LineWidth', 3)
hold on
grid on
plot(z_out(:,1), z_out(:,2), 'LineWidth', 3, LineStyle='--')
legend({'True','Prediction'}, 'fontsize', 30)
xlabel('$z$', 'Interpreter', 'latex', 'fontsize', 30);
ylabel('$\dot{z}$', 'Interpreter', 'latex', 'fontsize', 30);
title('Dynamics in reduced coordinates')
set(gca, 'fontsize', 30)
set(gcf,'position',[50 50 750 600]);

% Dynamics in full coordinates
figure()
plot(t, x(outdof,:), 'LineWidth', 3);
hold on
grid on
plot(t, xRec(outdof,:), 'LineWidth', 3, LineStyle="--");
plot(t, xRep(outdof,:), 'LineWidth', 3, LineStyle="--");
xlabel('Time (s)', 'fontsize', 30);
ylabel('$q_{tip}$ (m)', 'Interpreter','latex', 'fontsize', 30);
legend({'Original', 'Reconstructed', 'Reproduced'}, 'fontsize', 30)
title('Displacement of tip of beam over time', 'fontsize', 30)
set(gca, 'fontsize', 30)
set(gcf,'position',[50 50 1000 600]);

%% Adding external forcing
% Define variables
T = 2*pi/eigfreq; % time period of forcing
h = T/1000; %5e-4;
tmax = NumPeriods*T;

% FOM
q0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));
qd0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));
qdd0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));

F_ext_FOM = @(t) epsilon*BeamAssembly.unconstrain_vector(Mc * phi1) * sin(eigfreq * t);

if redo_FOM
    residual_FOM = @(q,qd,qdd,t) residual_nonlinear(q,qd,qdd,t,BeamAssembly,F_ext_FOM);
    TI_FOM = ImplicitNewmark('timestep',h,'alpha',0.005,'linear',false);
    TI_FOM.Integrate(q0_FOM,qd0_FOM,qdd0_FOM,tmax,residual_FOM);
    x_FOM = TI_FOM.Solution.q;
    t = TI_FOM.Solution.time;
else
    name = strcat('data\trajectories\forced_non_linear_', test, '.mat');
    load(name)
end

% ROM
q0_ROM = [encode(q0_FOM, enc_weights); encode(qd0_FOM, enc_weights)];
F_ext_ROM = @(t,z) [0; alpha(z, dec_weights) * BeamAssembly.constrain_vector(M \ F_ext_FOM(t))];

[t_ROM, z_out_ROM] = ode78(@(t,y) R*phi(y) + F_ext_ROM(t, y(1)), 0:h:tmax, q0_ROM);
x_ROM = decode(z_out_ROM(:,1)', dec_weights);

normedTrajDist = mean(vecnorm(x_FOM(:,end-1000:end) - x_ROM(:,end-1000:end), 2, 1)) / max(vecnorm(x_FOM(:,end-1000:end), 2, 1));
NMTE = mean(normedTrajDist)*100;
disp(['Normalized mean trajectory error (External Forcing) = ' num2str(NMTE) '%'])

figure()
plot(t_FOM, x_FOM(outdof,:), 'LineWidth', 3);
hold on
grid on
plot(t_ROM, x_ROM(outdof,:), 'LineWidth', 3, LineStyle="--");
xlabel('Time (s)', 'fontsize', 30);
ylabel('$q_{tip}$ (m)', 'Interpreter','latex', 'fontsize', 30);
legend({'FOM', 'ROM'}, 'fontsize', 30)
title('Displacement of tip of beam over time')
set(gca, 'fontsize', 30)
set(gcf,'position',[50 50 1000 600]);

%% FRC
% Zero initial condition
q0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));
qd0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));
qdd0_FOM = BeamAssembly.constrain_vector(zeros(nDofs, 1));

q0_ROM = [encode(q0_FOM, enc_weights); encode(qd0_FOM, enc_weights)];

% Arrays for saving
freqs = linspace(0.8, 1.2, 33)*eigfreq;
amps_FOM = zeros(size(freqs));
phs_FOM = zeros(size(freqs));
amps_ROM = zeros(size(freqs));
phs_ROM = zeros(size(freqs));

i=0;
for freq = freqs
        % Define variables
        T = 2*pi/freq; % time period of forcing
        h = T/1000; %5e-4;
        tmax = 200*T;
    
        % Simulate
        F_ext_FOM = @(t) epsilon * BeamAssembly.unconstrain_vector(Mc * phi1) * sin(freq * t);
        if redo_FOM
            residual_FOM = @(q,qd,qdd,t) residual_nonlinear(q,qd,qdd,t,BeamAssembly,F_ext_FOM);
            TI_FOM = ImplicitNewmark('timestep',h,'alpha',0.005,'linear',false);
            TI_FOM.Integrate(q0_FOM,qd0_FOM,qdd0_FOM,tmax,residual_FOM);
            x_FOM = TI_FOM.Solution.q;
            t = TI_FOM.Solution.time;
        else
            t=0:h:tmax;
        end
    
        tStart = t(1);
        tEnd = t(end);
        nSamp = length(t);
    
        F_ext_ROM = @(t,z) [0; alpha(z, dec_weights) * BeamAssembly.constrain_vector(M \ F_ext_FOM(t))];
        [~, z_out_ROM] = ode78(@(t,y) R*phi(y) + F_ext_ROM(t, y(1)), linspace(tStart, tEnd, nSamp), q0_ROM);
        x_ROM = decode(z_out_ROM(:,1)', dec_weights);
        
        i=i+1;
        fprintf('%i', i)
        % Save amplitude and phase using last period
        phiEval = -2*pi*t(end-T/h+1:end)/T;
        cPhi = cos(phiEval); sPhi = sin(phiEval);

        if redo_FOM
            amps_FOM(i) = max(x_FOM(outdof,end-T/h+1:end));
            zfTemp_FOM = t(2)* sum(x_FOM(outdof,end-T/h+1:end).*(cPhi+1i*sPhi));
            phs_FOM(i) = atan2(imag(zfTemp_FOM), real(zfTemp_FOM)) + pi/2;
        end

        amps_ROM(i) = max(x_ROM(outdof,end-T/h+1:end));
        zfTemp_ROM = t(2)* sum(x_ROM(outdof,end-T/h+1:end).*(cPhi+1i*sPhi));
        phs_ROM(i) = atan2(imag(zfTemp_ROM), real(zfTemp_ROM)) + pi/2;
end

% Make sure -2pi<phase<0
phs_ROM = phs_ROM .* (phs_ROM <= 0) + (phs_ROM - 2*pi) .* (phs_ROM > 0);
phs_FOM = phs_FOM .* (phs_FOM <= 0) + (phs_FOM - 2*pi) .* (phs_FOM > 0);

if ~redo_FOM
    name = strcat('data\forced_response_curves\non_linear_', test, '.mat');
    load(name);
end

figure()
plot(freqs_FOM, amps_FOM, 'LineWidth', 3);
hold on
grid on
plot(freqs, amps_ROM, 'LineStyle', '--', 'LineWidth', 3);
xlabel('Forcing frequency (Hz)', 'fontsize', 30);
ylabel('Amplitude of $q_{tip}$ (m)', 'Interpreter','latex', 'fontsize', 30); 
title('FRC of Amplitude')
legend({'FOM', 'ROM'}, 'fontsize', 30)
set(gca, 'fontsize', 30)
set(gcf,'position',[50 50 1000 600]);

figure()
plot(freqs_FOM, phs_FOM, 'LineWidth', 3);
hold on
grid on
plot(freqs, phs_ROM, 'LineStyle', '--', 'LineWidth', 3);
set(gca,'YTick',-pi:pi/4:0)
set(gca,'YTickLabel',{'-\pi','-3\pi/4','\pi/2','-\pi/4','0'},'fontsize',30)
xlabel('Forcing frequency (Hz)', 'fontsize', 30);
ylabel('Phase of $q_{tip}$', 'Interpreter','latex', 'fontsize', 30); 
title('FRC of Phase')
legend({'FOM', 'ROM'}, 'fontsize', 30)
set(gcf,'position',[50 50 1000 600]);

%% Functions for autoencoder
function x_out = elu(x_in)
    a = 1;
    x_out = (a*(exp(x_in)-1)).*((x_in < 0)) + (x_in).*((x_in > 0));
end

function x = decode(z, dec_weights)
    depth = length(dec_weights);
    x = z;
    for i=1:depth
        x = dec_weights{i}' * x;
        % Activation function if not at final layer
        if i < depth
            x = elu(x);
        end
    end
    x = double(x);
end

function z = encode(x, enc_weights)
    depth = length(enc_weights);
    z = x;
    for i=1:depth
        z = enc_weights{i}' * z;  
        % Activation function if not at final layer
        if i < depth
            z = elu(z);
        end
    end
    z = double(z);
end

function [dz, ddz] = encode_d(x, dx, ddx, enc_weights)
    depth = length(enc_weights);
    dz = dx;
    ddz = ddx;
    for i=1:depth-1
        x = enc_weights{i}' * x;
        dz_prev = enc_weights{i}' * dz;
        elu_derivative = min(exp(x),1.0);
        mask = zeros(size(x));
        mask(x < 0) = 1;
        elu_derivative2 = exp(x) .* mask;
        dz = dz_prev .* elu_derivative;
        ddz = elu_derivative2 .* dz_prev.^2 ...
              + elu_derivative .* (enc_weights{i}' * ddz);
        x = elu(x);
    end
    dz = enc_weights{end}' * dz;
    ddz = enc_weights{end}' * ddz;

    dz = double(dz);
    ddz = double(ddz);
end

function alph = alpha(q, dec_weights)
    depth = length(dec_weights);
    dz = eye(size(q));
    for i=1:depth-1
        q = dec_weights{i}' * q;
        dz_prev = dec_weights{i}' * dz;
        elu_derivative = min(exp(q),1.0);
        dz = dz_prev .* elu_derivative;
        q = elu(q);
    end
    dz = dec_weights{end}' * dz;

    alph = dz/(vecnorm(dz, 2)^2);
    alph = double(alph)';
end