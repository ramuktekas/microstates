%% =========================================================
% eeg_latent_space-style VAR + Microstates (MATLAB)
% Plain OLS implementation - NO external VAR toolbox
%% =========================================================

clear; clc;

%% =========================
% ENVIRONMENT SETUP
%% =========================

rng(42);

addpath('/home/kumarsak/micro-VAR-states/matlab');
addpath('/home/kumarsak/eeglab2025.1.0');
eeglab nogui;

LEMON_DIR = '/store/projects/kumarsak/LEMON_data';
OUT_DIR   = './microstate_results_eeg_latent_style_matlab';

SUBJECT_ID = 'sub-010006_EC';
N_STATES   = 4;
P_FIXED    = 10;

% VAR segment (seconds)
SEGMENT_START  = 60;
SEGMENT_END    = 240;   % 3 minutes

% Simulation length
SIM_SECONDS = 3600;

% Post-VAR microstate filter
MS_FILTER = [2 20];   % Hz

if ~exist(OUT_DIR,'dir')
    mkdir(OUT_DIR);
end

%% =========================
% LOAD EEG
%% =========================

EEG = pop_loadset('filename', [SUBJECT_ID '.set'], ...
                  'filepath', LEMON_DIR);

Fs = EEG.srate;
nch = EEG.nbchan;

fprintf('\n============================================================\n');
fprintf('Processing %s (p = %d) - eeg_latent_space style\n', SUBJECT_ID, P_FIXED);
fprintf('============================================================\n');

fprintf('Full data size: %d channels x %d samples\n', nch, EEG.pnts);
fprintf('Duration: %.1f seconds\n', EEG.xmax);

%% =========================
% STEP 1: Crop segment for VAR (NO filter, NO avg-ref)
%% =========================

EEG_var = pop_select(EEG, 'time', [SEGMENT_START SEGMENT_END]);

data_var = double(EEG_var.data);   % channels x time

% EEGLAB stores data in µV by default
% Sanity check: typical EEG is ~10-100 µV range
data_range = [min(data_var(:)), max(data_var(:))];
if max(abs(data_range)) < 1e-3
    warning('Data looks like Volts, not µV! Converting...');
    data_var = data_var * 1e6;
    data_range = [min(data_var(:)), max(data_var(:))];
end

[nch, T] = size(data_var);

fprintf('\nVAR segment: %d - %d s\n', SEGMENT_START, SEGMENT_END);
fprintf('Segment size: %d x %d\n', nch, T);
fprintf('Data range for VAR (uV): [%.3f, %.3f]\n', data_range(1), data_range(2));

%% =========================
% STEP 2: VAR fitting - PLAIN OLS with intercept
%% =========================

fprintf('\n=== Fitting VAR(%d) with intercept (plain OLS) ===\n', P_FIXED);

% Build lagged matrices manually
% Y(t) = c + A1*Y(t-1) + A2*Y(t-2) + ... + Ap*Y(t-p) + e(t)
%
% Rewrite as: Y = X * B + E
% where:
%   Y is (T-p) x nch       (observations)
%   X is (T-p) x (1 + nch*p)  (constant + lagged values)
%   B is (1 + nch*p) x nch    (coefficients)

Teff = T - P_FIXED;

% Y: current values (time x channels)
Y = data_var(:, P_FIXED+1:end).';  % (T-p) x nch

% X: [1, y(t-1), y(t-2), ..., y(t-p)]
X = ones(Teff, 1 + nch * P_FIXED);  % first column is constant

for lag = 1:P_FIXED
    col_start = 2 + (lag-1)*nch;
    col_end = 1 + lag*nch;
    X(:, col_start:col_end) = data_var(:, P_FIXED+1-lag:T-lag).';
end

fprintf('Y size: %d x %d\n', size(Y,1), size(Y,2));
fprintf('X size: %d x %d\n', size(X,1), size(X,2));

% OLS with ridge regularization (like Python VAR.py)
% Without regularization, X'X is nearly singular for EEG data
XtX = X.' * X;
XtY = X.' * Y;

fprintf('X''X condition number (raw): %.3e\n', cond(XtX));

% Add small ridge regularization
lambda_reg = 1e-6 * trace(XtX) / size(XtX, 1);
XtX_reg = XtX + lambda_reg * eye(size(XtX, 1));

fprintf('X''X condition number (regularized): %.3e\n', cond(XtX_reg));

% Solve regularized OLS
B = XtX_reg \ XtY;   % (1 + nch*p) x nch

fprintf('B size: %d x %d\n', size(B,1), size(B,2));

% Extract intercept and A matrices
intercept = B(1, :);  % 1 x nch

A = zeros(nch, nch, P_FIXED);
for lag = 1:P_FIXED
    row_start = 2 + (lag-1)*nch;
    row_end = 1 + lag*nch;
    A(:,:,lag) = B(row_start:row_end, :).';  % nch x nch
end

% Residuals and covariance
E = Y - X * B;  % (T-p) x nch
Sigma = (E.' * E) / (Teff - size(B,1));  % nch x nch
Sigma = (Sigma + Sigma.') / 2;  % ensure symmetric

fprintf('Intercept norm: %.3e\n', norm(intercept));
fprintf('A sum of abs: %.3e\n', sum(abs(A(:))));
fprintf('Sigma condition number: %.3e\n', cond(Sigma));

% Companion matrix eigenvalues (diagnostic only)
F = zeros(nch * P_FIXED);
for k = 1:P_FIXED
    F(1:nch, (k-1)*nch+1:k*nch) = A(:,:,k);
end
if P_FIXED > 1
    F(nch+1:end, 1:end-nch) = eye(nch*(P_FIXED-1));
end

max_eig = max(abs(eig(F)));
fprintf('Max eigenvalue: %.6f\n', max_eig);
fprintf('Stable (<1): %d\n', max_eig < 1);

%% =========================
% STEP 3: VAR simulation - PLAIN LOOP
%% =========================

Tvar = SIM_SECONDS * Fs;
fprintf('\n=== Simulating VAR for %d samples (%.0f s) ===\n', Tvar, SIM_SECONDS);

% Generate noise from N(0, Sigma)
% Use Cholesky: if Sigma = L*L', then L * randn gives N(0, Sigma)
L = chol(Sigma, 'lower');
noise = (L * randn(nch, Tvar)).';  % Tvar x nch, then transpose -> nch x Tvar
noise = noise.';  % back to Tvar x nch for loop, will fix

% Actually let's keep it simple: nch x Tvar
noise = L * randn(nch, Tvar);  % nch x Tvar

% Initialize simulation
data_sim = zeros(nch, Tvar);

% Simulation loop
% y(t) = c + A1*y(t-1) + A2*y(t-2) + ... + Ap*y(t-p) + e(t)
for t = P_FIXED+1 : Tvar
    data_sim(:, t) = intercept.';  % add intercept
    for lag = 1:P_FIXED
        data_sim(:, t) = data_sim(:, t) + A(:,:,lag) * data_sim(:, t-lag);
    end
    data_sim(:, t) = data_sim(:, t) + noise(:, t);
end

fprintf('Simulated range (uV): [%.3f, %.3f]\n', ...
        min(data_sim(:)), max(data_sim(:)));

%% =========================
% STEP 4: Convert to EEGLAB dataset
%% =========================

EEG_sim = EEG;
EEG_sim.data  = data_sim;
EEG_sim.pnts  = size(data_sim, 2);
EEG_sim.times = (0:EEG_sim.pnts-1) / Fs * 1000;  % in ms
EEG_sim.xmax  = (EEG_sim.pnts-1) / Fs;
EEG_sim = eeg_checkset(EEG_sim);

%% =========================
% STEP 5: FILTER AFTER VAR
%% =========================

EEG_sim = pop_eegfiltnew(EEG_sim, MS_FILTER(1), MS_FILTER(2));

fprintf('\nAfter filtering %d-%d Hz:\n', MS_FILTER(1), MS_FILTER(2));
fprintf('Range (uV): [%.3f, %.3f]\n', ...
        min(EEG_sim.data(:)), max(EEG_sim.data(:)));

%% =========================
% STEP 6: AVG-REFERENCE AFTER VAR
%% =========================

EEG_sim = pop_reref(EEG_sim, []);

fprintf('After avg-ref range (uV): [%.3f, %.3f]\n', ...
        min(EEG_sim.data(:)), max(EEG_sim.data(:)));

%% =========================
% STEP 7: GFP + Microstates
%% =========================

ms = microVARstates.microstates();
data_ms = double(EEG_sim.data);

[gfp_peaks, gfp_curve] = ms.get_gfp_peaks(data_ms);

fprintf('\n=== GFP Analysis ===\n');
fprintf('Number of GFP peaks: %d\n', length(gfp_peaks));
fprintf('GFP range: [%.3e, %.3e]\n', min(gfp_curve), max(gfp_curve));

km_maps = ms.run_modified_k_means( ...
    data_ms(:, gfp_peaks), N_STATES);

fprintf('Microstate maps range: [%.3f, %.3f]\n', ...
        min(km_maps(:)), max(km_maps(:)));

%% =========================
% STEP 8: Template matching
%% =========================

templates = microVARstates.KoenigTemplates.load(N_STATES);

[perm, corr_vals] = ms.match_topomaps_per_template( ...
    km_maps.', EEG_sim.chanlocs, templates);

km_maps_labeled = km_maps(perm,:);

fprintf('\nTemplate correlations:\n');
disp(corr_vals(:).');

%% =========================
% STEP 9: Plot topomaps
%% =========================

figure('Visible', 'off');
for s = 1:N_STATES
    subplot(1, N_STATES, s);
    topoplot(km_maps_labeled(s,:), EEG_sim.chanlocs, 'electrodes', 'off');
    title(['State ' char('A'+s-1)]);
end
sgtitle([SUBJECT_ID ' - VAR Microstates']);

%% =========================
% SAVE RESULTS
%% =========================

out_dir = fullfile(OUT_DIR, SUBJECT_ID);
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

save(fullfile(out_dir, 'var_km_maps.mat'), 'km_maps');
save(fullfile(out_dir, 'var_km_maps_labeled.mat'), 'km_maps_labeled');
save(fullfile(out_dir, 'var_perm.mat'), 'perm');
save(fullfile(out_dir, 'var_corr.mat'), 'corr_vals');
save(fullfile(out_dir, 'var_A.mat'), 'A');
save(fullfile(out_dir, 'var_Sigma.mat'), 'Sigma');
save(fullfile(out_dir, 'var_intercept.mat'), 'intercept');

saveas(gcf, fullfile(out_dir, [SUBJECT_ID '_microstates.png']));
close;

fprintf('\nAll done. Results saved to:\n%s\n', out_dir);
