%% =========================
% ENVIRONMENT SETUP
% =========================

addpath('/home/kumarsak/micro-VAR-states/matlab');

rng(42);
% --- EEGLAB + SIFT ---
addpath('/home/kumarsak/eeglab2025.1.0');
eeglab nogui;

addpath(genpath('/home/kumarsak/eeglab2025.1.0/plugins/SIFT'));
% addpath(genpath('/home/kumarsak/VAR-Toolbox/v3dot0/'))

LEMON_DIR    = '/store/projects/kumarsak/LEMON_data';
OUT_DIR      = './microstate_results_matlab_var';

N_SUBJECTS = 1;
N_STATES   = 4;
SUBJECT_ID =  'sub-010006_EC'; % [] means random


if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

function sets = load_lemon_subjects(lemon_dir, n_subjects)
%LOAD_LEMON_SUBJECTS Load LEMON EEG .set files
%
% n_subjects:
%   'all' → all subjects
%   integer → first n subjects

    files = dir(fullfile(lemon_dir, '*_EC.set'));

    % sort deterministically
    [~, idx] = sort({files.name});
    files = files(idx);

    if ischar(n_subjects) || isstring(n_subjects)
        if strcmp(n_subjects, 'all')
            sets = files;
            return
        else
            error("n_subjects must be 'all' or an integer");
        end
    end

    if isnumeric(n_subjects) && isscalar(n_subjects)
        sets = files(1:min(n_subjects, numel(files)));
        return
    end

    error("n_subjects must be 'all' or an integer");
end

function subject_maps = subject_maps_to_struct(km_maps, EEG)
    subject_maps.data     = km_maps;     % states × channels
    subject_maps.chanlocs = EEG.chanlocs;
    subject_maps.nbchan   = EEG.nbchan;
    subject_maps.n_states = size(km_maps, 1);
end
function [gfp_peaks, gfp_curve, EEG] = preprocess_for_gfp_peaks(EEG, l_freq, h_freq, do_ica)
%PREPROCESS_FOR_GFP_PEAKS Classical EEGLAB microstate preprocessing
%
% Inputs
%   EEG     : EEGLAB EEG structure
%   l_freq  : low cutoff frequency (e.g. 1)
%   h_freq  : high cutoff frequency (e.g. 40)
%   do_ica  : true / false (whether to run ICA cleaning)
%
% Outputs
%   gfp_peaks : indices of GFP peaks
%   gfp_curve : GFP time series
%   EEG       : cleaned EEG (important!)

    if nargin < 4
        do_ica = false;
    end

    %% --------------------------------------------------
    % 1. Band-pass filter (EEGLAB FIR, zero-phase)
    %% -------------------------------------------------
    EEG = pop_eegfiltnew(EEG, l_freq, h_freq);
    EEG = eeg_checkset(EEG);
    %% --------------------------------------------------
    % 3. Average reference (MANDATORY for microstates)
    %% --------------------------------------------------
    EEG = pop_reref(EEG, []);
    EEG = eeg_checkset(EEG);

    %% --------------------------------------------------
    % 2. Run ICA for artifact removal (OPTIONAL but recommended)
    %% --------------------------------------------------
    if do_ica
        % Run ICA
        EEG = pop_runica(EEG, 'extended', 1, 'stop', 1e-7);
        EEG = eeg_checkset(EEG);

        % Classify components
        EEG = pop_iclabel(EEG, 'default');

        % Reject non-brain components automatically
        EEG = pop_icflag(EEG, ...
            [ NaN NaN;    % Brain
              0.9 1;      % Muscle
              0.9 1;      % Eye
              0.9 1;      % Heart
              0.9 1;      % Line Noise
              0.9 1;      % Channel Noise
              NaN NaN ]); % Other

        EEG = eeg_checkset(EEG);
    end
    %% --------------------------------------------------
    % 4. GFP peaks on voltage data
    %% --------------------------------------------------
    data = double(EEG.data);   % channels × time

    ms = microVARstates.microstates();
    [gfp_peaks, gfp_curve] = ms.get_gfp_peaks(data);

end

function plot_microstate_maps(km_maps, EEG, subject_id, out_dir)

    figure('Visible','off');
    n_states = size(km_maps,1);

    for s = 1:n_states
        subplot(1,n_states,s);
        topoplot(km_maps(s,:), EEG.chanlocs, ...
                 'electrodes','off');
        title(['State ' char('A'+s-1)]);
    end

    sgtitle([subject_id ' – Microstate Topographies']);

    if ~exist(out_dir,'dir')
        mkdir(out_dir);
    end

    saveas(gcf, fullfile(out_dir, ...
        [subject_id '_microstates_ABCD.png']));
    close;
end

function [p_opt, aic_vals, diagnostics] = estimate_var_order(setfile)
    EEG = pop_loadset('filename', setfile.name, ...
                      'filepath', setfile.folder);
    
    % Same preprocessing as main pipeline
    [~, ~, EEG] = preprocess_for_gfp_peaks(EEG, 1, 40, false);
    data = double(EEG.data);
    
    varModel = microVARstates.VAR();
    varModel.verbose = true;  % Print diagnostics
    varModel.save_diagnostics = true;  % Include in output
    
    % IMPORTANT: use limited p_max
    p_max = 10;
    [p_opt, aic_vals, diagnostics] = varModel.select_order(data, p_max, true);
end

function [km_maps_labeled, EEG, subj_id, out_dir, diagnostics] = ...
    process_subject_fixed_p(setfile, OUT_DIR, N_STATES, p_fixed)

    subj_id = erase(setfile.name, '.set');
    fprintf('\n=== Processing %s (p = %d) ===\n', subj_id, p_fixed);
    
    % Initialize diagnostics struct
    diagnostics = struct();
    diagnostics.subject_id = subj_id;
    diagnostics.p_fixed = p_fixed;
    diagnostics.n_states = N_STATES;

    %% --- Load EEG ---
    EEG = pop_loadset('filename', setfile.name, ...
                      'filepath', setfile.folder);
    
    diagnostics.original_data_size = size(EEG.data);
    diagnostics.srate = EEG.srate;

    %% --- Preprocessing ---
    [~, ~, EEG] = preprocess_for_gfp_peaks(EEG, 1, 40, false);
    data = double(EEG.data);
    
    diagnostics.preprocessed_data_size = size(data);
    diagnostics.data_range_original = [min(data(:)), max(data(:))];

    %% --- VAR with fixed order ---
    varModel = microVARstates.VAR();
    varModel.verbose = true;
    varModel.save_diagnostics = true;

    model = varModel.fit(data, p_fixed);
    
    % Store fit diagnostics
    if isfield(model, 'diagnostics')
        diagnostics.var_fit = model.diagnostics;
    end

    %% --- Simulate long VAR data ---
    Tvar = 3600 * EEG.srate;
    [data_sim, sim_diag] = varModel.simulate(model, Tvar);
    
    % Store simulation diagnostics
    diagnostics.var_simulation = sim_diag;
    
    % Detailed checks
    diagnostics.simulation_checks = struct();
    diagnostics.simulation_checks.size = size(data_sim);
    diagnostics.simulation_checks.range = [min(data_sim(:)), max(data_sim(:))];
    diagnostics.simulation_checks.contains_nan = any(isnan(data_sim(:)));
    diagnostics.simulation_checks.contains_inf = any(isinf(data_sim(:)));
    diagnostics.simulation_checks.mean_abs = mean(abs(data_sim(:)));
    diagnostics.simulation_checks.std_mean = mean(std(data_sim, 0, 2));
    
    % ABORT if simulation failed
    if diagnostics.simulation_checks.contains_nan || ...
       diagnostics.simulation_checks.contains_inf || ...
       diagnostics.simulation_checks.mean_abs > 1e6
        error('VAR simulation produced invalid data. Model is still unstable.');
    end

    %% --- GFP Analysis ---
    ms = microVARstates.microstates();
    [gfp_peaks_var, gfp_curve] = ms.get_gfp_peaks(data_sim);
    
    diagnostics.gfp = struct();
    diagnostics.gfp.n_peaks = length(gfp_peaks_var);
    diagnostics.gfp.peak_range = [min(gfp_peaks_var), max(gfp_peaks_var)];
    diagnostics.gfp.gfp_curve_range = [min(gfp_curve), max(gfp_curve)];
    diagnostics.gfp.gfp_curve_mean = mean(gfp_curve);
    
    if length(gfp_peaks_var) < N_STATES
        error('Not enough GFP peaks (%d) for %d states', ...
              length(gfp_peaks_var), N_STATES);
    end
    
    peaks_data = data_sim(:, gfp_peaks_var);
    diagnostics.gfp.peaks_data_range = [min(peaks_data(:)), max(peaks_data(:))];
    diagnostics.gfp.peaks_contain_nan = any(isnan(peaks_data(:)));

    %% --- Microstates ---
    km_maps = ms.run_modified_k_means(data_sim(:, gfp_peaks_var), N_STATES);
    
    diagnostics.microstates = struct();
    diagnostics.microstates.n_states = size(km_maps, 1);
    diagnostics.microstates.n_channels = size(km_maps, 2);
    diagnostics.microstates.maps_range = [min(km_maps(:)), max(km_maps(:))];

    %% --- Template matching ---
    subject_maps = subject_maps_to_struct(km_maps, EEG);
    templates = microVARstates.KoenigTemplates.load(N_STATES);

    [attribution, corr] = ms.match_topomaps_per_template( ...
        subject_maps.data.', subject_maps.chanlocs, templates);

    km_maps_labeled = km_maps(attribution, :);
    
    diagnostics.template_matching = struct();
    diagnostics.template_matching.attribution = attribution;
    diagnostics.template_matching.correlations = corr;
    diagnostics.template_matching.mean_correlation = mean(corr);

    %% --- Save everything ---
    out_dir = fullfile(OUT_DIR, subj_id);
    if ~exist(out_dir,'dir')
        mkdir(out_dir);
    end
    
    % Save model and results
    save(fullfile(out_dir,'var_model.mat'), 'model');
    save(fullfile(out_dir,'var_p_fixed.mat'), 'p_fixed');
    save(fullfile(out_dir,'var_km_maps.mat'), 'km_maps');
    save(fullfile(out_dir,'var_km_maps_labeled.mat'), 'km_maps_labeled');
    save(fullfile(out_dir,'var_permutation.mat'), 'attribution');
    
    % Save comprehensive diagnostics
    save(fullfile(out_dir,'var_diagnostics_full.mat'), 'diagnostics');
    
    % Also save a summary text file
    write_diagnostics_summary(fullfile(out_dir,'var_diagnostics_summary.txt'), diagnostics);
end

function write_diagnostics_summary(filename, diagnostics)
    % Write human-readable diagnostics summary
    fid = fopen(filename, 'w');
    
    fprintf(fid, '=== VAR-Microstate Analysis Diagnostics ===\n\n');
    fprintf(fid, 'Subject: %s\n', diagnostics.subject_id);
    fprintf(fid, 'VAR Order: %d\n', diagnostics.p_fixed);
    fprintf(fid, 'Number of States: %d\n\n', diagnostics.n_states);
    
    fprintf(fid, '--- Data ---\n');
    fprintf(fid, 'Original size: %d × %d\n', diagnostics.original_data_size(1), diagnostics.original_data_size(2));
    fprintf(fid, 'Sampling rate: %.1f Hz\n', diagnostics.srate);
    fprintf(fid, 'Data range: [%.4e, %.4e]\n\n', diagnostics.data_range_original(1), diagnostics.data_range_original(2));
    
    if isfield(diagnostics, 'var_fit')
        fprintf(fid, '--- VAR Fitting ---\n');
        fprintf(fid, 'Original max eigenvalue: %.6f\n', diagnostics.var_fit.max_eig_original);
        fprintf(fid, 'Final max eigenvalue: %.6f\n', diagnostics.var_fit.max_eig_final);
        fprintf(fid, 'Rescale iterations: %d\n', diagnostics.var_fit.rescale_iterations);
        if isfield(diagnostics.var_fit, 'stabilized')
            fprintf(fid, 'Stabilized: %d\n\n', diagnostics.var_fit.stabilized);
        else
            fprintf(fid, 'Stabilized: 0\n\n');
        end

    end
    
    if isfield(diagnostics, 'var_simulation')
        fprintf(fid, '--- VAR Simulation ---\n');
        fprintf(fid, 'Simulated length: %d samples\n', diagnostics.var_simulation.T);
        fprintf(fid, 'Sigma condition number: %.4e\n', diagnostics.var_simulation.sigma_condition);
        fprintf(fid, 'Data range: [%.4e, %.4e]\n', diagnostics.var_simulation.data_range(1), diagnostics.var_simulation.data_range(2));
        fprintf(fid, 'Contains NaN: %d\n', diagnostics.var_simulation.contains_nan);
        fprintf(fid, 'Contains Inf: %d\n', diagnostics.var_simulation.contains_inf);
        fprintf(fid, 'Mean absolute value: %.4e\n', diagnostics.var_simulation.mean_abs);
        fprintf(fid, 'Mean std: %.4e\n\n', diagnostics.var_simulation.mean_std);
    end
    
    if isfield(diagnostics, 'gfp')
        fprintf(fid, '--- GFP Analysis ---\n');
        fprintf(fid, 'Number of peaks: %d\n', diagnostics.gfp.n_peaks);
        fprintf(fid, 'GFP curve range: [%.4e, %.4e]\n', diagnostics.gfp.gfp_curve_range(1), diagnostics.gfp.gfp_curve_range(2));
        fprintf(fid, 'Peaks data range: [%.4e, %.4e]\n\n', diagnostics.gfp.peaks_data_range(1), diagnostics.gfp.peaks_data_range(2));
    end
    
    if isfield(diagnostics, 'template_matching')
        fprintf(fid, '--- Template Matching ---\n');
        fprintf(fid, 'Attribution: [%s]\n', num2str(diagnostics.template_matching.attribution));
        fprintf(fid, 'Correlations: [%s]\n', num2str(diagnostics.template_matching.correlations, '%.4f '));
        fprintf(fid, 'Mean correlation: %.4f\n', diagnostics.template_matching.mean_correlation);
    end
    
    fclose(fid);
    fprintf('Diagnostics summary saved to: %s\n', filename);
end

%% =========================
% MODIFIED LOOP IN MAIN SCRIPT
% =========================

% In your main script, update the loops like this:
%% =========================
% LOAD SUBJECTS (THIS WAS MISSING)
% =========================

if ~isempty(SUBJECT_ID)
    % Load exactly one subject
    sets = dir(fullfile(LEMON_DIR, [SUBJECT_ID '.set']));
    if isempty(sets)
        error('Subject not found: %s', SUBJECT_ID);
    end
else
    % Load first N subjects
    sets = load_lemon_subjects(LEMON_DIR, N_SUBJECTS);
end

nSubj = numel(sets);


%% ---------- PASS 1: estimate p for each subject ----------
fprintf('\n=== Estimating VAR order per subject ===\n');

for i = 1:nSubj
    % Get p_opt, AIC values, AND diagnostics
    [p_opts(i), aic_vals, order_diagnostics] = estimate_var_order(sets(i));
    
    % Get subject ID
    subj_id = erase(sets(i).name, '.set');
    
    % Create output directory
    out_dir = fullfile(OUT_DIR, subj_id);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    
    % Save AIC values, optimal p, AND diagnostics
    p_opt = p_opts(i);
    save(fullfile(out_dir, 'var_aic.mat'), 'aic_vals', 'p_opt');
    save(fullfile(out_dir, 'var_order_selection_diagnostics.mat'), 'order_diagnostics');
    
    fprintf('Subject %d (%s): p_opt = %d\n', i, subj_id, p_opts(i));
end

p_med = median(p_opts);
fprintf('\n>>> Median VAR order across subjects: p = %d <<<\n', p_med);

save(fullfile(OUT_DIR,'var_popts_all_subjects.mat'), 'p_opts', 'p_med');

%% ---------- PASS 2: full pipeline using fixed p ----------
fprintf('\n=== Running full VAR + microstates with fixed p ===\n');

all_diagnostics = cell(nSubj, 1);

for i = 1:nSubj
    [km_maps, EEG, subj_id, out_dir, diagnostics] = ...
        process_subject_fixed_p(sets(i), OUT_DIR, N_STATES, 10);

    plot_microstate_maps(km_maps, EEG, subj_id, out_dir);
    
    % Store diagnostics for aggregation
    all_diagnostics{i} = diagnostics;
end

% Save aggregated diagnostics across all subjects
save(fullfile(OUT_DIR, 'all_subjects_diagnostics.mat'), 'all_diagnostics');

disp('All subjects processed.');
