%% =========================
% ENVIRONMENT SETUP
% =========================

addpath('/home/kumarsak/micro-VAR-states/matlab');

rng(42);
% --- EEGLAB + SIFT ---
addpath('/home/kumarsak/eeglab2025.1.0');
eeglab nogui;

addpath(genpath('/home/kumarsak/eeglab2025.1.0/plugins/SIFT'));
addpath(genpath('/home/kumarsak/VAR-Toolbox/v3dot0/'))

LEMON_DIR    = '/store/projects/kumarsak/LEMON_data';
OUT_DIR      = './microstate_results_matlab_var';

N_SUBJECTS = 'all';
N_STATES   = 4;
SUBJECT_ID =  ''; % [] means random


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

function p_opt = estimate_var_order(setfile)

    EEG = pop_loadset('filename', setfile.name, ...
                      'filepath', setfile.folder);

    % same preprocessing as main pipeline
    [~, ~, EEG] = preprocess_for_gfp_peaks(EEG, 1, 40, false);

    data = double(EEG.data);

    varModel = microVARstates.VAR();

    % IMPORTANT: use limited p_max
    p_max = 10;
    [p_opt, ~] = varModel.select_order(data, p_max);

end
function [km_maps_labeled, EEG, subj_id, out_dir] = ...
    process_subject_fixed_p(setfile, OUT_DIR, N_STATES, p_fixed)

    subj_id = erase(setfile.name, '.set');
    fprintf('\n=== Processing %s (p = %d) ===\n', subj_id, p_fixed);

    %% --- Load EEG ---
    EEG = pop_loadset('filename', setfile.name, ...
                      'filepath', setfile.folder);

    %% --- Preprocessing ---
    [~, ~, EEG] = preprocess_for_gfp_peaks(EEG, 1, 40, false);
    data = double(EEG.data);

    %% --- VAR with fixed order ---
    varModel = microVARstates.VAR();

    model = varModel.fit(data, p_fixed);

    Tvar = 3600 * EEG.srate;
    data_sim = varModel.simulate(model, Tvar);

    %% --- Microstates on VAR data ---
    ms = microVARstates.microstates();

    [gfp_peaks_var, ~] = ms.get_gfp_peaks(data_sim);

    km_maps = ms.run_modified_k_means( ...
        data_sim(:, gfp_peaks_var), ...
        N_STATES);

    %% --- Template matching ---
    subject_maps = subject_maps_to_struct(km_maps, EEG);
    templates = microVARstates.KoenigTemplates.load(N_STATES);

    [attribution, corr] = ms.match_topomaps_per_template( ...
        subject_maps.data.', subject_maps.chanlocs, templates);

    km_maps_labeled = km_maps(attribution, :);

    %% --- Save ---
    out_dir = fullfile(OUT_DIR, subj_id);
    if ~exist(out_dir,'dir')
        mkdir(out_dir);
    end

    save(fullfile(out_dir,'var_model.mat'), 'model');
    save(fullfile(out_dir,'var_p_fixed.mat'), 'p_fixed');
    save(fullfile(out_dir,'var_km_maps.mat'), 'km_maps');
    save(fullfile(out_dir,'var_km_maps_labeled.mat'), 'km_maps_labeled');
    save(fullfile(out_dir,'var_permutation.mat'), 'attribution');

end
%% =========================
% RUN
% =========================

if ~isempty(SUBJECT_ID)
    sets = dir(fullfile(LEMON_DIR, [SUBJECT_ID '.set']));
    if isempty(sets)
        error('Subject %s not found', SUBJECT_ID);
    end
else
    sets = load_lemon_subjects(LEMON_DIR, N_SUBJECTS);
end

nSubj = numel(sets);
p_opts = zeros(nSubj,1);

%% ---------- PASS 1: estimate p for each subject ----------
fprintf('\n=== Estimating VAR order per subject ===\n');

for i = 1:nSubj
    p_opts(i) = estimate_var_order(sets(i));
    fprintf('Subject %d: p_opt = %d\n', i, p_opts(i));
end

p_med = median(p_opts);
fprintf('\n>>> Median VAR order across subjects: p = %d <<<\n', p_med);

save(fullfile(OUT_DIR,'var_popts_all_subjects.mat'), 'p_opts', 'p_med');

%% ---------- PASS 2: full pipeline using fixed p ----------
fprintf('\n=== Running full VAR + microstates with fixed p ===\n');

for i = 1:nSubj
    [km_maps, EEG, subj_id, out_dir] = ...
        process_subject_fixed_p(sets(i), OUT_DIR, N_STATES, p_med);

    plot_microstate_maps(km_maps, EEG, subj_id, out_dir);
end

disp('All subjects processed.');
