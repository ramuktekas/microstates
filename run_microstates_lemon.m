%% =========================
% ENVIRONMENT SETUP
% =========================

addpath('/home/kumarsak/micro-VAR-states/matlab');

rng(42);

LEMON_DIR    = '/store/projects/kumarsak/LEMON_data';
OUT_DIR      = './microstate_results_matlab';

N_SUBJECTS = 1;
N_STATES   = 4;

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

function sets = load_lemon_subjects(lemon_dir, n_subjects)

    files = dir(fullfile(lemon_dir, '*_EC.set'));

    if numel(files) < n_subjects
        error('Requested %d subjects, but only %d found', ...
              n_subjects, numel(files));
    end

    idx  = randperm(numel(files), n_subjects);
    sets = files(idx);
end
function [gfp_peaks, gfp_curve] = preprocess_for_gfp_peaks(EEG, l_freq, h_freq)

    % Bandpass filter
    EEGf = pop_eegfiltnew(EEG, l_freq, h_freq);

    % Hilbert envelope
    EEGf.data = abs(hilbert(double(EEGf.data')')) ;

    % data: channels × time
    data = double(EEGf.data);

    % microstates toolbox
    ms = microVARstates.microstates();

    [gfp_peaks, gfp_curve] = ms.get_gfp_peaks(data);
end
function subject_maps = subject_maps_to_struct(km_maps, EEG)

    subject_maps.data      = km_maps;     % states × channels
    subject_maps.chanlocs  = EEG.chanlocs;
    subject_maps.nbchan    = EEG.nbchan;
    subject_maps.n_states  = size(km_maps, 1);
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
function [km_maps_labeled, EEG, subj_id, out_dir] = process_subject(setfile, OUT_DIR, N_STATES)

    subj_id = erase(setfile.name, '.set');
    fprintf('\n=== Processing %s ===\n', subj_id);

    %% --- Load EEG ---
    EEG = pop_loadset('filename', setfile.name, ...
                      'filepath', setfile.folder);

    data = double(EEG.data);  % channels × time

    %% --- GFP peaks ---
    [gfp_peaks, ~] = preprocess_for_gfp_peaks(EEG, 1, 40);

    %% --- Microstate maps (sensor space) ---
    ms = microVARstates.microstates();

    km_maps = ms.run_modified_k_means( ...
        data(:, gfp_peaks), ...
        N_STATES);

    % km_maps: states × channels

    %% --- Wrap subject maps ---
    subject_maps = subject_maps_to_struct(km_maps, EEG);

    %% --- Load Koenig templates ---
    templates = microVARstates.KoenigTemplates.load(N_STATES);
    
    %% --- Match topomaps (Nikola-style) ---
    ms = microVARstates.microstates();

    [attribution, corr] = ...
        ms.match_topomaps_per_template(subject_maps.data.', templates)


    %% --- Apply permutation ---
    km_maps_labeled = km_maps(attribution, :);

    %% --- Save ---
    out_dir = fullfile(OUT_DIR, subj_id);
    if ~exist(out_dir,'dir')
        mkdir(out_dir);
    end

    save(fullfile(out_dir,'km_maps.mat'), 'km_maps');
    save(fullfile(out_dir,'km_maps_labeled.mat'), 'km_maps_labeled');
    save(fullfile(out_dir,'microstate_permutation.mat'), 'attribution');
    save(fullfile(out_dir,'template_correlations.mat'), 'corr');
end
%% =========================
% RUN
% =========================

sets = load_lemon_subjects(LEMON_DIR, N_SUBJECTS);

for i = 1:numel(sets)

    [km_maps, EEG, subj_id, out_dir] = ...
        process_subject(sets(i), OUT_DIR, N_STATES);

    plot_microstate_maps(km_maps, EEG, subj_id, out_dir);
end

disp('All subjects processed.');
