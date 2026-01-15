classdef VAR
    %VAR Vector Autoregressive (VAR) model utilities
    %
    % This class provides stateless methods for:
    %   1) VAR order selection using AIC
    %   2) VAR model fitting
    %   3) VAR time series simulation
    %
    % All methods operate on data matrices of shape:
    %   (n_channels × n_timepoints)

    methods

        %--------------------------------------------------
        % SELECT_ORDER
        %--------------------------------------------------
        function [p_opt, aic_vals] = select_order(obj, data, p_max, elbow)
        %SELECT_ORDER Select VAR order using AIC (with diagnostics & stabilization)
        %
        % Inputs
        % -------
        % obj    : VAR object with method obj.fit(data, p)
        % data   : double matrix (channels × time)
        % p_max  : integer, maximum VAR order
        % elbow  : logical
        %          false → strict AIC minimum
        %          true  → elbow of AIC curve
        %
        % Outputs
        % --------
        % p_opt    : selected VAR order
        % aic_vals : 1 × p_max vector of AIC values (p = 0…p_max)

            if nargin < 4
                elbow = false;
            end

            [n_ch, T] = size(data);
            aic_vals = nan(1, p_max);

            % Demean data (required for VAR likelihood)
            data = data - mean(data, 2);

            fprintf('\n p   Teff   rcond(Sigma)     logdet(Sigma)        AIC\n');
            fprintf('--------------------------------------------------------\n');

            % -------------------------------------------------
            % Compute AIC for each p
            % -------------------------------------------------
            for p = 1:p_max

                Teff = T - p;
                if Teff <= n_ch
                    fprintf('%2d   %5d   TOO_FEW_SAMPLES\n', p, Teff);
                    continue
                end

                try
                    % Fit VAR(p)
                    model = obj.fit(data, p);
                    Sigma = model.noise_cov;

                    % Conditioning diagnostic
                    rc = rcond(Sigma);

                    % -------- Safe log-determinant --------
                    % First try WITHOUT regularization
                    [R, flag] = chol(Sigma);

                    % If Cholesky fails, apply tiny diagonal regularization
                    if flag ~= 0 || rc < 1e-10
                        eps_reg = 1e-6 * trace(Sigma) / n_ch;
                        Sigma = Sigma + eps_reg * eye(n_ch);
                        [R, flag] = chol(Sigma);
                        if flag ~= 0
                            fprintf('%2d   %5d   %.2e   CHOL_FAIL\n', p, Teff, rc);
                            continue
                        end
                    end

                    logdetSigma = 2 * sum(log(diag(R)));

                    % -------- AIC computation --------
                    % Number of parameters
                    k = n_ch^2 * p + n_ch;

                    % Gaussian log-likelihood
                    logL = -Teff/2 * ( ...
                        n_ch*log(2*pi) + logdetSigma + n_ch );

                    % AIC
                    aic = -2*logL + 2*k;
                    if p > 0
                        aic_vals(p) = aic;
                    end

                    fprintf('%2d   %5d   %.2e   %+14.3f   %12.3f\n', ...
                            p, Teff, rc, logdetSigma, aic);

                catch ME
                    fprintf('%2d   ERROR: %s\n', p, ME.message);
                end
            end

            % -------------------------------------------------
            % Select optimal p
            % -------------------------------------------------
            % Strict AIC minimum
            if ~elbow
                [~, p_opt] = min(aic_vals);
                return;
            end

            % -------------------------------------------------
            % ELBOW SELECTION USING KNEEDLE ALGORITHM
            % -------------------------------------------------
            
            % Build x (orders) and y (AIC)
            x = 1:p_max;
            y = aic_vals;
            
            % Remove invalid points
            valid = isfinite(y);
            x_v = x(valid);
            y_v = y(valid);
            
            % Fallback if insufficient points
            if numel(x_v) < 3
                [~, p_opt] = min(aic_vals);
                return;
            end
            
            % -------- KNEEDLE ALGORITHM --------
            % 1) Flip AIC (since we minimize AIC)
            y_flip = -y_v;
            
            % 2) Normalize x and y to [0, 1]
            x_norm = (x_v - min(x_v)) / (max(x_v) - min(x_v));
            y_norm = (y_flip - min(y_flip)) / (max(y_flip) - min(y_flip));
            
            % 3) Compute distance from diagonal
            %    g(x) = y_norm - x_norm
            g = y_norm - x_norm;
            
            % 4) Knee = maximum deviation
            [~, idx_knee] = max(g);
            
            p_opt = x_v(idx_knee);
            
            % -------- Safety fallback --------
            if isempty(p_opt) || ~isfinite(p_opt)
                [~, p_opt] = min(aic_vals);
            end

        end



        %--------------------------------------------------
        % FIT
        %--------------------------------------------------
        function model = fit(obj, data, p)
            %FIT Fit VAR(p) model to multichannel data
            %
            % PARAMS
            %   obj  : microVARstates.VAR
            %
            %   data : numeric matrix (n_channels × n_timepoints)
            %          Input multivariate time series.
            %
            %   p    : integer
            %          VAR model order.
            %
            % RETURNS
            %   model : struct with fields
            %       .A         : cell array (1 × p)
            %                    Each cell is (n_channels × n_channels)
            %                    VAR coefficient matrices.
            %
            %       .noise_cov : numeric matrix (n_channels × n_channels)
            %                    Covariance of Gaussian innovations.
            %
            %       .p         : integer
            %                    Model order.

            if p == 0
                model.A = {};
                model.noise_cov = cov(data.');
                model.p = 0;
                return
            end

            [n_ch, T] = size(data);

            % Build regression matrices
            Y = data(:, p+1:T);
            X = [];

            for k = 1:p
                X = [X; data(:, p+1-k:T-k)];
            end

            % Least-squares estimation
            A = Y * X' / (X * X');

            % Residuals
            E = Y - A * X;
            Sigma = cov(E.');

            model.A = mat2cell(A, n_ch, repmat(n_ch, 1, p));
            model.noise_cov = Sigma;
            model.p = p;
        end

        %--------------------------------------------------
        % SIMULATE
        %--------------------------------------------------
        function data_sim = simulate(obj, model, T)
            %SIMULATE Generate synthetic time series from VAR model
            %
            % PARAMS
            %   obj   : microVARstates.VAR
            %
            %   model : struct
            %           Output of VAR.fit containing coefficients and noise.
            %
            %   T     : integer
            %           Number of time points to simulate.
            %
            % RETURNS
            %   data_sim : numeric matrix (n_channels × T)
            %              Simulated multivariate time series.

            n_ch = size(model.noise_cov, 1);
            p = model.p;

            data_sim = zeros(n_ch, T);

            % Initial conditions
            if p > 0
                data_sim(:, 1:p) = mvnrnd( ...
                    zeros(n_ch,1), model.noise_cov, p ).';
            end

            for t = p+1:T
                x = zeros(n_ch,1);
                for k = 1:p
                    x = x + model.A{k} * data_sim(:, t-k);
                end
                noise = mvnrnd(zeros(n_ch,1), model.noise_cov).';
                data_sim(:, t) = x + noise;
            end
        end

    end
end
