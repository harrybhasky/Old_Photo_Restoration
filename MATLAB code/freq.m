%% --- Color Image Restoration + Frequency Enhancement Pipeline ---
% Team Past Pixels
% K Hari Bhaskaran, Rohit R, Vedant Suhas Rane
% Midterm Report Version (Oct 2025)

clc; clear; close all;

%% --- Parameters ---
max_iterations = 4;      % Number of main processing stages (informational)
psnr_threshold = 0.3;    % not used directly here but kept
use_freq_domain = true;  % Toggle frequency-domain enhancement

%% --- Load Input Image ---
img_file = 'testimg4.jpg';
try
    img_orig = imread(img_file);
catch
    error('Error: Image file not found. Please check the filename: %s', img_file);
end

% Work in double [0,1]
img_orig = im2double(img_orig);
img_current = img_orig;   % will be updated stage-by-stage
fprintf('Loaded image of size %dx%dx%d\n', size(img_orig));

%% --- Analysis (grayscale stats) ---
img_gray = rgb2gray(img_orig);
mean_intensity = mean(img_gray(:)) * 255;
std_intensity  = std(img_gray(:)) * 255;
fprintf('Mean Intensity: %.2f, Std Dev: %.2f\n', mean_intensity, std_intensity);

%% Containers to store outputs and metrics
stages = {}; labels = {};
psnr_vs_orig = []; ssim_vs_orig = [];

%% --- Stage 1: CLAHE (on luminance) ---
labels{end+1} = 'CLAHE';
img_ycbcr = rgb2ycbcr(img_current);
Y = img_ycbcr(:,:,1);
Y_eq = adapthisteq(Y, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
img_ycbcr(:,:,1) = Y_eq;
img_clahe = ycbcr2rgb(img_ycbcr);
stages{end+1} = img_clahe;

% Metrics vs original
psnr_vs_orig(end+1) = psnr(img_clahe, img_orig);
ssim_vs_orig(end+1) = ssim(rgb2gray(img_clahe), rgb2gray(img_orig));

fprintf('Stage 1 (CLAHE): PSNR %.2f dB | SSIM %.4f\n', psnr_vs_orig(end), ssim_vs_orig(end));

% show result
figure; imshowpair(img_current, img_clahe, 'montage');
title('Stage 1: Before (Left) vs After CLAHE (Right)');

% update current
img_current = img_clahe;

%% --- Stage 2: Gamma Correction (Adaptive) ---
labels{end+1} = 'GammaCorr';
% gamma formula: keep it clamped to a reasonable range
gamma_val = 1.2 - 0.001 * (mean_intensity - 100);
gamma_val = max(0.5, min(2.0, gamma_val)); % clamp
img_gamma = imadjust(img_current, [], [], gamma_val);

stages{end+1} = img_gamma;
psnr_vs_orig(end+1) = psnr(img_gamma, img_orig);
ssim_vs_orig(end+1) = ssim(rgb2gray(img_gamma), rgb2gray(img_orig));

fprintf('Stage 2 (Gamma %.3f): PSNR %.2f dB | SSIM %.4f\n', gamma_val, psnr_vs_orig(end), ssim_vs_orig(end));

figure; imshowpair(img_current, img_gamma, 'montage');
title(sprintf('Stage 2: Before (Left) vs After Gamma=%.3f (Right)', gamma_val));

img_current = img_gamma;

%% --- Stage 3: Median Filtering (Noise Reduction) ---
labels{end+1} = 'Median';
img_med = img_current;
for c = 1:3
    img_med(:,:,c) = medfilt2(img_current(:,:,c), [3 3]);
end

stages{end+1} = img_med;
psnr_vs_orig(end+1) = psnr(img_med, img_orig);
ssim_vs_orig(end+1) = ssim(rgb2gray(img_med), rgb2gray(img_orig));

fprintf('Stage 3 (Median 3x3): PSNR %.2f dB | SSIM %.4f\n', psnr_vs_orig(end), ssim_vs_orig(end));

figure; imshowpair(img_current, img_med, 'montage');
title('Stage 3: Before (Left) vs After Median Filter (Right)');

img_current = img_med;

%% --- Stage 4: Inpainting (Structure Restoration) ---
labels{end+1} = 'Inpaint';
fprintf('\n--- Stage 4: Large Damage Inpainting ---\n');

% Build mask from edges and bright/dark anomalies
img_gray_med = rgb2gray(img_current);
edges_mask = edge(img_gray_med, 'sobel');
% expand/clean mask a bit
mask = imdilate(edges_mask, strel('disk',2));
mask = imfill(mask, 'holes');
mask = bwareaopen(mask, 50);

% visualize mask
figure; imshow(mask); title('Stage 4: Detected Mask');

% perform regionfill on grayscale and recompose
img_inpaint_gray = regionfill(img_gray_med, mask);
img_ycbcr = rgb2ycbcr(img_current);
% ensure the inpainted gray is in [0,1] scale already
img_ycbcr(:,:,1) = img_inpaint_gray;
img_inpaint = ycbcr2rgb(img_ycbcr);

stages{end+1} = img_inpaint;
psnr_vs_orig(end+1) = psnr(img_inpaint, img_orig);
ssim_vs_orig(end+1) = ssim(rgb2gray(img_inpaint), rgb2gray(img_orig));

fprintf('Stage 4 (Inpainting): PSNR %.2f dB | SSIM %.4f\n', psnr_vs_orig(end), ssim_vs_orig(end));

figure; imshowpair(img_current, img_inpaint, 'montage');
title('Stage 4: Before (Left) vs After Inpainting (Right)');

img_current = img_inpaint;

%% --- Optional Stage 5: Frequency Domain Enhancement ---
if use_freq_domain
    labels{end+1} = 'FreqEnh';
    fprintf('\nStage 5 (Frequency Enhancement): Applying High-Boost Filter...\n');

    % use current image luminance
    img_gray_curr = rgb2gray(img_current);
    [M,N] = size(img_gray_curr);
    F = fft2(img_gray_curr);
    F_shift = fftshift(F);

    D0 = 40;  % cutoff — tweak as needed
    [U, V] = meshgrid(1:N, 1:M);
    D = sqrt((U - N/2).^2 + (V - M/2).^2);
    H = 1 + 0.5 * (1 - exp(-(D.^2) / (2 * D0^2))); % high-boost-like

    F_filtered = H .* F_shift;
    img_freq = real(ifft2(ifftshift(F_filtered)));
    img_freq = mat2gray(img_freq); % normalize to [0,1]

    % Merge enhanced luminance with current chroma in YCbCr
    img_ycbcr = rgb2ycbcr(img_current);
    img_ycbcr(:,:,1) = img_freq;
    img_freq_enhanced = ycbcr2rgb(img_ycbcr);

    % Ensure numeric bounds
    img_freq_enhanced = max(0, min(1, img_freq_enhanced));

    stages{end+1} = img_freq_enhanced;
    psnr_vs_orig(end+1) = psnr(img_freq_enhanced, img_orig);
    ssim_vs_orig(end+1) = ssim(rgb2gray(img_freq_enhanced), rgb2gray(img_orig));

    fprintf('Stage 5 (Freq Domain): PSNR %.2f dB | SSIM %.4f\n', psnr_vs_orig(end), ssim_vs_orig(end));

    figure; imshowpair(img_current, img_freq_enhanced, 'montage');
    title('Stage 5: Before (Left) vs After Frequency Enhancement (Right)');

    img_current = img_freq_enhanced;
end

%% --- Final restored image ---
img_restored = img_current;

figure('Name', 'Image Restoration Pipeline Results', 'NumberTitle', 'off');
subplot(1,2,1); imshow(img_orig); title('Original Image');
subplot(1,2,2); imshow(img_restored); title('Restored Image');

%% --- Summary Table (PSNR & SSIM vs Original) ---
fprintf('\n--- Summary of PSNR & SSIM (all vs ORIGINAL) ---\n');
fprintf('Stage\t\tOperation\t\tPSNR (dB)\tSSIM\n');
for i = 1:numel(stages)
    fprintf('%d\t\t%-12s\t\t%.2f\t\t%.4f\n', i, labels{i}, psnr_vs_orig(i), ssim_vs_orig(i));
end

fprintf('\nFinal restoration completed successfully.\n');
