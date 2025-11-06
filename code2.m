clc; clear; close all;

% Parameters
max_iterations = 5;      % Maximum iterations
psnr_threshold = 0.5;    % Minimum PSNR improvement to keep changes

% Read original (old/degraded) image
img_orig = imread('testimg4.jpg');  
img_restored = img_orig;  % Initialize restored image

psnr_values = zeros(1, max_iterations);  % Track PSNR per iteration

for iter = 1:max_iterations
    % --- Preprocessing ---
    img_gray = double(rgb2gray(img_restored));  % For metric calculation
    mean_intensity = mean(img_gray(:));
    std_intensity  = std(img_gray(:));

    fprintf('\nIteration %d\n', iter);
    fprintf('Mean Intensity: %.2f (0-255 scale)\n', mean_intensity);
    fprintf('Standard Deviation of Intensity: %.2f\n', std_intensity);

    % Evaluate PSNR vs original
    if iter == 1
        psnr_values(iter) = psnr(img_restored, img_orig);
    else
        psnr_values(iter) = psnr(img_restored, prev_img);
    end
    prev_img = img_restored;

    % --- Filter Decision & Application ---
    img_temp = img_restored;

    % 1. Contrast stretching / color enhancement
    if std_intensity < 60
        R = imadjust(img_temp(:,:,1));
        G = imadjust(img_temp(:,:,2));
        B = imadjust(img_temp(:,:,3));
        img_temp = cat(3,R,G,B);
    end

    % 2. Gamma correction (brighten dark images)
    if mean_intensity < 120
        gamma_val = 0.8;
        img_temp = im2double(img_temp);
        img_temp(:,:,1) = img_temp(:,:,1).^gamma_val;
        img_temp(:,:,2) = img_temp(:,:,2).^gamma_val;
        img_temp(:,:,3) = img_temp(:,:,3).^gamma_val;
        img_temp = im2uint8(img_temp);
    end

    % 3. Noise detection (Salt & Pepper)
    sp_noise_ratio = sum(sum(img_gray==0 | img_gray==255)) / numel(img_gray);
    if sp_noise_ratio > 0.01
        R = medfilt2(img_temp(:,:,1));
        G = medfilt2(img_temp(:,:,2));
        B = medfilt2(img_temp(:,:,3));
        img_temp = cat(3,R,G,B);
    end

    % 4. Blur / Detail loss detection (using Laplacian variance)
    lap = double(imfilter(img_gray, fspecial('laplacian')));
    lap_var = var(lap(:));
    if lap_var < 50
        img_temp = imsharpen(img_temp,'Radius',1,'Amount',1);
    end

    % --- PSNR Check ---
    new_psnr = psnr(img_temp, img_orig);

    if iter == 1 || (new_psnr - psnr_values(max(iter-1,1)) >= psnr_threshold)
        img_restored = img_temp;
        psnr_values(iter) = new_psnr;
    else
        psnr_values(iter) = psnr_values(max(iter-1,1));
    end
end

%% --- Stage 2: Damage Detection ---
[mask, damage_pct] = detect_damage_mask(img_restored);

fprintf('Detected damage area: %.2f%% of image\n', damage_pct);

% --- Decision: Whether to Apply Inpainting ---
if damage_pct > 0.3 && damage_pct < 10
    disp('Moderate damage detected — applying inpainting...');
    apply_inpaint = true;
elseif damage_pct >= 10
    disp('Severe damage detected — skipping automated inpainting (manual review suggested).');
    apply_inpaint = false;
else
    disp('Minimal damage detected — skipping inpainting.');
    apply_inpaint = false;
end

%% --- Stage 3: Conditional Inpainting ---
if apply_inpaint
    img_inpainted = img_restored;
    for c = 1:3
        img_inpainted(:,:,c) = regionfill(img_restored(:,:,c), mask);
    end
    img_inpainted = imclose(img_inpainted, strel('disk',1));
else
    img_inpainted = img_restored;
end

%% --- Display Results ---
figure;
imshowpair(img_orig, img_inpainted, 'montage');
title('Original (left) vs Final Restored (right)');

%% --- PSNR vs Iteration Plot ---
figure;
plot(1:max_iterations, psnr_values, '-o','LineWidth',1.5);
xlabel('Iteration'); ylabel('PSNR');
title('PSNR vs Iteration');
grid on;

%% --- Helper Function: Damage Mask Detection ---
function [mask, damage_pct] = detect_damage_mask(img)
    gray = rgb2gray(img);
    edges = edge(gray, 'canny', [0.05 0.2]);

    mask = imclose(edges, strel('line',3,0));
    mask = imdilate(mask, strel('disk',1));
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 100);

    figure; imshow(mask); title('Detected Damage Mask');
    damage_pct = 100 * sum(mask(:)) / numel(mask);
    fprintf('Estimated damage area: %.2f%%\n', damage_pct);
end
