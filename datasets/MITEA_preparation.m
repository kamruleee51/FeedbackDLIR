% Description:
% This script processes 3D cardiac MRI data for multiple patients. It performs resizing, 
% segmentation, and calculation of key cardiac volumes (EDV, ESV) and ejection fraction (EF).
% The processed data is saved in a structured format for further analysis or use in machine learning models.

% Requirements:
% 1. MATLAB environment with the following toolboxes:
%    - Image Processing Toolbox
% 2. Required helper functions:
%    - Ejection_Fraction: Custom function to compute EDV, ESV, and EF.
%    - make_nii, save_nii: Functions for handling NIfTI files.

% Input:
% - Directory containing patient MRI images and segmentation masks (`.nii.gz` format).
% - A CSV file with patient metadata (e.g., demographics).

% Output:
% - Resized images and masks saved in specified directories.
% - CSV file with extracted metrics (EDV, ESV, EF) for original and resized data.

% Initialize workspace
clc; close all; clear;

% Path to input data
dataDir = 'E:\ICL\Data from MITEA\mitea\images\';
labelDir = 'E:\ICL\Data from MITEA\mitea\labels\';
metaFilePath = 'E:\ICL\Data from MITEA\mitea\demographics.csv';

% Load patient data
All_Patients = dir(fullfile(dataDir, '*_ED.nii.gz'));
MetaFile = readtable(metaFilePath);
MetaFile = convertCharsToStrings([MetaFile.mitea_id, MetaFile.category]);

% Constants
LV = 2; % Label value for left ventricle
MYO = 1; % Label value for myocardium
SIZES = 64; % Resizing dimension

% Create directories for output
outputDir = num2str(SIZES);
mkdir(outputDir);
mkdir(fullfile(outputDir, 'mask'));
mkdir(fullfile(outputDir, 'image'));

% Initialize storage for volume metrics and metadata
volumes = zeros(length(All_Patients), 6); % Columns: [EDV, ESV, EF, resized EDV, ESV, EF]
Meta = cell(length(All_Patients), 2); % Columns: [PatientName, Condition]

% Process each patient
for pID = 1:length(All_Patients)
    % Display patient processing status
    disp(All_Patients(pID).name);

    % Extract patient ID
    PatientName = convertCharsToStrings(All_Patients(pID).name(1:9));
    disp(PatientName);

    % Find patient metadata
    index = find(strcmp(MetaFile(:, 1), PatientName));
    if isempty(index)
        disp('No metadata found for this patient.');
        continue;
    end
    disp(MetaFile(index, 2));

    % Store metadata
    Meta{pID, 1} = convertCharsToStrings(All_Patients(pID).name(1:15));
    Meta{pID, 2} = MetaFile(index, 2);

    % Load MRI images and segmentation masks
    ED_img = niftiread(fullfile(dataDir, All_Patients(pID).name));
    ES_img = niftiread(fullfile(dataDir, replace(All_Patients(pID).name, 'ED', 'ES')));
    ED_mask = niftiread(fullfile(labelDir, All_Patients(pID).name));
    ES_mask = niftiread(fullfile(labelDir, replace(All_Patients(pID).name, 'ED', 'ES')));

    % Compute original voxel dimensions
    voxelDimensions = [size(ED_mask, 1) / SIZES, size(ED_mask, 2) / SIZES, size(ED_mask, 3) / SIZES];

    % Calculate EDV, ESV, EF using original masks
    [EDV, ESV, EF] = Ejection_Fraction(ED_mask, ES_mask, 1.0*1.0*1.0);
    volumes(pID, 1:3) = [EDV, ESV, EF];

    % Resize images and masks
    ED_img_resized = imresize3(ED_img, [SIZES, SIZES, SIZES], 'cubic');
    ES_img_resized = imresize3(ES_img, [SIZES, SIZES, SIZES], 'cubic');
    ED_mask_resized = imresize3(ED_mask, [SIZES, SIZES, SIZES], 'nearest');
    ES_mask_resized = imresize3(ES_mask, [SIZES, SIZES, SIZES], 'nearest');

    % Calculate EDV, ESV, EF using resized masks
    [EDV_resized, ESV_resized, EF_resized] = Ejection_Fraction(ED_mask_resized, ES_mask_resized, prod(voxelDimensions));
    volumes(pID, 4:6) = [EDV_resized, ESV_resized, EF_resized];

    % Save resized images and masks as NIfTI
    ED_img_nii = make_nii(ED_img_resized, voxelDimensions);
    ES_img_nii = make_nii(ES_img_resized, voxelDimensions);
    ED_mask_nii = make_nii(ED_mask_resized, voxelDimensions);
    ES_mask_nii = make_nii(ES_mask_resized, voxelDimensions);

    save_nii(ED_img_nii, [num2str(SIZES) '\image\' All_Patients(pID).name(1:16) 'fixed.nii']);
    save_nii(ES_img_nii, [num2str(SIZES) '\image\' All_Patients(pID).name(1:16) 'mving.nii']);

    save_nii(ED_mask_nii, [num2str(SIZES) '\mask\' All_Patients(pID).name(1:16) 'fixed.nii']);
    save_nii(ES_mask_nii, [num2str(SIZES) '\mask\' All_Patients(pID).name(1:16) 'mving.nii']);

    disp('-------------------------------------------------------');

end

% Save extracted metrics to CSV
outputCSV = ['EDV_ESV_EF_MITEA_' num2str(SIZES) '.csv'];
writetable(array2table([Meta, num2cell(volumes)], ...
    'VariableNames', {'PatientName', 'Condition', 'EDV_original', 'ESV_original', 'EF_original', 'EDV_resize', 'ESV_resize', 'EF_resize'}), ...
    outputCSV);

disp('Processing complete.');
