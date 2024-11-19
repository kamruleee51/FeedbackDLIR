% Description:
% This function computes three key cardiac metrics: End-Diastolic Volume (EDV), 
% End-Systolic Volume (ESV), and Ejection Fraction (EF) based on 3D segmentation masks
% of the left ventricle during end-diastolic (ED) and end-systolic (ES) phases.
%
% Inputs:
% - ED_mask: 3D binary segmentation mask for the left ventricle in the ED phase.
% - ES_mask: 3D binary segmentation mask for the left ventricle in the ES phase.
% - scale: Scaling factor for converting voxel counts to physical volume (e.g., mm³ to mL).
%
% Outputs:
% - EDV: End-Diastolic Volume (in mL), calculated from the ED mask.
% - ESV: End-Systolic Volume (in mL), calculated from the ES mask.
% - EF: Ejection Fraction (in %), calculated as:
%       EF = ((EDV - ESV) / EDV) * 100
%
% Notes:
% - The masks are assumed to contain specific labels for different regions. The label
%   for the left ventricle is assumed to be `2`, and all other labels are ignored.

function [EDV, ESV, EF] = Ejection_Fraction(ED_mask, ES_mask, scale)
    % Process ED mask: Retain only left ventricle region (label = 2)
    ED_ROI = ED_mask;
    ED_ROI(ED_ROI == 0) = 0; % Ensure background is 0
    ED_ROI(ED_ROI == 1) = 0; % Ignore other labels
    ED_ROI(ED_ROI == 2) = 1; % Keep left ventricle region only

    % Process ES mask: Retain only left ventricle region (label = 2)
    ES_ROI = ES_mask;
    ES_ROI(ES_ROI == 0) = 0; % Ensure background is 0
    ES_ROI(ES_ROI == 1) = 0; % Ignore other labels
    ES_ROI(ES_ROI == 2) = 1; % Keep left ventricle region only

    % Calculate volumes in mL
    EDV = scale * sum(ED_ROI(:)) / 1000; % End-Diastolic Volume
    ESV = scale * sum(ES_ROI(:)) / 1000; % End-Systolic Volume

    % Calculate Ejection Fraction
    EF = ((EDV - ESV) / EDV) * 100; % Ejection Fraction (%)
end
