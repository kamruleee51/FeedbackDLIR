## Feedback Attention to Enhance Unsupervised DL Image Registration in 3D Echocardiography

<p style="text-align: justify;">
Cardiac motion estimation in 3D echocardiography provides critical insights into heart health but remains challenging due to complex geometry and limited 3D DLIR implementations. We propose a <strong>spatial feedback attention module (FBA)</strong> to enhance unsupervised 3D deep learning image registration (DLIR) (as shown in the figure below). FBA leverages initial registration results to generate co-attention maps of residual errors, improving self-supervision by minimizing these errors.
</p>

<p align="justify">
</p>
<p align="center">
<img width="1000" alt="DLIR_model" src="https://github.com/user-attachments/assets/59062d56-abcb-4967-81ee-d26a4e03c33d">
</p>


<p style="text-align: justify;">
FBA improves various 3D DLIR designs, including transformer-enhanced networks, and performs well on both fetal and adult 3D echocardiography. Combining FBA with a spatial transformer and attention-modified backbone achieves state-of-the-art results, highlighting the effectiveness of spatial attention in scaling DLIR from 2D to 3D.
</p>


We utilized two distinct echocardiographic datasets for our study: the publicly available [**3D MITEA dataset**](https://www.cardiacatlas.org/mitea/) and a proprietary in-house **3D fetal dataset**. The source code provided in this repository is designed to be reproducible using the **3D MITEA dataset**, which can be accessed through the provided link.

We would like to express our gratitude to the following repository, which was utilized as a reference for our work:
- [COSNet](https://github.com/carrierlxk/COSNet)  


