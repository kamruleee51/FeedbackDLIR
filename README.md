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


### Optimal Configuration of FBA DLIR

**Description:**  
<p style="text-align: justify;">
Fig. 2 illustrates the optimal configuration identified for our <strong>FBA DLIR</strong> framework. This configuration integrates the <strong>FBA module</strong> (denoted as Block C) with an existing transformer-based DLIR network. The modularity of this design allows for the removal or replacement of various components, enabling the exploration of alternative DLIR network architectures.
</p>

<p align="justify">
</p>
<p align="center">
<img width="1000" alt="FBA_DLIR" src="https://github.com/user-attachments/assets/71652ab5-bd2e-45f5-a93c-5570eb8a8d54">
</p>

**Key Points:**  
- **Block C:** Represents the **FBA module**.  
- The figure demonstrates how the FBA module can be incorporated into different **3D CNN** and **Transformer-based DLIR networks**.  
- This modular approach allows us to systematically evaluate the impact of the FBA module on registration performance across diverse architectures.  

By conducting this analysis, we aim to assess whether the inclusion of the FBA module offers consistent improvements in the performance of DLIR networks.


We utilized two distinct echocardiographic datasets for our study: the publicly available [**3D MITEA dataset**](https://www.cardiacatlas.org/mitea/) and a proprietary in-house **3D fetal dataset**. The source code provided in this repository is designed to be reproducible using the **3D MITEA dataset**, which can be accessed through the provided link.

We would like to express our gratitude to the following repository, which was utilized as a reference for our work:
- [COSNet](https://github.com/carrierlxk/COSNet)

## **Details coming soon!**


