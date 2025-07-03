# Cross-domain Hyperspectral Image Classification based on Bi-directional Domain Adaptation

# Abstract

Utilizing hyperspectral remote sensing technology enables the extraction of fine-grained land cover classes. Typically, satellite or airborne images used for training and testing are acquired from different regions or times, where the same class has significant spectral shifts in different scenes. In this paper, we propose a Bi-directional Domain Adaptation (BiDA) framework for cross-domain hyperspectral image (HSI) classification, which focuses on extracting both domain-invariant features and domain-specific information in the independent adaptive space, thereby enhancing the adaptability and separability to the target scene. In the proposed BiDA, a triple-branch transformer architecture (the source branch, target branch, and coupled branch) with semantic tokenizer is designed as the backbone. Specifically, the source branch and target branch independently learn the adaptive space of source and target domains, a Coupled Multi-head Cross-attention (CMCA) mechanism is developed in coupled branch for feature interaction and inter-domain correlation mining. Furthermore, a bi-directional distillation loss is designed to guide adaptive space learning using inter-domain correlation. Finally, we propose an Adaptive Reinforcement Strategy (ARS) to encourage the model to focus on specific generalized feature extraction within both source and target scenes in noise condition. Experimental results on cross-temporal/scene airborne and satellite datasets demonstrate that the proposed BiDA performs significantly better than some state-of-the-art domain adaptation approaches. In the cross-temporal tree species classification task, the proposed BiDA is more than 3%~5% higher than the most advanced method.
![Method](assets/BiDA.png "Method")


Paper: **coming soon**

Code: **coming soon**
