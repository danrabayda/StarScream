# NoiseNet: Airfield Noise Monitoring with Deep Acoustic Clustering
Joshua Dickey, Daniel Rabayda, Ryan Law, and Mitchell Solomon

Continued exposure to aircraft noise is a persistent environmental issue with serious health concerns, including increased reliance on sleep aids and increased risk of heart disease, especially for communities living in close proximity to airfields ([Franssen 2004](https://oem.bmj.com/content/61/5/405); [Torija 2018](https://www.researchgate.net/profile/Antonio-Torija/publication/322328656_Aircraft_classification_for_efficient_modelling_of_environmental_noise_impact_of_aviation/links/5aaabc2845851517881b4434/Aircraft-classification-for-efficient-modelling-of-environmental-noise-impact-of-aviation.pdf)). To protect these communities, noise monitoring around airports remains a cruicial tool ([Asensio 2012](https://www.sciencedirect.com/science/article/abs/pii/S0003682X11002477)). 

Unfortunately, the automated identification of aircraft noise in residential environments can be challenging, due to the confounding presence of additional anthropogenic noise sources ([Tarabini 2014](https://www.sciencedirect.com/science/article/abs/pii/S0003682X1400070X)). To discriminate between these confounding sources, several techniques have been employed including acoustic classification ([Asensio 2010](https://oa.upm.es/7652/2/INVE_MEM_2010_80172.pdf)), noise coincidence with airfield radar tracks ([Timmerman 1991](https://asa.scitation.org/doi/10.1121/1.2029280)), and coincidence with ADS-B tracks ([Giladi 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7481859/pdf/main.pdf)).

In this work, we apply a deep neural network to perform feature extraction and unsupervised clustering on acoustic recordings at an airfield. We then provide a framework for rapidly classifying the clusters via human-machine teaming. Finally, we demonstrate a k-nearest-neighbors algorithm for the automated identification of aircraft noise vs other urban noise sources.

This work represents the first end-to-end discriminative airfield noise monitoring solution that requires only a single microphone with no external knowledge of airfield activities, significantly reducing the barrier to entry for accurate airfield noise monitoring. 
