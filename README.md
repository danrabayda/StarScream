# NoiseNet: Airfield Noise Monitoring with Deep Acoustic Clustering

Continued exposure to aircraft noise is a persistent environmental issue with serious health concerns, including increased reliance on sleep aids and increased risk of heart disease, especially for communities living in close proximity to airfields ([Franssen 2004](https://oem.bmj.com/content/61/5/405); [Torija 2018](https://www.researchgate.net/profile/Antonio-Torija/publication/322328656_Aircraft_classification_for_efficient_modelling_of_environmental_noise_impact_of_aviation/links/5aaabc2845851517881b4434/Aircraft-classification-for-efficient-modelling-of-environmental-noise-impact-of-aviation.pdf)). To protect these communities, noise monitoring around airports remains a cruicial tool ([Asensio 2012](https://www.sciencedirect.com/science/article/abs/pii/S0003682X11002477)). 

Unfortunately, the automated identification of aircraft noise in residential environments can be challenging, due to the confounding presence of additional anthropogenic noise sources ([Tarabini 2014](https://www.sciencedirect.com/science/article/abs/pii/S0003682X1400070X)). To discriminate between these confounding sources, several techniques have been employed including acoustic classification ([Asensio 2010](https://oa.upm.es/7652/2/INVE_MEM_2010_80172.pdf)), noise coincidence with airfield radar tracks ([Timmerman 1991](https://asa.scitation.org/doi/10.1121/1.2029280)), and coincidence with ADS-B tracks ([Giladi 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7481859/pdf/main.pdf)).

In this work, we apply a deep neural network to perform feature extraction and unsupervised clustering on acoustic recordings at an airfield. We then provide a framework for rapidly classifying the clusters via human-machine teaming. Finally, we demonstrate a k-nearest-neighbors algorithm for the automated identification of aircraft noise vs other urban noise sources.

This work represents the first end-to-end discriminative airfield noise monitoring solution that requires only a single microphone with no external knowledge of airfield activities, significantly reducing the barrier to entry for accurate airfield noise monitoring. 

-Josh Dickey



## NN Architecture
We use a CNN that implements Triplet Semi-Hard Loss

To understand the Triplet Loss and what it is doing we need to explain the Generator just a bit so the Loss Function makes sense in our context. First we start with the long time-series data stream in raw data form. The job of the generator is to take one random "region" and take nK random spectrogram windows out of that region. We then take a total of nP regions and get out a matrix of spectrograms that is nP x nK = batch size. The actual data fed into the nn is a flattened list of these spectrograms with a labels list that label each spectrogram as the p row (region) it belongs to to give a total of nP labels. (the data shape is batch size x spectrogram height x spectrogram width, and the labels shape is just batch size)

The embeded vectors for all the spectrograms are then created by the neural network and output in the same order so we have an embedding vector list of batch size x output dimensions (which we set to 128). We then generate a distance matrix from all of these vectors and are left with a distance matrix of size batch size x batch size.

The distance matrix is obtained using the euclidian distance which for our case is really just the pythagorean theorem on the difference of vectors in multi dimensions so we treat the vectors as like 128 dimensional cartesian vectors and say **x** = **a** - **b** where **a** and **b** are 128 dimensional vectors and the resultant difference vector is just an element-wise subtraction in cartesian coordinates (x1=a1-b1, x2=a2-b2, etc). We then find the length of **x** by doing x1^2 + x2^2 + x3^2 + ... + x128^2 = dist^2 and we solve for dist^2 (for our purposes we don't really need to take the square root). 

The vectors **a** and **b** are chosen sequentially from our input list of vectors and done in order, we will define the function D(**a,b**) that does just this distance calculation between vector **a** and vector **b**. Let's do an example where we have a case that nP=2 and nK=2, then we will have a batch size of nP x nK = 4 and our y_pred (embeded vector list) will be of shape 4 x 128.

[**v1, v2, v3, v4**]

Given our input shape and the fact that nP=2 and nK=2 we know that we have 2 different classes, we will call them 0 and 1. Here are the labels (y_true)

y_true = [0, 0, 1, 1]

Now the distance matrix will be a 4 x 4 matrix and will look like this:

[D(**v1,v1**), D(**v1,v2**), D(**v1,v3**), D(**v1,v4**),<br>
 D(**v2,v1**), D(**v2,v2**), D(**v2,v3**), D(**v2,v4**),<br>
 D(**v3,v1**), D(**v3,v2**), D(**v3,v3**), D(**v3,v4**),<br>
 D(**v4,v1**), D(**v4,v2**), D(**v4,v3**), D(**v4,v4**)]

Before we continue it should be noted now that all the anchors are now defined. D(a,b) is the anchor-(pos/neg) pair where a is the anchor and b is either a positive or negative match for that anchor.

Based on our y_true labels that come from the generator, we can say that v1 and v2 should be positive to one another, and negative to v3 and v4. Below is the same distance matrix but this time we have highlighted every anchor-positive distance in green and every anchor-negative distance in red.

<span style="color:green">[D(**v1,v1**), D(**v1,v2**),</span> <span style="color:red">D(**v1,v3**), D(**v1,v4**),<span> <br>
 <span style="color:green">D(**v2,v1**), D(**v2,v2**),<span> <span style="color:red">D(**v2,v3**), D(**v2,v4**),<span> <br>
 <span style="color:red">D(**v3,v1**), D(**v3,v2**),<span> <span style="color:green">D(**v3,v3**), D(**v3,v4**),<span> <br>
 <span style="color:red">D(**v4,v1**), D(**v4,v2**),<span> <span style="color:green">D(**v4,v3**), D(**v4,v4**)]<span>
 

$ loss=max(||f(A)-f(P)||^2 - ||f(A)-f(N)||^2+a, 0) $

We have two different types of distances now $A-P: ||f(A)-f(P)||^2 and A-N: ||f(A)-f(N)||^2$ What we want is to increase the distance from the anchor to the negatives while simultaneously decreasing the distance between the anchor and positives. To do this we want to make our loss very high if the A-P distance is large (anchor is far from positive matches) or if the A-N distance is very small (anchor is close to negatives), or conversely to make loss low when A-P distance is small or the A-N distance is large. That is all the base function $||f(A)-f(P)||^2 - ||f(A)-f(N)||^2$ is doing.

Now we also have an added a margin that defines a 'hard' zone where we may have $||f(A)-f(P)||^2 < ||f(A)-f(N)||^2$ (the anchor is slightly closer to a negative than a positive). In Triplet semi-hard loss we take all these "close calls" within the margin a and we train based on those.

This means that we are taking all the possible differences from our distance matrix that are smaller than that value for a and we are using only those to compute loss for a given row. In Josh's custom loss he used the min positive and max negatives distances for each row to do this calculation but in reality you can take any number of them as long as theyre within the margin.

It should also be worth mentioning that the Tensorflow Triplet Semi-hard loss function does not require all the classes to be in order like we have set them up. we can input any list of spectrograms with any class order so like y_true=[1,1,5,0,5,0] would be fine, it would take these into account, but all our custom metrics depend on the placement so for our metrics we keep it so similar classes are together ([1,1,5,5,0,0]). 
