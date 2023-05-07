# BCI_Smart_Home
BCI System to control a Smart Home System
This Project was created in the scope of a Master Thesis.

For the data aquisition the Cyton Biosensing board from OpenBCI is used.
Following the used components for the aquisition proccess are shown

![Hardware_used](https://user-images.githubusercontent.com/57944502/236663596-e4370f7c-38d2-40b1-9dff-7ed441f2aa3e.jpeg)


The data is cleaned using the z-score and the 50Hz mains frequency is filtered from the data.

For the feature extraction the Descrete Wavelet Trasnformation is used.

Four different Classifiers are used LDA, MLS, CNN and SVM

The Models where tested using 4 labels as well as 2 labels.
The 4 labels where "light on", "light off", "light bright", "light dim".
When using only 2 labels the "light on" and "light off" labels are used.

MLS showed the best perfomances while CNN the worst ones.
Using 4 labels an accuracy of 78% is achieved.
Using 2 labels an accuracy of 86% is achieved.
These results are achieved using dataset eeg_data_1_method_1.
Following the Confusion Matrix for both results are demonstrated:

![MLP_2_400_5000_logistic_dataset1](https://user-images.githubusercontent.com/57944502/236664449-355e1840-c535-4f6c-af5e-c6d65477528b.png)

Confusion Matrix using 4 Labels and MLP Model with Sigmoid as an Activations function, 2 hidden layers, max_iter of 5000 and 400 Neurons per layer.

![MLP_2_400_2000_logistic_dataset1](https://user-images.githubusercontent.com/57944502/236664509-feb8b5fa-72f3-4098-8d3d-964359acaaf9.png)

Confusion Matrix using 2 Labels and MLP Model with Sigmoid as an Activations function, 2 hidden layers, max_iter of 2000 and 400 Neurons per layer.


