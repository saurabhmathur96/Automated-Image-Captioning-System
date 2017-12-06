# Automated Image Captioning System
My undergraduate capstone project. A deep learning based image caption generator.


## Abstract

Many forms of media frequently lack alternative text for images. This lack of image
captions hampers the accessibility of their content. Hiring people to write captions
for those pictures is often prohibitively expensive. The use of an automated system to
write the captions would be a viable alternative. The objective of the project is to
develop such a system which can generate descriptive captions for a given image.


Advances in Computer Vision have led to the development of Artificial Neural
Network (ANN) based feature extractors like the Convolutional Neural Network
(CNN). These feature extractors can convert digital images into rich high-level
representations using a statistical model previously learned from labeled data.
Additionally, ANNs can be modified to model sequences of symbols like sentences.
One such type of ANN is the Recurrent Neural Network (RNN).


The proposed implementation of the image captioning system would feed the
features extracted by a CNN to an RNN. The RNN would generate a caption based on
the previously obtained image features.

## Project Phases

This project will follow the phases similar to the [Team Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview).


![Team Data Science Process Lifecycle](./images/lifecycle.png)


![Team Data Science Process Tasks](./images/tasks.png)