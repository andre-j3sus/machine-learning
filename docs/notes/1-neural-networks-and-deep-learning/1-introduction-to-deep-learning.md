# Introduction to Deep Learning

> Deep Learning is a subfield of machine learning (ML is a subfield of AI) concerned with algorithms inspired by the structure and function of the brain called **artificial neural networks**. 

<p align="center">
  <img align="center" width="150" src="../../imgs/deep-learning.png" alt="Introduction to Deep Learning"/>
</p>

## Neural Networks

> A neural network is a computational model inspired by the structure and function of the brain. It is a collection of nodes called **neurons** that are connected together by **edges**.

* **Neurons** are the basic building blocks of a neural network. They receive input (x), perform some computation, and produce output (y);
* **Edges** are the connections between neurons. They convey information from one neuron to another;
* **Layers** are groups of neurons that perform a specific set of computations. They are organized in a hierarchy, with each layer feeding into the next layer.
* **Weights** are the parameters that define the strength of the connections between neurons. They are adjusted during the training process.
* **Activation functions** are used to introduce non-linearity into the network. They are applied to the output of each neuron and transform the output using a non-linear function.

<p align="center">
  <img align="center" width="600" src="https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg" alt="Neural Network"/>
</p>

> **Note**: **ReLU** stands for **Rectified Linear Unit**. It is a function that returns the maximum of 0 and the input value. It is used to introduce non-linearity in the network. It is a very common **activation function** in neural networks.

---

## Supervised Learning with Neural Networks

> Supervised learning is a type of machine learning where the goal is to **learn a function that maps an input to an output based on example input-output pairs**. It is called supervised learning because the process of learning the function is "supervised" by a set of example pairs.

### Neural Network Types

There are the following types of neural networks:

* **Standard NN**: a neural network with one or more hidden layers; used for classification and regression;
* **Convolutional NN**: a neural network that uses convolutional layers; used for image classification, for example;
* **Recurrent NN**: a neural network that uses recurrent layers; used for time series prediction, for example.

Some times, its necessary to use more than one type of neural network to solve a problem. For example, to solve problems related to images or radar information, it is common to use a combination of convolutional neural networks and recurrent neural networks.

<p align="center">
  <img align="center" width="600" src="https://vitalflux.com/wp-content/uploads/2021/11/deep-neural-network-examples.png" alt="Neural Network Types"/>
</p>

> **Note**: **Structured Data vs Unstructured Data**
> 
> **Structured data** is data that is **organized in a predefined format**. It is usually stored in a database or a spreadsheet. 
> 
> **Unstructured data** is data that is **not organized in a predefined format** (e.g. audio, image, text). It is usually stored in a file system or a document management system.

In this course, the variable `m` is used to denote the **number of training** examples.

> _Scale drives deep learning progress: with a long NN, with more labeled data, we have more performance_