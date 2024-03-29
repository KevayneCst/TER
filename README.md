# TER

## Installation

### Dependencies

#### EvData code

The code uses functions from [this](https://github.com/amygruel/EvData) repository, made by @amygruel.
Once her repository is cloned, please modify :
- The link [here](https://github.com/KevayneCst/TER/blob/main/delay_learning_v3.py#L14) by the absolute link to where you have put the "translate_2_formats " folder. (This folder is inside the EvData repository)
- The link [here](https://github.com/amygruel/EvData/blob/master/translate_2_formats/events2spikes.py#L19) by the absolute link to where you have put the "read_event_data" folder. (This folder is inside the EvData repository)

#### Other packages

To facilitate the installation, we invite you to use the provided conda environment. You can download conda [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

In a terminal, run the following command : 

`conda env create -f ter_conda_env.yml`

### Run the program

In a terminal, once in the folder containing the script code <i>delay_learning_v3.py</i>, run the following command by replacing <i>[NUMBER_OF_CONVOLUTION]</i> by a number (preferably 2, 4 or 8) : 

`python3 delay_learning_v3.py nest --plot-figure --nb-convolution [NUMBER_OF_CONVOLUTION]`


## Description

Spiking Neural Networks (SNN) represent a special class of artificial neural networks, where neurons communicate by sequences of spikes [Ponulak, 2011]. Contrary to deep convolutional networks, spiking neurons do not fire at each propagation cycle, but rather fire only when their activation level (or membrane potential, an intrinsic quality of the neuron related to its membrane electrical charge) reaches a specific threshold value. Therefore, the network is asynchronous and allegedly likely to handle well temporal data such as video. When a neuron fires, it generates a non-binary signal that travels to other neurons, which in turn increases their potentials. The activation level either increases with incoming spikes, or decays over time. Regarding inference, SNN does not rely on stochastic gradient descent and backpropagation. Instead, neurons are connected through synapses, that implement learning mechanisms inspired from biology for updating synaptic weights (strength of connections) or delays (propagation time for an action potential).
The development of event-based cameras, inspired by the retina, fosters the application of SNN to computer vision. Instead of measuring the intensity of every pixel in a fixed time interval like standard cameras, they report events of significant pixel intensity changes. Every such event is represented by its position, sign of change, and timestamp -- accurate to the microsecond. Due to their asynchronous course of operation, considering the precise occurrence of spikes, Spiking Neural Networks are a natural match for event-based cameras. State-of-the-art approaches in machine learning provide excellent results with standard cameras, however, asynchronous event sequences require special handling, and spiking networks can take advantage of this asynchrony.
In this project, we wish to extend a previous project in which we designed a specific elementary SNN inspired by [Nadafian, 2020] and tuned synaptic delays to recognise prototypical temporal patterns from event cameras, inspired by Reichardt detectors. The architecture is currently implemented on the Python CPU simulator PyNN [Davison, 2009].
The objective of this work is separated into two parts:
1) apply this model to real-world data in order to recognize features such as motion direction and speed ;
2) adapt this architecture to run on the Human Brain Project SpiNNaker, a novel neuromorphic hardware simulating SNN at high speed and low cost [Furber, 2020].
At the end of the project, the candidate might be offered an internship. This project takes place within the EU programme APROVIS3D (http://www.chistera.eu/projects/aprovis3d) which started in April 2020, and that targets embedded bio-inspired machine learning and computer vision, with an application to autonomous drone navigation.
