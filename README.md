# recursive_mlp

Python: 3.10.9

Multi-Layer perceptron with customable number of neurons and layers and customable activation functions

The function add allows to add layers to the MLP. Add requires the dimension of the matrix it is holding, and an object Parameters. Each layer is a node, and the MLP is structered around a linked list. When performing fit or predict, the MLP iterates recursively through its nodes to update the weights or predict values.

The class Parameters holds onto the chosen activation function, its derivative and the derivative of loss function in respect to the activation function respectively.


