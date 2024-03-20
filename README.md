# recursive_mlp

Multi-Layer perceptron with customable number of neurons and layers and customable activation functions

The function add allows to add layers to the MLP. Add requires the dimension of the matrix it is holding, and an object Parameters. Each layer is a node, and the MLP is structered along a linked list. It means that the functions fit and predict pass through the list recursively.

The class Parameters holds onto the chosen activation function, its derivative and the derivative of loss function in respect to the activation function respectively.

For now the MLP is only customable for binary classification
