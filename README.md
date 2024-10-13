# Simultaneous Projection and Linear Informed Training (SPLIT)

Code for the Master Thesis ["Leveraging Autoencoders to Enhance Model Order Reduction for Non-linear Mechanical Dynamical Systems"](https://resolver.tudelft.nl/uuid:ab828e4a-7ccb-436c-a7a8-2eefca77f437) by Aron Schouten.

The code is forked from [SindyAutoencoders](https://github.com/kpchamp/SindyAutoencoders).

The code contains results for four example problems. Three of them are inherited from the SINDy Autoencoders, and discussed in the paper ["Data-driven discovery of coordinates and governing equations"](https://arxiv.org/abs/1904.02107) by Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. One problem, based on a 2D cantilever beam, has been specifically designed for SPLIT, and is discussed in the Thesis ["Leveraging Autoencoders to Enhance Model Order Reduction for Non-linear Mechanical Dynamical Systems"](https://resolver.tudelft.nl/uuid:ab828e4a-7ccb-436c-a7a8-2eefca77f437). Code for each example can be found in the respective subfolder in the examples folder. For each example, there are jupyter notebooks for (1) running the training procedure and (2) analyzing the resulting models.

Creating the network architecture and running the training procedure requires the specification of several parameters.

A description of the parameters for SPLIT is as follows:
* `save_results` - boolean, whether or not to produce a summary output, saved in a Pandas dataframe
* `use_bias` - boolean, whether or not to include biases in network
* `include_constant` - boolean, whether or not to include a constant term in the SINDy library
* `linear_terms` - List of tuples containing (`index`, `coefficient`) of the a-priori known linear model

A description of the parameters inherited from the SINDy Autoencoders is as follows:
* `input_dim` - dimension of each sample of the input data
* `latent_dim` - dimension of the latent space
* `model_order` - either 1 or 2; determines whether the SINDy model predicts first or second order derivatives
* `poly_order` - maximum polynomial order to which to build the SINDy library; integer from 1-5
* `include_sine` - boolean, whether or not to include sine functions in the SINDy library
* `library_dim` - total number of library functions; this is determined based on the `latent_dim`, `model_order`, `poly_order`, and `include_sine` parameters and can be calculated using the function `library_side` in `sindy_utils.py`

* `sequential_thresholding` - boolean, whether or not to perform sequential thresholding on the SINDy coefficient matrix
* `coefficient_threshold` - float, minimum magnitude of coefficients to keep in the SINDy coefficient matrix when performing thresholding
*  `threshold_frequency` - integer, number of epochs after which to perform thresholding
* `coefficient_mask` - matrix of ones and zeros that determines which coefficients are still included in the SINDy model; typically initialized to all ones and will be modified by the sequential thresholding procedure
* `coefficient_initialization` - how to initialize the SINDy coefficient matrix; options are `'constant'` (initialize as all 1s), `'xavier'` (initialize using the xavier initialization approach), `'specified'` (pass in an additional parameter `init_coefficients` that has the values to use to initialize the SINDy coefficient matrix)

* `loss_weight_decoder` - float, weighting of the autoencoder reconstruction in the loss function (should keep this at 1.0 and adjust the other weightings proportionally)
* `loss_weight_sindy_z`- float, weighting of the SINDy prediction in the latent space in the loss function
* `loss_weight_sindy_x` - float, weighting of the SINDy prediction passed back to the input space in the loss function
* `loss_weight_sindy_regularization` - float, weighting of the L1 regularization on the SINDy coefficients in the loss function

* `activation` - activation function to be used in the network; options are `'sigmoid'`, `'relu'`, `'linear'`, or `'elu'`
* `widths` - list of ints specifying the number of units for each layer of the encoder; decoder widths will be the reverse order of these widths

* `epoch_size` - number of training samples in an epoch
* `batch_size` - number of samples to use in a batch of training
* `learning rate` - float; learning rate passed to the adam optimizer
* `data_path` - path specifying where to save the resulting models
* `print_progress` - boolean, whether or not to print updates during training
* `print_frequency` - print progress at intervals of this many epochs
* `max_epochs` - how many epochs to run the training procedure for
* `refinement_epochs` - how many epochs to run the refinement training for (see paper for description of the refinement procedure)
