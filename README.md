# RUL-Estimation-with-CNN

Tried the following so far:
- [x] Checked the data shapes at different steps in the pipeline.
- [x] Checked the values (e.g. if they are normalized correctly).
- [x] Specifying Xavier as weight initializer at each layer.
- [x] Normalizing the labels (the RUL values) both for training and validating.
- [x] Overfitting on one training example.
- [x] Concatenating the layers differently. Should be (N<sub>filters</sub>, N<sub>steps</sub>, N<sub>features</sub>) according to the paper.
- [x] Fitting the model on dummy data -> it works, although the training is very sensitive to the window_size.
- [x] Normalizing the labels similarly to the training data (min-max norm).
- [x] Implementing step learning rate.
- [x] Implementing the piece-wise RUL function.
- [ ] Shuffle the windows.

## References
<a id="1">[1]</a> 
Li et. al. (2018). 
Remaining useful life estimation in prognostics using deep convolution neural networks. 
Reliability Engineering and System Safety 172 (2018) 1-11
