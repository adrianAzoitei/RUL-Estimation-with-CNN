# RUL-Estimation-with-CNN

Tried the following so far:
- [x] Checked the data shape.
- [x] Checked the values (e.g. if they are normalized correctly).
- [x] Specifying Xavier as weight initializer at each layer.
- [x] Normalizing the labels (the RUL values) both for training and validating.
- [x] Overfitting on one training example.
- [ ] Concatenating the layers differently. Should be (N<sub>filters</sub>, N<sub>steps</sub>, N<sub>features</sub>) according to the paper.

## References
<a id="1">[1]</a> 
Li et. al. (2018). 
Remaining useful life estimation in prognostics using deep convolution neural networks. 
Reliability Engineering and System Safety 172 (2018) 1-11
