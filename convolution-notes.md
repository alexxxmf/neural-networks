When applying a filter to an image, the result will have a smaller dimension compared to the input image so we are losing some information in the process.
If the architecture of the network involves loads of layers we might need to avoid this so hence we use padding.
Without padding we can see that the pixels on the borders of the image are barely used for a convolution operation compared to central ones (ie, 1 time vs n^2 times being n dimenion of the filter)
So in a way we use padding to avoid losing information related to the edges of the image.

Why we normally use odd numbers for filter dimensions?

- That way the filter has a center pixel.
- We prevent using different amount of padding for keeping the dimension of an image. p = (f - 1) / 2 when we want to preserve the image so this means for an even number p = 3/2 which means we will need to pixels on one side and 1 pixel on the other side

The stride is the parameter that controls how big is the step to take when convolving, by default we tend to move the filter one step

Max pooling is a way to down-sample an image applied after convolution so it keeps the most important features coming from that applied filter
less size + keeping most important features = easier to compute w/o losing too much info
