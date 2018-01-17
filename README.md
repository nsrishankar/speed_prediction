# Speed prediction in video input

Uses a 3D convolution network over 20 timesteps explained in "Learning Spatiotemporal Features with 3D Convolutional Networks". Initially, predictions are inaccurate, but becomes better after 9 minutes.  Better hyperparameter tuning or network choice is needed.

Video: [YouTube](https://www.youtube.com/watch?v=dqXhfM-s-Ko&list=PLMr_u-BsTKSr5Yl5eUxjsPM3TDsJP_KJQ&index=12&t=578s)  

## Improvements
- Use of a CNN-LSTM network to process semantic information in an image, and sequential information of subsequent frames (In process)
- Implementing an optical flow model
