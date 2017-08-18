# cntk_resnet_fcn

This is a CNTK implementation of Fully Convolutional Network, which is a deep learning segmentation method proposed by J. Long et al. The FCN was originally proposed using VGG, but here we use ResNet-18 as the base model.

- FCN: [Fully Convolutional Networks for Semantic Segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

# Example Usage

Tested with cntk-2.1-gpu-python3.5 [docker](https://hub.docker.com/r/microsoft/cntk/)

```python
import numpy as np
import cntk_resnet_fcn
import simulation
%matplotlib inline
import helper

import cntk as C
from cntk.learners import learning_rate_schedule, UnitType
```

## Check some images/masks from simulation
```python
# Generate some random images
input_images, target_masks = simulation.generate_random_data(192, 192, count=3)

print(input_images.shape, target_masks.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [(x.swapaxes(0, 2).swapaxes(0,1).repeat(3, axis=2) * -255 + 255).astype(np.uint8) for x in input_images]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]

# Left: Input image, Right: Target mask
helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
```

    (3, 1, 192, 192) (3, 6, 192, 192)

### Left: Input image, Right: Target mask
![Images and masks from simulation](https://raw.githubusercontent.com/usuyama/cntk-fully-convolutional-networks/master/cntk_resnet_fcn_example/output_1_1.png)

## Prepare the resnet-fcn model

```python
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]

    return sx, sy

def measure_error(data_x, data_y, x, y, trainer, minibatch_size):
    errors = []
    for i in range(0, int(len(data_x) / minibatch_size)):
        data_sx, data_sy = slice_minibatch(data_x, data_y, i, minibatch_size)

        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)

def train(images, masks, use_existing=False):
    shape = input_images[0].shape
    data_size = input_images.shape[0]

    # Split data
    test_portion = int(data_size * 0.1)
    indices = np.random.permutation(data_size)
    test_indices = indices[:test_portion]
    training_indices = indices[test_portion:]

    test_data = (images[test_indices], masks[test_indices])
    training_data = (images[training_indices], masks[training_indices])

    # Create model
    x = C.input_variable(shape)
    y = C.input_variable(masks[0].shape)

    z = cntk_resnet_fcn.create_model(x, masks.shape[1])
    dice_coef = cntk_resnet_fcn.dice_coefficient(z, y)

    # Load the saved model if specified
    checkpoint_file = "cntk-resnet-fcn.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.0001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0.9)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 8
    num_epochs = 50

    training_errors = []
    test_errors = []

    for e in range(0, num_epochs):
        for i in range(0, int(len(training_data[0]) / minibatch_size)):
            data_x, data_y = slice_minibatch(training_data[0], training_data[1], i, minibatch_size)

            trainer.train_minibatch({x: data_x, y: data_y})

        # Measure training error
        training_error = measure_error(training_data[0], training_data[1], x, y, trainer, minibatch_size)
        training_errors.append(training_error)

        # Measure test error
        test_error = measure_error(test_data[0], test_data[1], x, y, trainer, minibatch_size)
        test_errors.append(test_error)

        print("epoch #{}: training_error={}, test_error={}".format(e, training_errors[-1], test_errors[-1]))

        trainer.save_checkpoint(checkpoint_file)

    return trainer, training_errors, test_errors
```

## Training

```python
input_images, target_masks = input_images, target_masks = simulation.generate_random_data(192, 192, count=1024)

trainer, training_errors, test_errors = train(input_images, target_masks)
```

    epoch #0: training_error=-0.017798021160390066, test_error=-0.018451113952323794
    epoch #1: training_error=-0.1391007762240327, test_error=-0.14523974185188612
    epoch #2: training_error=-0.3251049741454746, test_error=-0.3291884511709213
    epoch #3: training_error=-0.40855012069577756, test_error=-0.41476351022720337
    epoch #4: training_error=-0.44601072746774423, test_error=-0.4511391098300616
    epoch #5: training_error=-0.4810214545415795, test_error=-0.48489775508642197
    epoch #6: training_error=-0.5151172067808069, test_error=-0.5200231522321701
    epoch #7: training_error=-0.5922727802525396, test_error=-0.5973579933245977
    epoch #8: training_error=-0.749630199826282, test_error=-0.7541888852914175
    epoch #9: training_error=-0.7754635240720666, test_error=-0.778565322359403
    epoch #10: training_error=-0.8706006376639657, test_error=-0.8741355637709299
    epoch #11: training_error=-0.9253758440846982, test_error=-0.9278035958607992
    epoch #12: training_error=-0.9409363124681556, test_error=-0.943161795536677
    epoch #13: training_error=-0.9504859722178916, test_error=-0.9518442749977112
    epoch #14: training_error=-0.9561804066533628, test_error=-0.9564324915409088
    epoch #15: training_error=-0.9596312388129856, test_error=-0.958900640408198
    epoch #16: training_error=-0.9619116700213889, test_error=-0.9606296718120575
    epoch #17: training_error=-0.963296625925147, test_error=-0.9618712464968363
    epoch #18: training_error=-0.964468306562175, test_error=-0.962962140639623
    epoch #19: training_error=-0.9656051786049552, test_error=-0.9633625497420629
    epoch #20: training_error=-0.9661645360614942, test_error=-0.9637840042511622
    epoch #21: training_error=-0.9670840688373732, test_error=-0.9645407944917679
    epoch #22: training_error=-0.9675297908160998, test_error=-0.9647647142410278
    epoch #23: training_error=-0.968075982902361, test_error=-0.9654373526573181
    epoch #24: training_error=-0.9680173241573832, test_error=-0.9652755657831827
    epoch #25: training_error=-0.96848623752594, test_error=-0.9659683257341385
    epoch #26: training_error=-0.9682907306629679, test_error=-0.9664795845746994
    epoch #27: training_error=-0.9695260338161302, test_error=-0.9665666818618774
    epoch #28: training_error=-0.969839212168818, test_error=-0.9669433981180191
    epoch #29: training_error=-0.9700202615364738, test_error=-0.9667912622292837
    epoch #30: training_error=-0.9708342692126398, test_error=-0.9675299723943075
    epoch #31: training_error=-0.9703854773355567, test_error=-0.9673667003711065
    epoch #32: training_error=-0.9717840562696042, test_error=-0.9684023261070251
    epoch #33: training_error=-0.9726218985474628, test_error=-0.9691992004712423
    epoch #34: training_error=-0.9721553678097932, test_error=-0.9685578594605128
    epoch #35: training_error=-0.9730600165284198, test_error=-0.9691728303829829
    epoch #36: training_error=-0.9736596802006597, test_error=-0.9698172907034556
    epoch #37: training_error=-0.9731561370517896, test_error=-0.9691229710976282
    epoch #38: training_error=-0.9742445463719576, test_error=-0.9703827102979025
    epoch #39: training_error=-0.972710659192956, test_error=-0.9692197690407435
    epoch #40: training_error=-0.9743008660233539, test_error=-0.9704541166623434
    epoch #41: training_error=-0.9747222724168197, test_error=-0.9709257930517197
    epoch #42: training_error=-0.9754152588222338, test_error=-0.9714237848917643
    epoch #43: training_error=-0.9743199861567954, test_error=-0.9697967072327932
    epoch #44: training_error=-0.9753414858942446, test_error=-0.9713153938452402
    epoch #45: training_error=-0.9763206186501876, test_error=-0.9717517246802648
    epoch #46: training_error=-0.9767339353976042, test_error=-0.9718629717826843
    epoch #47: training_error=-0.972210144996643, test_error=-0.9703837434450785
    epoch #48: training_error=-0.9680927250696265, test_error=-0.967069461941719
    epoch #49: training_error=-0.9752375457597815, test_error=-0.9707983434200287

## Learning curve (Training/Test error)

```python
helper.plot_errors({"training": training_errors, "test": test_errors}, title="Simulation Learning Curve")
```

![Learning curve](https://raw.githubusercontent.com/usuyama/cntk-fully-convolutional-networks/master/cntk_resnet_fcn_example/output_4_0.png)

## Use the trained model

```python
# Generate some random images
input_images, target_masks = input_images, target_masks = simulation.generate_random_data(192, 192, count=10)

# Predict
pred = trainer.model.eval(input_images)

print(input_images.shape, target_masks.shape, pred.shape)
```

    (10, 1, 192, 192) (10, 6, 192, 192) (10, 6, 192, 192)


```python
# Change channel-order and make 3 channels for matplot
input_images_rgb = [(x.swapaxes(0, 2).swapaxes(0,1).repeat(3, axis=2) * -255 + 255).astype(np.uint8) for x in input_images]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

# Left: Input image, Middle: Correct mask (Ground-truth), Rigth: Predicted mask
helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
```

### Left: Input image, Middle: Correct mask (Ground-truth), Rigth: Predicted mask

![Predicted masks from the trained model](https://raw.githubusercontent.com/usuyama/cntk-fully-convolutional-networks/master/cntk_resnet_fcn_example/output_6_0.png)
