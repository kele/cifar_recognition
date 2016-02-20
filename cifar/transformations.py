import numpy as np
import random
import PIL
import PIL.ImageOps



def augment_data(image_batch, disturbers=None):
    if disturbers is None:
        disturbers = []

    new = []
    for i in range(len(image_batch)):
        data = image_batch[i]
        data = data.transpose(1, 2, 0)

        img = PIL.Image.fromarray(data)

        for d, p in disturbers:
            if random.random() <= p:
                img = d(img)

        #img = PIL.ImageOps.autocontrast(img)

        data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        data = data * 2.0 / 255.0 - 1.0
        new.append(data)

    return np.array(new)

#import data
#from matplotlib import pyplot
#
#d = data.load_datastream(100)
#
#import sys
#def display_samples():
#
#    def display_dataset():
#        fc = 0
#        for inputs, _ in d['train'].get_epoch_iterator():
#            fc += 1
#            pyplot.figure(fc)
#
#            for i in range(len(inputs)):
#                print inputs[i].shape
#                img = inputs[i].transpose(1, 2, 0)
#                img = inputs[i].transpose(2, 0, 1)
#                img = inputs[i].transpose(1, 2, 0)
#                shape = img.shape
#                print shape
#
#                import PIL.ImageOps
#                import PIL.Image
#
#                z = PIL.ImageOps.equalize(PIL.Image.fromarray(img))
#                y = np.array(z)
#
#                pyplot.imshow(y)
#                pyplot.show()
#
#                y = y.transpose(2, 0, 1)
#                print y.shape
#
#
#    display_dataset()
#
#display_samples()
