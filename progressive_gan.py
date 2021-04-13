#%%

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from progressbar import progressbar
import time
from functools import reduce
import os
from PIL import Image
import random
import gc

#%%

final_img_dim = (128, 128)


class BetterMaxUnpool(nn.Module):
    def __init__(self, shape, output_shape):
        super(BetterMaxUnpool, self).__init__()
        self.shape = shape
        # self.unpool = nn.MaxUnpool2d(shape)
        self.output_shape = output_shape

    def forward(self, input):
        output = torch.zeros([input.shape[0], input.shape[1], input.shape[2], self.shape, input.shape[3], self.shape]).to(device)
        output[:,:,:,0,:,0] = input
        output = torch.reshape(output, [input.shape[0], input.shape[1], input.shape[2] * self.shape, input.shape[3] * self.shape])
        if not self.output_shape[0] - output.shape[2] < self.shape:
            print(input.shape)
            print(self.output_shape)
            print(output.shape)
            raise Exception('output_shape[0] wrong')
        if not self.output_shape[1] - output.shape[3] < self.shape:
            print(input.shape)
            print(self.output_shape)
            print(output.shape)
            raise Exception('output_shape[1] wrong')
        output = output[:,:,:self.output_shape[0],:self.output_shape[1]]
        return output

def conv(in_channel, out_channel, kernel_size, pooling, dropout=0.):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size//2),
        pooling,
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout)
    )

def deconv(in_channel, out_channel, kernel_size, upsample, dropout=0.):
    return nn.Sequential(
        upsample,
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size//2),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(out_channel),
        #nn.Dropout(dropout)
    )


#%%

class CNNDecoder(nn.Module):
    def __init__(self, output_shape, latent_size, kernel_size, channels, upsampling, use_sigmoid=False, dropout=0.):
        super(CNNDecoder, self).__init__()

        kernel_size = list(kernel_size)
        channels = list(channels)
        upsampling = list(upsampling)

        upsampling_shapes = [np.array(output_shape)]
        #upsampling_shapes = []

        #current_shape = upsampling_shapes[0]
        current_shape = np.array(output_shape)
        for u, k in zip(reversed(upsampling), reversed(kernel_size[1:-1])):
            current_shape = (current_shape) // u# + 2 * k // 2
            # current_shape = np.ceil(float_shape)
            upsampling_shapes.append(current_shape)

        # current_shape = current_shape - kernel_size[0] + 1

        self.latent_size = latent_size


        print(self.latent_size)

        upsampling_shapes.reverse()

        self.upsamples = nn.ModuleList([nn.Upsample(tuple(upsampling_shape.astype(int))) for upsampling_shape in upsampling_shapes])

        # self.conv_in_shape = [channels[0]] + list(current_shape.astype(int))
        self.output_shape = output_shape
        self.use_sigmoid = use_sigmoid

        self.deconvs = [deconv(channels[i], channels[i+1], kernel_size=k, upsample=u, dropout=dropout)
                                      for i, (k, u)
                                      in enumerate(zip(kernel_size[1:], self.upsamples))]
        self.deconvs.insert(0, nn.Sequential(
                                        nn.ConvTranspose2d(self.latent_size, channels[0], kernel_size=kernel_size[0]),
                                        nn.LeakyReLU(0.2),
                                        nn.BatchNorm2d(channels[0]),
                                        #nn.Dropout(dropout)
                                    ))

        self.deconvs = nn.ModuleList(self.deconvs)

        # self.deconv = nn.Sequential(*deconvs)

        self.to_rgb = nn.Sequential(nn.Conv2d(channels[-1], 3, kernel_size=1),
                                    nn.Sigmoid())

    def forward(self, X, use_layers=None, alpha=1):
        i = torch.reshape(X, [X.shape[0], X.shape[1], 1, 1])
        if alpha == 1:
            deconv = nn.Sequential(*self.deconvs[:use_layers])
            o = deconv(i)
            rgb = self.to_rgb(o)
        elif alpha < 1:
            deconv_old = nn.Sequential(*self.deconvs[:use_layers - 1])
            deconv_added = self.deconvs[use_layers - 1]
            o_old = deconv_old(i)
            o_new = deconv_added(o_old)
            o_old = self.upsamples[use_layers - 2](o_old)
            rgb_old = self.to_rgb(o_old)
            rgb_new = self.to_rgb(o_new)
            rgb = (1-alpha) * rgb_old + alpha * rgb_new
        else:
            raise Exception("Alpha not legal")
        #if self.use_sigmoid:
        #    rgb = torch.sigmoid(rgb)
        return rgb

#%%

class AdverserialTest(nn.Module):
    def __init__(self, image_size=None, kernel_size=None, channels=None, pooling=None):
        super(AdverserialTest, self).__init__()
        self.from_rgb = nn.Conv2d(3, channels[0], kernel_size=1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(p) for p in pooling])
        self.convs = nn.ModuleList([conv(channels[i], channels[i+1], kernel_size=k, pooling=p, dropout=0.4)
                                    for i, (k, p)
                                    in enumerate(zip(kernel_size[:-1], self.poolings))])
        self.convs.append(nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_size[-1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        ))

        # self.convolutional = nn.Sequential(*convs)
        img_dummy = torch.zeros([1, 3] + image_size)
        conv_in_dummy = self.from_rgb(img_dummy)
        conv_out_size = nn.Sequential(*self.convs)(conv_in_dummy).shape[1:]

        linear_in_size = reduce(lambda a,b: a*b, conv_out_size)
        self.classifier = nn.Linear(linear_in_size, 1)

    def forward(self, X, use_layers=None, alpha=1):
        if alpha == 1:
            inp = self.from_rgb(X)
            convolutional = nn.Sequential(*self.convs[-use_layers:])
            conv = convolutional(inp)
        elif alpha < 1:
            new_inp = self.from_rgb(X)
            old_inp = self.from_rgb(self.poolings[-use_layers + 1](X))
            old_convolutional = nn.Sequential(*self.convs[-use_layers + 1:])
            new_convolutional = self.convs[-use_layers]
            new_conv = new_convolutional(new_inp)
            weighted_avg = (1 - alpha) * old_inp + alpha * new_conv
            conv = old_convolutional(weighted_avg)
        else:
            raise Exception('Invalid alpha')
        conv_flattened = conv.flatten(1, -1)
        # conv_flattened = nn.Dropout(0.4)(conv_flattened)
        out = self.classifier(conv_flattened)
        return out

#%%

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.decoder = generator
        self.test = discriminator

    def forward(self, X, use_layers=None, alpha=1):
        latent_sample = torch.normal(0,1, size=(X.shape[0], self.decoder.latent_size)).to(device)
        decoder_samples = self.decoder(latent_sample, use_layers, alpha)
        # test_samples = torch.cat([X, decoder_samples], dim=0)
        # test_scores = self.test(test_samples)
        # real_scores, generator_scores = test_scores[:X.shape[0]], test_scores[X.shape[0]:]
        real_scores = self.test(X, use_layers, alpha)
        generator_scores = self.test(decoder_samples, use_layers, alpha)
        return real_scores, generator_scores

#%%

def load_images(shuffle=True, use_progressbar=False, use_only=None, filenames=None, skip_blackscreen=True):
    if filenames is None:
        filenames = os.listdir('screenshots')
    if shuffle and use_only is None:
        random.shuffle(filenames)
    if use_only:
        filenames = filenames[:use_only]
    if use_progressbar:
        filenames = progressbar(filenames)
    for file in filenames:
        if file.endswith('.png'):
            img = Image.open(os.path.join('screenshots', file))
            if skip_blackscreen:
                if img.getbbox():
                    yield img
            else:
                yield img

#%%

def preprocess_image(img, img_dim):
    img_rs = img.resize(img_dim)
    img_arr = np.array(img_rs).astype(np.float32)
    img_arr = img_arr / 256.
    img_arr_tp = np.transpose(img_arr, [2, 0, 1])
    return img_arr_tp

#%%

def get_images_batches(img_dim, batch_size=16, use_progressbar=True, use_only=None, shuffle=True, filenames=None, skip_blackscreen=True):
    current_batch = []
    for image in load_images(use_progressbar=use_progressbar, use_only=use_only, shuffle=shuffle, filenames=filenames, skip_blackscreen=skip_blackscreen):
        current_batch.append(preprocess_image(image, img_dim))
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if len(current_batch) > 0:
        yield current_batch

#%%

val_perc = 0.2
shuffled = os.listdir('screenshots')
np.random.shuffle(shuffled)
train = shuffled[np.ceil(len(shuffled) * val_perc).astype(int):]
val = shuffled[:np.ceil(len(shuffled) * val_perc).astype(int)]

#%%

def plot_samples(s):
    fig, ax_rows = plt.subplots(2,2)
    i = 0
    for row in ax_rows:
        for ax in row:
            ax.imshow(s[i].cpu().detach().numpy().transpose([1,2,0]))
            ax.axis('off')
            i += 1
    plt.show()

#%%

def train_gan(gan, generator_optimizer=None, test_optimizer=None, train_batch_generator=None, val_batch_generator=None, use_layers=None, use_alpha=False, batch_size=48):
    if generator_optimizer is None:
        generator_optimizer = optim.Adam(gan.decoder.parameters(), lr=2e-4, betas=(0.5, 0.999))
    if test_optimizer is None:
        test_optimizer = optim.Adam(gan.test.parameters(), lr=2e-4, betas=(0.5, 0.999))

    batch_size = batch_size

    gan.train()

    no_epochs = 400

    # alpha_stepsize = np.power(no_epochs, 1 / no_epochs)
    # print(alpha_stepsize)
    # alpha = 1 / no_epochs

    for epoch in range(no_epochs):

        if use_alpha:
            alpha = (epoch + 1) / no_epochs
        else:
            alpha = 1.

        if epoch % 20 == 0:
            # print(torch.rand((9,) + (gan.decoder.latent_shape,)))
            generate_samples(gan, alpha=alpha, use_layers=use_layers)

        losses = []
        for X_batch_discriminator, X_batch_generator in zip(train_batch_generator(batch_size), train_batch_generator(batch_size, use_progressbar=False)):
            real_scores, generator_scores = gan(X_batch_discriminator, use_layers, alpha)
            minimax_loss = nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) + nn.BCEWithLogitsLoss()(generator_scores, torch.zeros_like(generator_scores))
            losses.append(minimax_loss.item())

            test_optimizer.zero_grad()
            minimax_loss.backward()
            torch.nn.utils.clip_grad_norm_(gan.test.parameters(), 1.)
            test_optimizer.step()


            _, generator_scores = gan(X_batch_generator, use_layers, alpha)
            generator_loss = nn.BCEWithLogitsLoss()(generator_scores, torch.ones_like(generator_scores))
            # generator_loss = -torch.mean(torch.log(generator_scores))

            generator_optimizer.zero_grad()
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(gan.decoder.parameters(), 1.)
            generator_optimizer.step()

            del X_batch_discriminator
            del X_batch_generator
            del real_scores
            del generator_scores
            del minimax_loss
            del generator_loss

            gc.collect()

        #time.sleep(1)

        val_losses = []

        for X_batch_val in val_batch_generator(batch_size):
            real_scores, generator_scores = gan(X_batch_val, use_layers, alpha)
            minimax_loss = nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) + nn.BCEWithLogitsLoss()(generator_scores, torch.zeros_like(generator_scores))
            val_losses.append(minimax_loss.item())

            del X_batch_val
            del real_scores
            del generator_scores
            del minimax_loss

            gc.collect()

        print('Average Loss (epoch {}): TRAIN: {}, VAL: {}'.format(epoch, np.mean(losses), np.mean(val_losses)))
        #plt.scatter(range(len(losses)), losses)
        #plt.show()

        #time.sleep(1)
        try:
            torch.save(gan.state_dict(), 'model-{}-{}.model'.format(final_img_dim, use_layers).replace(' ', ''))
        except:
            print('Save failed')


        #plt.imshow(generated[0].cpu().detach().numpy().reshape(28,28), cmap='gray')
        #plt.show()

    return generator_optimizer, test_optimizer


def generate_samples(gan, alpha=1., use_layers=None, no_samples=1):
    for _ in range(no_samples):
        generated = gan.decoder(torch.normal(0, 1, size=(4, gan.decoder.latent_size)).to(device), use_layers, alpha)
        #print('Generated items are similar: {}'.format(torch.allclose(generated[0], generated[1], atol=0.05)))
        plot_samples(generated)
        del generated


#%%

batch_sizes = [64, 48, 32, 24, 16, 8]


def train_gan_progressive(gan, train_filenames, val_filenames):
    current_batch_size_idx = 0
    generator_optimizer, discriminator_optimizer = None, None
    for use_layers in range(1, len(gan.decoder.deconvs) + 1):
        img_dim = gan.decoder(torch.zeros(1, gan.decoder.latent_size).to(device), use_layers).shape[2:]
        print(f'Training \{img_dim} generator')
        # reversed_img_dim = list(reversed(img_dim))
        while current_batch_size_idx < len(batch_sizes):
            try:
                if np.all(np.array(img_dim) < 64):
                    train_batch_generator = get_images_generator_in_memory(train_filenames, img_dim)
                    val_batch_generator = get_images_generator_in_memory(val_filenames, img_dim)
                else:
                    train_batch_generator = get_images_generator_in_memory(train_filenames, img_dim, cache_in_device=False)
                    val_batch_generator = get_images_generator_in_memory(val_filenames, img_dim, cache_in_device=False)
                use_alpha = use_layers != 1
                generator_optimizer, discriminator_optimizer = train_gan(gan,
                                                                         train_batch_generator=train_batch_generator,
                                                                         val_batch_generator=val_batch_generator,
                                                                         use_layers=use_layers,
                                                                         use_alpha=use_alpha,
                                                                         batch_size=batch_sizes[current_batch_size_idx],
                                                                         generator_optimizer=generator_optimizer,
                                                                         test_optimizer=discriminator_optimizer)
                break
            except RuntimeError as e:
                current_batch_size_idx += 1
                print('CUDA out of meory. Decreasing batch size to {}'.format(batch_sizes[current_batch_size_idx]))
                print(e)
                gc.collect()


generaror_kernel_size = [4, 3, 3, 3, 3, 3]
generator_pooling = [2, 2, 2, 2, 2, 2]
generator_channels = [128, 128, 128, 128, 128, 128]

disc_kernel_size = [3, 3, 3, 3, 3, 4]
disc_channels = [64, 64, 64, 64, 64, 64]
disc_pooling = [2, 2, 2, 2, 2]


def get_images_generator(filenames, img_dim):
    def f(batch_size, use_progressbar=True):
        for img in get_images_batches(img_dim, batch_size, filenames=filenames, use_progressbar=use_progressbar):
            X_batch = torch.from_numpy(np.array(img)).to(device)
            yield X_batch
    return f

def get_images_generator_in_memory(filenames, img_dim, cache_in_device=True):
    img_batches = [batch for batch in get_images_batches(img_dim, 256, filenames=filenames)]
    images = np.concatenate(img_batches)
    torch_images = torch.from_numpy(images)
    if cache_in_device:
        torch_images = torch_images.to(device)
    def f(batch_size, use_progressbar=True):
        permutation = torch.randperm(len(torch_images))
        randomized_images = torch_images[permutation]
        r = range(0, len(randomized_images), batch_size)
        if use_progressbar:
            r = progressbar(r)
        for batch_start in r:
            yield randomized_images[batch_start:batch_start+batch_size].to(device)
        if batch_start and batch_start + batch_size < len(randomized_images):
            yield randomized_images[batch_start+batch_size:].to(device)
    return f


reversed_img_dim = list(reversed(final_img_dim))

device = torch.device("cuda:0")

decoder = CNNDecoder(reversed_img_dim, 256, generaror_kernel_size, generator_channels, generator_pooling, dropout=0.4, use_sigmoid=True)
adversary = AdverserialTest(reversed_img_dim, kernel_size=disc_kernel_size, channels=disc_channels, pooling=disc_pooling)
gan = GAN(decoder, adversary).to(device)

#print(decoder)
#print(adversary)

#%%


generator_opt = None
test_opt = None
#train_batch_generator = get_images_generator_in_memory(train, cache_in_device=False, img_dim=final_img_dim)
#val_batch_generator = get_images_generator_in_memory(val, cache_in_device=False, img_dim=final_img_dim)

#%%
gan.load_state_dict(torch.load('model-{}-{}.model'.format(tuple(final_img_dim), 6).replace(' ', '')))
#train_gan_progressive(gan, train_batch_generator=get_images_generator_in_memory(train, cache_in_device=False), val_batch_generator=get_images_generator_in_memory(val, cache_in_device=False))
#train_gan_progressive(gan, train_batch_generator=get_images_generator_in_memory(train), val_batch_generator=get_images_generator_in_memory(val))
#train_gan_progressive(gan, train_filenames=train, val_filenames=val)#, train_batch_generator=get_images_generator(train), val_batch_generator=get_images_generator(val))
#train_gan(gan, train_batch_generator=get_images_generator_in_memory(train, cache_in_device=False, img_dim=final_img_dim), val_batch_generator=get_images_generator_in_memory(val, cache_in_device=False, img_dim=final_img_dim), use_layers=6)
generate_samples(gan, no_samples=20)