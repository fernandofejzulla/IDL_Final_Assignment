
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape


def load_real_samples(image_dir, scale=False, img_size=64, limit=20000):
    images = []
    count = 0
    if not os.path.exists(image_dir):
        print(f"Warning: Directory {image_dir} not found. Ensure dataset path is correct.")
        return np.zeros((1, img_size, img_size, 3)) 

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            count += 1
            if count >= limit: break

    X = np.array(images, dtype=np.float32)
    if scale:
        X = (X - 127.5) * 2
    return X / 255.0

import json
with open("config.json", "r") as file:
    config = json.load(file)

dataset = load_real_samples(config["data_folder"])



def grid_plot(images, latent_dim, filters, epoch='', name='', n=3, save=False, scale=False, model_name="vae"):
    if scale:
        images = (images + 1) / 2.0
    plt.figure(figsize=(n*2, n*2))
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        filename = f"results/{model_name}/latent{latent_dim}_filters{filters}_epoch{epoch+1}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')
    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input)
    for _ in range(n_downsampling_layers):
        x = Conv2D(**default_args, filters=filters)(x)
    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x)
    return tf.keras.Model(inputs=input, outputs=x, name='Encoder')

def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x)
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')
    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x)
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)
    return tf.keras.Model(inputs=input, outputs=x, name='Decoder')

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

def build_vae(data_shape, latent_dim, filters=128):
    encoder = build_conv_net(data_shape, latent_dim*2, filters=filters)
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var])
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)
    
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))

    class KLLossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_var = inputs
            kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
            self.add_loss(kl_loss / tf.cast(tf.keras.backend.prod(data_shape), tf.float32))
            return inputs
    
    _, _ = KLLossLayer()([z_mean, z_var])
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    return encoder, decoder, vae


latent_dims = [64, 128]
filters = [128, 256]
SEEDS = [42, 101, 202] 
EPOCHS = 100

os.makedirs('results/vae', exist_ok=True)

for latent_dim in latent_dims:
    for filter_count in filters:
        print(f"\n=== Configuration: Latent Dim {latent_dim}, Filters {filter_count} ===")
        
        all_seeds_histories = []
        
        for seed_idx, seed in enumerate(SEEDS):
            print(f"   > Running Seed {seed} ({seed_idx + 1}/{len(SEEDS)})...")
            
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim=latent_dim, filters=filter_count)
            
            seed_loss_history = []
            
            for epoch in range(EPOCHS):
         
                history = vae.fit(x=dataset, y=dataset, epochs=1, batch_size=8, verbose=0)
                loss = history.history['loss'][0]
                seed_loss_history.append(loss)
                
                if seed_idx == 0:
                    coefficient = 6                               
                    latent_vectors = np.random.randn(9, latent_dim) 
                    
                    if epoch == 99:
                        images = decoder(latent_vectors / coefficient) 
                        grid_plot(images, latent_dim, filter_count, epoch, 
                                  name=f'VAE Generated (Latent:{latent_dim} Filter:{filter_count})', 
                                  n=3, save=True, model_name="vae")
                    
                    if epoch == 19:
                        point_a = np.random.randn(1, latent_dim)
                        point_b = np.random.randn(1, latent_dim)
                        point_interp = (point_a + point_b) / 2.0
                        latent_batch = np.vstack([point_a, point_interp, point_b])
                        generated_images = decoder(latent_batch / coefficient)
                        
                        plt.figure(figsize=(12, 4))
                        titles = ["Random Point A", "Interpolation (A+B)/2", "Random Point B"]
                        for i in range(3):
                            plt.subplot(1, 3, i + 1)
                            plt.imshow(generated_images[i])
                            plt.title(titles[i])
                            plt.axis('off')
                        plt.savefig(f'results/vae/vae_interpolation_latent{latent_dim}_filter{filter_count}.png')
                        plt.close()
            
            all_seeds_histories.append(seed_loss_history)

        all_seeds_histories = np.array(all_seeds_histories)
        
        mean_loss = np.mean(all_seeds_histories, axis=0)
        std_loss = np.std(all_seeds_histories, axis=0)
        epochs_range = range(1, EPOCHS + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, mean_loss, label='Mean Training Loss', color='#1f77b4')
        
        plt.fill_between(epochs_range, 
                         mean_loss - std_loss, 
                         mean_loss + std_loss, 
                         color='#1f77b4', 
                         alpha=0.3, 
                         label='Standard Deviation')
        
        plt.title(f'VAE Training Loss (Log Scale)\nLatent Dim: {latent_dim}, Filters: {filter_count}, Seeds: {SEEDS}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log') 
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        
        plot_filename = f'results/vae/loss_plot_log_latent{latent_dim}_filter{filter_count}.png'
        plt.savefig(plot_filename)
        plt.close()
