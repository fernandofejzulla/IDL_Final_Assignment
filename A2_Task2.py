import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, Input
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose
from tensorflow.keras.models import Model
from scipy.ndimage import rotate
import tensorflow as tf
tf.config.run_functions_eagerly(True)


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)

# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

    # One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    predicted = [''.join([unique_characters[i] for i in j]) for j in pred]

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

""" Build the text-to-text model"""

def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    text2text.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    text2text.summary()

    return text2text

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def exact_string_text2text(model, X, y_onehot, n_show_wrong=8):
    preds = model.predict(X, verbose=0)

    pred_txt = decode_labels(preds)
    true_txt = decode_labels(y_onehot)

    acc = np.mean([p == t for p, t in zip(pred_txt, true_txt)])
    print(f"\n➡️ text-to-text string accuracy: {acc:.4f}")

    print("Some mistakes:")
    shown = 0
    for p, t in zip(pred_txt, true_txt):
        if p != t:
            print(f"     pred='{p}' | true='{t}'")
            shown += 1
            if shown >= n_show_wrong:
                break

    return acc, pred_txt, true_txt

# Training
splits = {
    "50/50": 0.50,
    "25/75": 0.75,
    "10/90": 0.90
}
# Master dictionary to store history for all splits and models
# Structure: all_histories[split_name][model_name] = history_object
all_histories = {
    "50/50": {},
    "25/75": {},
    "10/90": {}
}
results_t2t = {}  

cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]

for name, test_size in splits.items():
    print(f"\n--- T2T SPLIT {name} ---")
    Xtr, Xte, ytr, yte = train_test_split(X_text_onehot, y_text_onehot, test_size=test_size, random_state=42)
    
    model = build_text2text_model()
    
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=50, batch_size=128, callbacks=cb, verbose=1)
    
    # SAVE HISTORY
    all_histories[name]['Text-to-Text'] = hist 
    
    acc, _, _ = exact_string_text2text(model, Xte, yte)
    results_t2t[name] = acc

print("\n\n FINAL TEXT→TEXT ACCURACY SUMMARY:")
for name, acc in results_t2t.items():
    print(f"{name} split → accuracy = {acc:.4f}")

""" Deeper model for text-to-text"""

def build_text2text_model_deep(hidden_size=256):
    input_len = X_text_onehot.shape[1]     # e.g. 5
    num_chars = X_text_onehot.shape[2]     # e.g. 13
    answer_len = y_text_onehot.shape[1]    # e.g. 3

    enc_inputs = Input(shape=(input_len, num_chars))

    # Two-layer encoder
    x = LSTM(hidden_size, return_sequences=True)(enc_inputs)
    enc_state = LSTM(hidden_size)(x)

    # Decoder (same as before)
    dec = RepeatVector(answer_len)(enc_state)
    dec = LSTM(hidden_size, return_sequences=True)(dec)
    outputs = TimeDistributed(Dense(num_chars, activation='softmax'))(dec)

    model = Model(enc_inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\n" + "="*60)
print("TRAINING DEEP TEXT→TEXT MODEL (extra LSTM layer) – 50/50 split")
print("="*60)

Xtr_deep, Xte_deep, ytr_deep, yte_deep = train_test_split(
    X_text_onehot, y_text_onehot,
    test_size=0.5,
    random_state=42
)

deep_model = build_text2text_model_deep(hidden_size=256)

cb_deep = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]

hist_deep = deep_model.fit(
    Xtr_deep, ytr_deep,
    validation_split=0.1,
    epochs=100,
    batch_size=128,
    callbacks=cb_deep,
    verbose=1
)

deep_acc, _, _ = exact_string_text2text(
    deep_model, Xte_deep, yte_deep, n_show_wrong=8
)

print(f"\nDeep text→text string accuracy (50/50 split): {deep_acc:.4f}")

""" Build the image-to-text model"""

def build_img2text_model(img_shape, answer_len=3, num_chars=13, hidden_size=256):
    
    img_inputs = Input(shape=img_shape)
    x = TimeDistributed(Flatten())(img_inputs)

    enc = LSTM(hidden_size)(x)

    dec = RepeatVector(answer_len)(enc)
    dec = LSTM(hidden_size, return_sequences=True)(dec)

    outputs = TimeDistributed(Dense(num_chars, activation='softmax'))(dec)

    model = Model(img_inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

X_img = X_img.astype('float32') / 255.0

if X_img.ndim == 4:
    X_img = np.expand_dims(X_img, -1)

print("X_image shape:", X_img.shape)
print("y_text_onehot shape:", y_text_onehot.shape)

splits = {
    "50/50": 0.50,
    "25/75": 0.75,
    "10/90": 0.90
}

results_i2t = {}
cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")] 

for name, test_size in splits.items():
    print(f"\n--- I2T SPLIT {name} ---")
    Xtr_i2t, Xte_i2t, ytr_i2t, yte_i2t = train_test_split(X_img, y_text_onehot, test_size=test_size, random_state=42)

    img_shape = Xtr_i2t.shape[1:]
    i2t_model = build_img2text_model(img_shape=img_shape, answer_len=y_text_onehot.shape[1], num_chars=y_text_onehot.shape[2])    

    hist_i2t = i2t_model.fit(Xtr_i2t, ytr_i2t, validation_split=0.1, epochs=50, batch_size=128, callbacks=cb, verbose=1)
    
    # SAVE HISTORY
    all_histories[name]['Image-to-Text'] = hist_i2t
    
    acc_i2t, _, _ = exact_string_text2text(i2t_model, Xte_i2t, yte_i2t, n_show_wrong=8)
    results_i2t[name] = acc_i2t

print("\n\n FINAL IMAGE→TEXT ACCURACY SUMMARY:")
for name, acc in results_i2t.items():
    print(f"{name} split → accuracy = {acc:.4f}")

""" Build the text-to-image model"""

def build_text2img_model(answer_img_shape, 
                         query_len, 
                         num_chars, 
                         hidden_size=256
    ):
   
    flat_dim = int(np.prod(answer_img_shape))

    # Encoder: text
    text_inputs = Input(shape=(query_len, num_chars))           
    enc = LSTM(hidden_size)(text_inputs)                        
    return text_inputs, enc, flat_dim, answer_img_shape

def build_text2img_model_full(query_len,
                              num_chars,
                              answer_len,
                              answer_img_shape,
                              hidden_size=256):
    flat_dim = int(np.prod(answer_img_shape))

    text_inputs = Input(shape=(query_len, num_chars))   # (batch, T_query, num_chars)
    enc = LSTM(hidden_size)(text_inputs)

    x = RepeatVector(answer_len)(enc)                   # (batch, T_answer, hidden_size)
    x = LSTM(hidden_size, return_sequences=True)(x)     # (batch, T_answer, hidden_size)

    x = TimeDistributed(Dense(flat_dim, activation='sigmoid'))(x)
    outputs = TimeDistributed(Reshape(answer_img_shape))(x)  

    model = Model(text_inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy'
    )
    return model

def show_text2img_examples(model, X, y_true, n_examples=3, save_prefix=None):

    idx = np.random.choice(len(X), size=n_examples, replace=False)
    X_sample = X[idx]
    y_sample = y_true[idx]

    y_pred = model.predict(X_sample, verbose=0)

    answer_len = y_sample.shape[1]
    img_shape = y_sample.shape[2:]

    for k in range(n_examples):
        fig, axes = plt.subplots(2, answer_len, figsize=(3*answer_len, 6))
        fig.suptitle(f"Example {k}", fontsize=14)

        for t in range(answer_len):
            # True
            ax_t = axes[0, t]
            img_t = y_sample[k, t]
            if img_t.ndim == 3 and img_t.shape[-1] == 1:
                img_t = img_t[..., 0]
            ax_t.imshow(img_t, cmap='gray')
            ax_t.axis('off')
            ax_t.set_title(f"True t={t}")

            # Pred
            ax_p = axes[1, t]
            img_p = y_pred[k, t]
            if img_p.ndim == 3 and img_p.shape[-1] == 1:
                img_p = img_p[..., 0]
            ax_p.imshow(img_p, cmap='gray')
            ax_p.axis('off')
            ax_p.set_title(f"Pred t={t}")

        plt.tight_layout()
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_example{k}.png")
        plt.show()

y_img = y_img.astype('float32') / 255.0

print("X_text_onehot shape:", X_text_onehot.shape)   # (N, T_query, num_chars)
print("y_img shape:", y_img.shape)                   # (N, T_answer, H, W[,C])

N, query_len, num_chars = X_text_onehot.shape
_, answer_len, *answer_img_shape = y_img.shape
answer_img_shape = tuple(answer_img_shape)           # e.g. (28, 28) or (28, 28, 1)

splits_t2i = {
    "50/50": 0.50,
    "25/75": 0.75,
    "10/90": 0.90
}
results_t2i_loss = {}
cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]

for name, test_size in splits_t2i.items():
    print(f"\n--- T2I SPLIT {name} ---")
    Xtr_t2i, Xte_t2i, ytr_t2i, yte_t2i = train_test_split(X_text_onehot, y_img, test_size=test_size, random_state=42)

    t2i_model = build_text2img_model_full(query_len, num_chars, answer_len, answer_img_shape, hidden_size=256)
    
    # Re-compile with accuracy so we can plot it
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    t2i_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    hist_t2i = t2i_model.fit(Xtr_t2i, ytr_t2i, validation_split=0.1, epochs=50, batch_size=128, callbacks=cb, verbose=1)

    # SAVE HISTORY
    all_histories[name]['Text-to-Image'] = hist_t2i

    # FIX: Unpack the results. evaluate returns [loss, accuracy]
    eval_results = t2i_model.evaluate(Xte_t2i, yte_t2i, verbose=0)
    test_loss = eval_results[0] # The first item is always the loss
    
    results_t2i_loss[name] = test_loss
    
    # Optional: Show examples
    show_text2img_examples(t2i_model, Xte_t2i, yte_t2i, n_examples=3, save_prefix=f"t2i_{name.replace('/','')}")

print("\n\n FINAL TEXT→IMAGE TEST LOSS SUMMARY:")
for name, loss in results_t2i_loss.items():
    print(f"{name} split → test loss = {loss:.4f}")



# ==============================================================================
# OPTIONAL TASK: Judge/Evaluator Model for Text-to-Image
# ==============================================================================

print("\n" + "="*60)
print("STARTING OPTIONAL TASK: SUPERVISED EVALUATOR (THE JUDGE)")
print("="*60)

# 1. HELPER: Create dataset for the Judge (Digits + Signs + Space)
def create_classifier_dataset(num_samples_per_class=2000):
    X_cls = []
    y_cls = []
    
    char_map = {char: i for i, char in enumerate(unique_characters)}
    
    # Add MNIST digits (0-9)
    for digit in range(10):
        inds = np.where(MNIST_labels == digit)[0]
        selected_inds = np.random.choice(inds, num_samples_per_class, replace=True)
        X_cls.append(MNIST_data[selected_inds])
        label = np.zeros((num_samples_per_class, len(unique_characters)))
        label[:, char_map[str(digit)]] = 1
        y_cls.append(label)
        
    # Add Signs (+, -)
    for sign in ['+', '-']:
        imgs = generate_images(num_samples_per_class, sign=sign)
        X_cls.append(imgs)
        label = np.zeros((num_samples_per_class, len(unique_characters)))
        label[:, char_map[sign]] = 1
        y_cls.append(label)

    # Add Space (' ')
    space_imgs = np.zeros((num_samples_per_class, 28, 28))
    X_cls.append(space_imgs)
    label = np.zeros((num_samples_per_class, len(unique_characters)))
    label[:, char_map[' ']] = 1
    y_cls.append(label)
    
    # Concatenate and normalize
    X_cls = np.concatenate(X_cls, axis=0)
    y_cls = np.concatenate(y_cls, axis=0)
    
    # Normalize images to 0-1
    X_cls = X_cls.astype('float32') / 255.0
    
    # Expand dims for CNN (N, 28, 28, 1)
    if X_cls.ndim == 3:
        X_cls = np.expand_dims(X_cls, -1)
        
    return X_cls, y_cls

# 2. HELPER: Build the Judge CNN
def build_judge_model():
    # Simple but accurate CNN
    model = tf.keras.Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(unique_characters), activation='softmax') # 13 classes
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. HELPER: Evaluation Function
def evaluate_text2img_with_judge(t2i_model, judge_model, X_text_test, y_text_test_str):
    print("\nGenerating images from test set...")
    gen_images = t2i_model.predict(X_text_test, verbose=0)
    
    if gen_images.shape[-1] != 1:
        gen_images = np.expand_dims(gen_images, -1)
        
    N, seq_len, H, W, C = gen_images.shape
    
    # Flatten sequence to classify individual characters
    flat_images = gen_images.reshape(N * seq_len, H, W, C)
    
    # Predict
    flat_preds = judge_model.predict(flat_images, verbose=0)
    flat_indices = np.argmax(flat_preds, axis=1)
    
    # Reshape back
    seq_indices = flat_indices.reshape(N, seq_len)
    
    # Decode
    pred_strings = []
    for row in seq_indices:
        s = "".join([unique_characters[idx] for idx in row])
        pred_strings.append(s)
        
    # Calculate Accuracy
    correct = 0
    for pred, true in zip(pred_strings, y_text_test_str):
        if pred == true:
            correct += 1
    acc = correct / len(y_text_test_str)
    
    return acc, pred_strings

# --- EXECUTION ---

# A. Generate Judge Data
print("Generating classifier training data...")
X_judge, y_judge = create_classifier_dataset()
X_judge_tr, X_judge_te, y_judge_tr, y_judge_te = train_test_split(X_judge, y_judge, test_size=0.1)

# B. Train Judge
print("Training the Judge Model...")
judge_model = build_judge_model()
judge_model.fit(X_judge_tr, y_judge_tr, validation_data=(X_judge_te, y_judge_te), epochs=100, batch_size=64, verbose=1)

# C. Evaluate the LAST trained Text-to-Image model (from the previous loop)
# We use the test set from the last split in memory (likely the 10/90 split)
print(f"\nEvaluating the final Text-to-Image model...")

# Convert the existing test set (Xte_t2i) to strings for comparison if not already done
# Note: Xte_t2i is the one-hot input. We need the ground truth strings.
# The variable 'yte_t2i' is images. We need the text labels corresponding to Xte_t2i.
# We can recover them from Xte_t2i because X->Y in this task (Text->Image), 
# but we need the ANSWER string.
# Easier way: Decode Xte_t2i to see the query, calculate the math, get the result string.
# OR: Just re-split the original text data using the same random_state to match indices.
# (Since we used random_state=42 in the loop, we can replicate the split here).

_, X_test_oh_judge, _, y_test_img_judge = train_test_split(
    X_text_onehot, y_img, test_size=0.90, random_state=42 # Matching the 10/90 split
)
# We need the TEXT answers for these specific samples.
_, _, _, y_test_str_judge_oh = train_test_split(
    X_text_onehot, y_text_onehot, test_size=0.90, random_state=42
)
y_test_strings_judge = decode_labels(y_test_str_judge_oh)

# Run Evaluation
judge_acc, judge_preds = evaluate_text2img_with_judge(t2i_model, judge_model, X_test_oh_judge, y_test_strings_judge)

print(f"\n✅ Final Text-to-Image Generative Accuracy: {judge_acc:.4f}")

# Show examples
print("\nJudge Examples (True vs Predicted by Judge reading the Image):")
for i in range(5):
    print(f"True Answer: '{y_test_strings_judge[i]}' | Generated Image read as: '{judge_preds[i]}'")


# ==============================================================================
# PLOTTING: Compare Training & Validation Loss for All 3 Models
# ==============================================================================
# ==============================================================================
# PLOTTING: Accuracy vs Val Accuracy (3 Splits x 3 Models)
# ==============================================================================

# ==============================================================================
# PLOTTING: Combined Accuracy per Split (3 Models on 1 Graph)
# ==============================================================================

# ==============================================================================
# PLOTTING: Combined Accuracy per Split (IN A ROW)
# ==============================================================================

def plot_combined_in_row(histories_dict, filename="combined_splits_row.png"):
    splits = ["50/50", "25/75", "10/90"]
    
    # Define colors
    model_configs = [
        ("Text-to-Text", "blue"),
        ("Image-to-Text", "orange"),
        ("Text-to-Image", "green")
    ]
    
    # Create a grid with 1 Row and 3 Columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Model Performance Comparison by Split", fontsize=16, fontweight='bold')

    for i, split in enumerate(splits):
        ax = axes[i]  # Get the specific subplot for this split
        ax.set_title(f"Split: {split}", fontsize=14)
        
        # Plot each model on this subplot
        for model_name, color in model_configs:
            if split in histories_dict and model_name in histories_dict[split]:
                hist = histories_dict[split][model_name]
                
                # Metric key detection
                acc_key = 'accuracy' if 'accuracy' in hist.history else 'binary_accuracy'
                val_acc_key = 'val_accuracy' if 'val_accuracy' in hist.history else 'val_binary_accuracy'
                
                if acc_key in hist.history:
                    acc = hist.history[acc_key]
                    val_acc = hist.history.get(val_acc_key, [])
                    epochs = range(1, len(acc) + 1)
                    
                    # Plot Training (Solid)
                    ax.plot(epochs, acc, color=color, linestyle='-', linewidth=1.5, 
                             label=f'{model_name} Train')
                    
                    # Plot Validation (Dashed)
                    if len(val_acc) > 0:
                        ax.plot(epochs, val_acc, color=color, linestyle='--', linewidth=1.5, 
                                 label=f'{model_name} Val')
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.5)
        # Only put legend on the first plot to avoid clutter, or put it on all
        if i == 0:
            ax.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for main title
    plt.savefig(filename)
    print(f"\n✅ Saved side-by-side plot: {filename}")
    plt.show()

# Run this instead
plot_combined_in_row(all_histories)

# ==============================================================================
# TASK 2 PART 5: DEEP MODELS (Additional LSTM Layers)
# ==============================================================================

print("\n" + "="*60)
print("STARTING PART 5: DEEP MODEL COMPARISON (25/75 SPLIT)")
print("="*60)

# --- A. Define Deep Versions of the Models ---

def build_img2text_model_deep(img_shape, answer_len=3, num_chars=13, hidden_size=256):
    """Adds an extra LSTM layer to the Encoder"""
    img_inputs = Input(shape=img_shape)
    x = TimeDistributed(Flatten())(img_inputs)
    
    # Encoder Layer 1: Returns full sequence
    x = LSTM(hidden_size, return_sequences=True)(x)
    
    # Encoder Layer 2: Returns final state (Compressed Vector)
    enc = LSTM(hidden_size)(x)

    # Decoder (Standard)
    dec = RepeatVector(answer_len)(enc)
    dec = LSTM(hidden_size, return_sequences=True)(dec)
    outputs = TimeDistributed(Dense(num_chars, activation='softmax'))(dec)

    model = Model(img_inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_text2img_model_deep(query_len, num_chars, answer_len, answer_img_shape, hidden_size=256):
    """Adds an extra LSTM layer to the Encoder"""
    flat_dim = int(np.prod(answer_img_shape))

    text_inputs = Input(shape=(query_len, num_chars))
    
    # Encoder Layer 1: Returns full sequence
    x = LSTM(hidden_size, return_sequences=True)(text_inputs)
    
    # Encoder Layer 2: Final state
    enc = LSTM(hidden_size)(x)

    # Decoder
    x = RepeatVector(answer_len)(enc)
    x = LSTM(hidden_size, return_sequences=True)(x)
    x = TimeDistributed(Dense(flat_dim, activation='sigmoid'))(x)
    outputs = TimeDistributed(Reshape(answer_img_shape))(x)  

    model = Model(text_inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- B. Train Deep Models on the 25/75 Split ---
# We use 25/75 because it's hard enough to show differences, but has enough data for deep models.

split_name = "25/75"
test_pct = 0.75
deep_histories = {}

# 1. Deep Image-to-Text
print(f"\nTraining Deep Image-to-Text on {split_name}...")
Xtr_i2t, Xte_i2t, ytr_i2t, yte_i2t = train_test_split(X_img, y_text_onehot, test_size=test_pct, random_state=42)
img_shape = Xtr_i2t.shape[1:]

deep_i2t = build_img2text_model_deep(img_shape, answer_len=3, num_chars=13)
hist_deep_i2t = deep_i2t.fit(Xtr_i2t, ytr_i2t, validation_split=0.1, epochs=40, batch_size=128, verbose=1)
deep_histories['Image-to-Text'] = hist_deep_i2t

# 2. Deep Text-to-Image
print(f"\nTraining Deep Text-to-Image on {split_name}...")
Xtr_t2i, Xte_t2i, ytr_t2i, yte_t2i = train_test_split(X_text_onehot, y_img, test_size=test_pct, random_state=42)

deep_t2i = build_text2img_model_deep(query_len, num_chars, answer_len, answer_img_shape)
hist_deep_t2i = deep_t2i.fit(Xtr_t2i, ytr_t2i, validation_split=0.1, epochs=40, batch_size=128, verbose=1)
deep_histories['Text-to-Image'] = hist_deep_t2i

# --- C. Compare Shallow vs Deep (Plotting) ---

def plot_deep_comparison(shallow_hist_dict, deep_hist_dict, split_key="25/75"):
    models_to_compare = ['Image-to-Text', 'Text-to-Image']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Shallow vs Deep Architecture (Split: {split_key})", fontsize=16)
    
    for i, model_name in enumerate(models_to_compare):
        ax = axes[i]
        
        # Get Shallow History (from your 'all_histories' global var)
        if split_key in shallow_hist_dict and model_name in shallow_hist_dict[split_key]:
            shallow_hist = shallow_hist_dict[split_key][model_name]
            # Handle metrics naming
            acc_key = 'accuracy' if 'accuracy' in shallow_hist.history else 'binary_accuracy'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in shallow_hist.history else 'val_binary_accuracy'
            
            ax.plot(shallow_hist.history[val_acc_key], 'b--', label='Shallow Val Acc')
        else:
            print(f"Warning: No shallow history found for {model_name} in {split_key}")

        # Get Deep History
        deep_hist = deep_hist_dict[model_name]
        d_acc_key = 'accuracy' if 'accuracy' in deep_hist.history else 'binary_accuracy'
        d_val_acc_key = 'val_accuracy' if 'val_accuracy' in deep_hist.history else 'val_binary_accuracy'
        
        ax.plot(deep_hist.history[d_val_acc_key], 'r-', label='Deep Val Acc', linewidth=2)
        
        ax.set_title(model_name)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Validation Accuracy")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("shallow_vs_deep_comparison.png")
    plt.show()

# Run the comparison plot
# Note: This relies on 'all_histories' being populated from your previous loops
plot_deep_comparison(all_histories, deep_histories, split_key="25/75")