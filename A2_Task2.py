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
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text


from sklearn.model_selection import train_test_split
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

history_t2t = {}

results_t2t = {}  

for name, test_size in splits.items():

    print("\n" + "="*60)
    print(f"TRAINING TEXT→TEXT MODEL FOR SPLIT {name} (test_size={test_size})")
    print("="*60)

    # Split dataset
    Xtr, Xte, ytr, yte = train_test_split(
        X_text_onehot, y_text_onehot,
        test_size=test_size,
        random_state=42
    )

    # Build fresh model
    model = build_text2text_model()

    # Train
    hist = model.fit(
        Xtr, ytr,
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    history_t2t[name] = hist.history

    # Evaluate (string accuracy)
    acc, preds, trues = exact_string_text2text(model, Xte, yte)

    results_t2t[name] = acc

print("\n\n FINAL TEXT→TEXT ACCURACY SUMMARY:")
for name, acc in results_t2t.items():
    print(f"{name} split → accuracy = {acc:.4f}")

# ---- Plot training/validation accuracy for text→text ----
plt.figure(figsize=(8, 5))
for split_name, h in history_t2t.items():
    epochs = range(1, len(h['accuracy']) + 1)
    plt.plot(epochs, h['accuracy'], marker='o', label=f'{split_name} train')
    plt.plot(epochs, h['val_accuracy'], linestyle='--', label=f'{split_name} val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Text→Text: training vs validation accuracy for different splits')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('t2t_accuracy_splits.png')
plt.show()

# ---- Plot training/validation loss for text→text ----
plt.figure(figsize=(8, 5))
for split_name, h in history_t2t.items():
    epochs = range(1, len(h['loss']) + 1)
    plt.plot(epochs, h['loss'], marker='o', label=f'{split_name} train')
    plt.plot(epochs, h['val_loss'], linestyle='--', label=f'{split_name} val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Text→Text: training vs validation loss for different splits')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('t2t_loss_splits.png')
plt.show()

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
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ✅ define this ONCE here, not inside the function
results_t2t_deep = {}

print("\n" + "="*60)
print("TRAINING DEEP TEXT→TEXT MODELS (extra LSTM layer) – ALL SPLITS")
print("="*60)

for split_name, test_size in splits.items():
    print(f"\n--- Deep Text→Text split {split_name} ---")

    # train/test split for this deep model
    Xtr_deep, Xte_deep, ytr_deep, yte_deep = train_test_split(
        X_text_onehot, y_text_onehot,
        test_size=test_size,
        random_state=42
    )

    # build fresh deep model
    deep_model = build_text2text_model_deep(hidden_size=256)

    # train
    hist_deep = deep_model.fit(
        Xtr_deep, ytr_deep,
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    # evaluate (string accuracy)
    deep_acc, _, _ = exact_string_text2text(
        deep_model, Xte_deep, yte_deep, n_show_wrong=5
    )

    print(f"Deep text→text string accuracy ({split_name} split): {deep_acc:.4f}")

    # store for Excel
    results_t2t_deep[split_name] = float(deep_acc)

""" Build the image-to-text model"""

def build_img2text_model(img_shape, answer_len=3, num_chars=13, hidden_size=256):
    
    img_inputs = Input(shape=img_shape)
    x = TimeDistributed(Flatten())(img_inputs)

    enc = LSTM(hidden_size)(x)

    dec = RepeatVector(answer_len)(enc)
    dec = LSTM(hidden_size, return_sequences=True)(dec)

    outputs = TimeDistributed(Dense(num_chars, activation='softmax'))(dec)

    model = Model(img_inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def show_img2text_examples(model, X, y_onehot, n_examples=3, save_prefix=None):
    """
    Show a few MNIST query sequences with true vs predicted text answers.
    X        : (N, T_query, H, W, 1)
    y_onehot: (N, T_answer, num_chars)
    """
    # Get predicted and true strings
    preds = model.predict(X, verbose=0)
    pred_txt = decode_labels(preds)
    true_txt = decode_labels(y_onehot)

    # Choose random examples
    idxs = np.random.choice(len(X), size=n_examples, replace=False)

    T_query = X.shape[1]

    for k, idx in enumerate(idxs):
        fig, axes = plt.subplots(1, T_query, figsize=(3*T_query, 3))
        fig.suptitle(
            f"Example {k} – true='{true_txt[idx]}' | pred='{pred_txt[idx]}'",
            fontsize=12
        )

        for t in range(T_query):
            img = X[idx, t]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            axes[t].imshow(img, cmap='gray')
            axes[t].axis('off')
            axes[t].set_title(f"t={t}")

        plt.tight_layout()
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_img2text_example{idx}.png")
        plt.show()
        plt.close(fig)

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

histories_i2t = {}

results_i2t = {}

#Training
for name, test_size in splits.items():

    print("\n" + "="*60)
    print(f"TRAINING IMAGE→TEXT MODEL FOR SPLIT {name} (test_size={test_size})")
    print("="*60)

    # Split dataset
    Xtr_i2t, Xte_i2t, ytr_i2t, yte_i2t = train_test_split(
        X_img, y_text_onehot,
        test_size=test_size,
        random_state=42
    )

    # Build fresh model
    img_shape = Xtr_i2t.shape[1:]
    i2t_model = build_img2text_model(
        img_shape = img_shape,
        answer_len = y_text_onehot.shape[1],
        num_chars = y_text_onehot.shape[2],
        hidden_size = 256
    )    

    #Train
    hist_i2t = i2t_model.fit(
        Xtr_i2t, ytr_i2t,
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    print(f"\nImage→Text visual examples for split {name}:")
    show_img2text_examples(i2t_model, Xte_i2t, yte_i2t, n_examples=3, save_prefix=f"img2text_{name.replace('/', '')}")

    histories_i2t[name] = hist_i2t.history

    # Evaluate (string accuracy)
    acc_i2t, preds_i2t, trues_i2t = exact_string_text2text(i2t_model, Xte_i2t, yte_i2t, n_show_wrong=8)
    results_i2t[name] = acc_i2t

print("\n\n FINAL IMAGE→TEXT ACCURACY SUMMARY:")
for name, acc in results_i2t.items():
    print(f"{name} split → accuracy = {acc:.4f}")

plt.figure(figsize=(8,5))
for split_name, h in histories_i2t.items():
    epochs = range(1, len(h['accuracy']) + 1)
    plt.plot(epochs, h['accuracy'], marker='o', label=f'{split_name} train')
    plt.plot(epochs, h['val_accuracy'], linestyle='--', label=f'{split_name} val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Image→Text: training vs validation accuracy for different splits')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('i2t_accuracy_splits.png')
plt.show()

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
    model.compile(
        optimizer='adam',
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
        plt.close(fig) 

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

histories_t2i = {}

results_t2i_loss = {}

for name, test_size in splits_t2i.items():
    print("\n" + "="*60)
    print(f"TRAINING TEXT→IMAGE MODEL FOR SPLIT {name} (test_size={test_size})")
    print("="*60)

    Xtr_t2i, Xte_t2i, ytr_t2i, yte_t2i = train_test_split(
        X_text_onehot, y_img,
        test_size=test_size,
        random_state=42
    )

    t2i_model = build_text2img_model_full(
        query_len=query_len,
        num_chars=num_chars,
        answer_len=answer_len,
        answer_img_shape=answer_img_shape,
        hidden_size=256
    )

    hist_t2i = t2i_model.fit(
        Xtr_t2i, ytr_t2i,
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    histories_t2i[name] = hist_t2i.history

    test_loss = t2i_model.evaluate(Xte_t2i, yte_t2i, verbose=0)
    print(f"Test binary crossentropy (pixel-wise) for split {name}: {test_loss:.4f}")
    results_t2i_loss[name] = test_loss

    # ---- Visual examples (TEXT→IMAGE ONLY) ----
    print(f"\nSome visual examples for TEXT→IMAGE, split {name}:")
    show_text2img_examples(
        t2i_model,
        Xte_t2i,
        yte_t2i,
        n_examples=3,
        save_prefix=f"text2img_{name.replace('/', '')}"
    )


print("\n\n FINAL TEXT→IMAGE TEST LOSS SUMMARY:")
for name, loss in results_t2i_loss.items():
    print(f"{name} split → test loss = {loss:.4f}")

plt.figure(figsize=(8,5))
for split_name, h in histories_t2i.items():
    epochs = range(1, len(h['loss']) + 1)
    plt.plot(epochs, h['loss'], marker='o', label=f'{split_name} train')
    plt.plot(epochs, h['val_loss'], linestyle='--', label=f'{split_name} val')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy loss')
plt.title('Text→Image: training vs validation loss for different splits')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('t2i_loss_splits.png')
plt.show()


import pandas as pd
#table summarizing all results
summary_rows = []

# 1) Baseline text→text
for split_name, acc in results_t2t.items():
    summary_rows.append({
        "task": "text2text_baseline",
        "split": split_name,
        "metric": "string_accuracy",
        "value": float(acc),
    })

# 2) Deep text→text (ALL splits)
for split_name, acc in results_t2t_deep.items():
    summary_rows.append({
        "task": "text2text_deep",
        "split": split_name,
        "metric": "string_accuracy",
        "value": float(acc),
    })

# 3) Image→text
for split_name, acc in results_i2t.items():
    summary_rows.append({
        "task": "img2text",
        "split": split_name,
        "metric": "string_accuracy",
        "value": float(acc),
    })

# 4) Text→image (loss)
for split_name, loss in results_t2i_loss.items():
    summary_rows.append({
        "task": "text2img",
        "split": split_name,
        "metric": "binary_crossentropy",
        "value": float(loss),
    })

df_summary = pd.DataFrame(summary_rows)

# save to Excel
excel_path = "seq2seq_results.xlsx"

try:
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"Results saved to: {excel_path}")
    print(df_summary)

except ModuleNotFoundError as e:
    print("\n Could not write Excel file. Missing dependency.")
    print("   Error:", e)
    print("   Install it inside your venv with:\n")
    print("   python -m pip install openpyxl\n")