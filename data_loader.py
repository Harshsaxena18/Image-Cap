import string
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_descriptions(token_path):
    doc = open(token_path, 'r').read()
    descriptions = {}
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            image_desc = ' '.join(tokens[1:])
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append(image_desc)
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc_list[i] = ' '.join(desc)
    return descriptions

def create_tokenizer(descriptions):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    new_descriptions = '\n'.join(lines)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_descriptions.split('\n'))
    return tokenizer

def load_train_images(train_images_path):
    doc = open(train_images_path, 'r').read()
    dataset = []
    for line in doc.split('\n'):
        if len(line) > 1:
            identifier = line.split('.')[0]
            dataset.append(identifier)
    return set(dataset)

def load_image_paths(images_path, train_images):
    img = glob.glob(images_path + '*.jpg')
    train_img = []
    for i in img:
        if i[len(images_path):] in train_images:
            train_img.append(i)
    return train_img

def load_train_descriptions(new_descriptions, train):
    train_descriptions = {}
    for line in new_descriptions.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in train:
            if image_id not in train_descriptions:
                train_descriptions[image_id] = []
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            train_descriptions[image_id].append(desc)
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    return train_descriptions, all_train_captions

def create_vocab(train_captions, word_count_threshold):
    word_counts = {}
    nsents = 0
    for sent in train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    return vocab

def create_word_mappings(vocab):
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    vocab_size = len(ixtoword) + 1
    return wordtoix, ixtoword, vocab_size

def create_sequences(train_descriptions, wordtoix, max_length):
    all_desc = []
    for key in train_descriptions.keys():
        [all_desc.append(d) for d in train_descriptions[key]]
    lines = all_desc
    sequences = []
    for line in lines:
        seq = [wordtoix[word] for word in line.split(' ') if word in wordtoix]
        sequences.append(seq)
    return sequences

def create_data_generator(sequences, photos, wordtoix, max_length, vocab_size, batch_size):
    n = 0
    X1, X2, y = [], [], []
    while True:
        for sequence in sequences:
            n += 1
            photo = photos[sequence[0][:-1] + '.jpg'][0]
            for i in range(1, len(sequence)):
                in_seq, out_seq = sequence[:i], sequence[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = [], [], []
                n = 0
