import numpy as np
from copy import deepcopy
from tqdm import tqdm
import os
import emoji
import pandas as pd


class CharByCharSynhthetizer():
    """ Synthetize text (char-by-char) from a trained RNN using a one-hot encoder."""

    def __init__(self, rnn, char_init, encode_lambda, onehot_encoder, decode_lambda, ts, n_step, path_out):
        self.rnn = rnn
        self.char_init = char_init
        self.encode_lambda = encode_lambda
        self.onehot_encoder = onehot_encoder
        self.decode_lambda = decode_lambda
        self.ts = ts
        self.n_step = n_step
        self.path_out = path_out

        try:
            os.remove(path_out)
        except OSError:
            pass

        self.idx_init = encode_lambda([char_init])[0]
        self.onehot_init = onehot_encoder(onehot_encoder(np.array([self.idx_init]).T, encode=True))

    def sample(self, lenght, p):
        """ Weighted sampling of next character based on RNN predicitons."""
        # select character from softmax weighted dist over all chars
        return np.random.choice(range(lenght), size=1, replace=True, p=p.flatten())

    def __call__(self, step):
        if step % self.n_step == 0:
            assert self.onehot_init.shape == (1, self.rnn.out_dim)
            x = deepcopy(self.onehot_init)
            sequence = []

            for t in range(self.ts):
                p = self.rnn.forward(x)
                x_idx = self.sample(lenght=self.rnn.out_dim, p=p)
                x_char = self.decode_lambda(x_idx)[0]
                sequence.append(x_char)
                x = self.onehot_encoder(np.array([x_idx]).T, encode=True)

            sequence_str = "".join(sequence)

            with open(self.path_out, "a") as f:
                f.write(f"step={step}\n\n")
                f.write(sequence_str)
                f.write("\n\n")

            tqdm.write(f"step={step}\n\n")
            tqdm.write(sequence_str)
            tqdm.write("\n\n")

        else:
            pass


class OneHotEncoder():
    """ One-hot encoder class.

    Attributes
    ----------
    length : int
        The length of the one-hot encoding.

    Methods
    -------
    __init__(layers)
        Constuctor.
    __call__(x, encode=True)
        Encode a sequence of integers into a one-hot encoded vectors,
        or decode a sequence of one-hot encoded vectors into a
        sequence of integers.
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, length):
        """ Constructor.

        Parameters
        ----------
        length : int
            The length of the one-hot encoding.

        Notes
        -----
        None
        """
        # length of one-hot encoding
        self.length = length

    def __call__(self, x, encode=True):
        """ Encode a sequence of integers into a one-hot encoded vectors,
        or decode a sequence of one-hot encoded vectors into a
        sequence of integers..

        Parameters
        ----------
        x : np.ndarray
            The sequence of index representation of chars, of shape (n_chars,)

        Returns
        -------
        e or d: np.ndarray
            The sequence of one-hot encoded vectors of chars, of shape (n_chars, length)

        Notes
        -----
        None
        """
        if encode:
            e = np.zeros((x.shape[0], self.length))
            e[np.arange(x.shape[0]), x] = 1
            return e.astype(int)
        else:
            d = np.argwhere(x == 1)[:, 1]
            return d.astype(int)

    def __repr__(self, ):
        """ Returns the string representation of class.

        Parameters
        ----------

        Returns
        -------
        repr_str : str
            The string representation of the class.

        Notes
        -----
        None
        """
        repr_str = "one-hot encoder"
        return repr_str


def unique_characters(data):
    """ Get the list of unique characters in a data.

    Parameters
    ----------
    data : list
        A list of strings. The strings may be of different lenghts.

    Returns
    -------
    np.ndarray
        The list of unique characters in all of the strings in data.

    Notes
    -----
    None
    """
    chars = []

    for text in data:
        chars_current = list(dict.fromkeys(text))
        chars = list(dict.fromkeys(chars + chars_current))

    return np.array(chars)


def char_to_idx(char, chars):
    """ Convert a char to an index from the encoder np array.

    Parameters
    ----------
    char : str
        A char.
    chars : np.ndarray
        All chars.

    Returns
    -------
    np.ndarray
        The index repre of char, of shape (,).

    Notes
    -----
    None
    """
    return np.argwhere(char == chars).flatten()[0]


def idx_to_char(idx, chars):
    """ Convert an index to char in the encoder np array.

    Parameters
    ----------
    idx : int
        The index repr of a char.
    chars : np.ndarray
        All chars.

    Returns
    -------
    str
        The char.

    Notes
    -----
    None
    """
    return chars[idx]


def encode(decoding, chars):
    """ Encode a sequence of chars into a sequence of indices based on the encoder.

    Parameters
    ----------
    decoding : np.ndarray
        The sequence of chars, of shape (n_chars,)
    chars : np.ndarray
        All chars.

    Returns
    -------
    encoding : np.ndarray
        The sequence of index representation of the chars, of shape (n_chars,)

    Notes
    -----
    None
    """
    encoding = []

    for d in decoding:
        encoding.append(char_to_idx(d, chars))

    encoding = np.array(encoding)

    return encoding


def decode(encoding, chars):
    """ Decode a sequence of indices into a sequence of chars based on the encoder.

    Parameters
    ----------
    encoding : np.ndarray
        The sequence of index representation of the chars, of shape (n_chars,)
    chars : np.ndarray
        All chars.

    Returns
    -------
    decoding : np.ndarray
        The sequence of chars, of shape (n_chars,)

    Notes
    -----
    None
    """
    decoding = []

    for e in encoding:
        decoding.append(idx_to_char(e, chars))

    decoding = np.array(decoding)

    return decoding


def make_decoded_dataset(dataset):
    """ Decode a dataset of strings into a list of characters.

    Parameters
    ----------
    dataset : list
        A list of strings (contexts) maybe of varying size.

    Returns
    -------
    decoded_dataset : list
        A list of lists (contexts) where a context is a list of characters.

    Notes
    -----
    None
    """
    decoded_dataset = []
    for context in dataset:
        context_elements = list(context)
        decoded_dataset.append(context_elements)
    return decoded_dataset


def make_encoded_dataset(decoded_dataset, chars):
    """ Encode a dataset of list of charcters into a list of integers.

    Parameters
    ----------
    decoded_dataset : list
        A list of lists (contexts) where a context is a list of characters.
    chars : np.ndarray
        All chars.

    Returns
    -------
    encoded_dataset : list
        A list of lists (contexts) where a context is a list of integers.
        An integer corresponds to its index in chars.

    Notes
    -----
    None
    """
    encoded_dataset = []
    for decoded_context in decoded_dataset:
        encoded_context = encode(decoded_context, chars)
        encoded_dataset.append(encoded_context)
    return encoded_dataset


def make_one_hot_encoded_dataset(encoded_dataset, onehot_encoder):
    """ One-hot encode a dataset of list of integers into a list of one-hot encoded vectors.

    Parameters
    ----------
    encoded_dataset : list
        A list of lists (contexts) where a context is a list of integers.
        An integer corresponds to its index in chars.
    onehot_encoder : OneHotEncoder
        A one-hot encoder initilaized with chars (all unique characters in the dataset).

    Returns
    -------
    onehot_encoded_dataset : list
        A list of one-hot encoded vectors (contexts).
        The index of 1s in the vectors corresponds to the index of the character in chars.

    Notes
    -----
    None
    """
    onehot_encoded_dataset = []
    for encoded_context in encoded_dataset:
        onehot_encoded_context = onehot_encoder(encoded_context, encode=True)
        onehot_encoded_dataset.append(onehot_encoded_context)

    return onehot_encoded_dataset


def give_emoji_free_text(text):
    """https://stackoverflow.com/a/50602709"""
    return emoji.get_emoji_regexp().sub(r'', text)


def add_eol_to_text(text, eol="."):
    return ''.join(list(text) + [eol])


def limit_text_length(df, col_name, max_length=140):
    mask = (df[col_name].str.len() <= max_length)
    df_filtered = df.loc[mask]
    print(f"reduced size from {df.shape[0]} to {df_filtered.shape[0]}\n")
    return df_filtered


def synthetize(rnn, eol, chars, onehot_encoder, ts, path_out):
    gen_times = 10

    for gen_time in range(gen_times):
        char_init = eol
        encode_lambda = lambda d: encode(d, chars)
        decode_lambda = lambda e: decode(e, chars)
        # n_step is just a dummy variable here for teh callback to work.
        n_step = 1

        synthetizer = CharByCharSynhthetizer(rnn, char_init, encode_lambda, onehot_encoder, decode_lambda,
                                             ts, n_step, path_out)

        synthetizer(step=n_step)
