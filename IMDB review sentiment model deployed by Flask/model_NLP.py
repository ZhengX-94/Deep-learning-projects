import tensorflow as tf
from tensorflow import keras

def comment_preprocess(comment):
    ### makes word_index
    imdb = keras.datasets.imdb
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    ###convert comment strings into vectors for training
    ###max vocabulery is 10000
    comment_new = []
    for x in comment.split():
        try:
            if word_index[x] < 10000:
                comment_new.append(word_index[x])
            else:
                comment_new.append(2)
        except:
            comment_new.append(2)

    comment_new = keras.preprocessing.sequence.pad_sequences([comment_new],
                                                             value=word_index['<PAD>'],
                                                             padding='post',
                                                             maxlen=256)
    return comment_new

if __name__ == '__main__':
    ##testing
    model = tf.keras.models.load_model('./model.h5')
    comment = "fawn I like this file so much, it has a lot of interesting things and it always makes me laugh!"
    print(model.predict([comment_preprocess(comment)]))
