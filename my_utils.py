import matplotlib.pyplot as plt

from keras import backend as K

def plot_learning_curve(history,annotations=True):
    x=list(range(len(history.history["loss"])))
    plt.plot(x,history.history["loss"], label='train_loss')
    ###loss###
    if "val_loss" in history.history:
        y=history.history["val_loss"]
        if(annotations==True):
            ymax = min(y)
            xpos = y.index(ymax)
            xmax = x[xpos]
            plt.annotate('min', xy=(xmax, ymax), xytext=(xmax, ymax),arrowprops=dict(facecolor='violet', shrink=0.05))
        plt.plot(x,y,label="val_loss")
        print("Overfiting after",xpos,"epochs.")
    plt.legend(loc="best")
    plt.show()
    ###acc###
    plt.plot(x,history.history["acc"], label='train_acc')
    if "val_acc" in history.history:
        y=history.history["val_acc"]
        if(annotations==True):
            ymax = y[xmax]
            plt.annotate('Best', xy=(xmax, ymax), xytext=(xmax, ymax),arrowprops=dict(facecolor='violet', shrink=0.05))
        plt.plot(x,y,label='valid_acc')
    plt.legend(loc="best")
    plt.show()
    ###f1###
    ###f1###
    if "f1" in history.history:
        plt.plot(x,history.history["f1"],label='train_f1')
        if "val_f1" in history.history:
            y=history.history["val_f1"]
            if(annotations==True):
                ymax = y[xmax]
                plt.annotate('Best', xy=(xmax, ymax), xytext=(xmax, ymax),arrowprops=dict(facecolor='violet', shrink=0.05))
            plt.plot(x,y,label='val_f1')
        plt.legend(loc="best")
        plt.show()

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

	
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter


def zipf(lis,cutoff):
    hindi_font = FontProperties(fname = './Nirmala.ttf') #if hindi , give relative address
    f=Counter(lis)
    keys=[]
    values=[]
    for key,value in (reversed(sorted(f.items(), key = itemgetter(1)))):
        keys.append(key)
        values.append(value)
    print(len(values))
    values=values[:cutoff]
    keys=keys[:cutoff]
    x=range(len(values))
    #plt.xticks(x,keys,rotation='vertical')#if english
    plt.xticks(x,keys,fontproperties=hindi_font,rotation='vertical')
    plt.plot(x,values)
    plt.show()

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG
#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

def plot_model_graph(model):
	plot_model(model, to_file='model.png')
	SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, K.floatx())
        a /= tf.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

