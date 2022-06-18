# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ! pip install tensorflow

from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt


# ## get data


class ArtificialData(object):
    """
    人口データを作成するクラス
    """

    def __init__(self, n_samples=100, noise_scale=.1):
        self.n_samples = n_samples
        self.noise_scale = noise_scale

    def true_function(self, x):
        """
        ノイズのない正しいデータを返す関数
        :param x:
        :return:
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def make_x(self):
        return np.sort(np.random.uniform(-1.5, 1.5, size=self.n_samples)).astype(np.float32).reshape(-1, 1)

    def make_noise(self, x):
        return np.random.normal(loc=0, scale=self.noise_scale, size=x.shape)

    def generate(self):
        x = self.make_x()
        y = self.true_function(x)
        y += self.make_noise(y)
        return x, y


def func2(x):
    """
    人口データの正しい関数その2
    :param np.ndarray x:
    :return:
    :rtype: np.ndarray
    """
    return np.sin(5 * x) * np.abs(x)


class Art2(ArtificialData):
    def true_function(self, x):
        return func2(x)

    def make_x(self):
        x1 = np.random.uniform(-1.5, -.5, size=int(self.n_samples / 2))
        x2 = np.random.uniform(.5, 1.5, size=self.n_samples - x1.shape[0])
        x = np.vstack((x1, x2)).reshape(-1, 1)
        return np.sort(x)


def make_data(size, seed=1):
    """
    人工データの作成
    
    :param int size: 
    :param str function_type:
    :param int seed: 
    :return: データと正しい関数の集合
    :rtype: tuple[np.array, np.array, function]
    """
    np.random.seed(seed)
    x, y = Art2(size).generate()
    return x, y


x_test = None
x_train, y_train = make_data(size=100)

plt.scatter(x_train, y_train)

# ## Linear Deep Learning via Chainer

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, GaussianDropout
from tensorflow.keras.initializers import RandomNormal


def build_model(input_dim, hidden_dim, p, activate="relu", mask="Dropout", apply_input=False, apply_hidden=True):
    inputs = keras.Input(shape=input_dim)
    inputs = eval(mask)(p)(inputs, training=apply_input)
    x = Dense(hidden_dim, activation="relu")(inputs)
    x = eval(mask)(p)(x, training=apply_hidden)
    x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = eval(mask)(p)(x, training=apply_hidden)
    x = Dense(hidden_dim, activation="relu")(x)
    outputs = Dense(1)(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return model


# ## Bayes Part


class Transformer(object):
    """
    変数変換の実行クラス
    初めて変数が与えられたとき, 変換と同時にスケーリングのパラメータを学習し保存します。
    二回目以降は、一度目で学習したパラメータを用いて変換を行います。
    """

    def __init__(self, scaling=False):
        """
        コンストラクタ
        :param bool transform_log: 目的変数をログ変換するかのbool. 
        :param bool scaling: 
        """
        self._is_fitted = False
        self.scaling = scaling
        if scaling:
            self.scaler = StandardScaler()

    def _scaling(self, x):
        """

        :param np.ndarray x: 変換する変数配列
        :return: 
        :rtype: np.ndarray
        """
        shape = x.shape
        if len(shape) == 1:
            x = x.reshape(-1, 1)

        if self._is_fitted:
            x = self.scaler.transform(x)
        else:
            x = self.scaler.fit_transform(x)
            self._is_fitted = True
        x = x.reshape(shape)
        return x

    def transform(self, x):
        """
        目的変数のリストを受け取って変換器を作成し, 変換後の値を返す
        scaling変換 [-1,+1]

        :param np.ndarray x:
        :rtype: np.ndarray
        """
        x_trains = x[:]
        if self.scaling:
            x_trains = self._scaling(x)

        return x_trains

    def inverse_transform(self, x):
        """
        変換された値を元の値に逆変換
        :param x: np.ndarray
        :return: np.ndarray
        """
        x_inv = x[:]
        if self.scaling:
            x_inv = self.scaler.inverse_transform(x)

        return x_inv


def save_logloss(loss, name, save=True):
    """
    loss の epoch による変化を plot して保存.
    :param list loss: loss が格納されたリスト
    :param str name: ファイル名
    :param bool save: 保存するかどうかのフラグ. True のとき name で保存する.
    :return:
    """
    loss = np.array(loss)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(loss[:, 1], color="C0")
    ax1.set_yscale("log")
    if save:
        fig.savefig("{name}.png".format(**locals()), dpi=150)
    return


# +
def preprocess(X, y=None):
    """
    入力変数の変換
    :param np.ndarray X:
    :param np.ndarray y:
    :return: 変換後の変数のタプル
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    """
    x_transformed = x_transformer.transform(X)
    if y is None:
        return x_transformed

    y_transformed = y_transformer.transform(y)
    return x_transformed, y_transformed

def inverse_y_transform(y):
    """
    予測値の逆変換
    :param y: 
    :return: 
    """
    return y_transformer.inverse_transform(y)

def preprocess_array_format(x):
    """
     array の shape, 及び type をチェックして chainer に投げられるようにする.

    1. shapeの修正:
         (n_samples, ) -> (n_samples, 1)
    2. dtype の修正:
        int -> np.int32
        float -> np.float32

    :param np.ndarray x:
    :return:
    :rtype: np.ndarray
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.int32)
    elif np.issubdtype(x.dtype, np.float):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)
    return x


# +
def plot_posterior(x_test, x_train=None, y_train=None, n_samples=100):
    xx = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)

    x_train, y_train = x_transformer.inverse_transform(x_train), inverse_y_transform(y_train)
    predict_values = posterior(xx, n=n_samples)

    predict_mean = predict_values.mean(axis=0)
    predict_var = predict_values.var(axis=0)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(x_train[:, 0], y_train[:, 0], "o", markersize=6., color="C0", label="Training Data Points",
             fillstyle="none")

    for i in range(100):
        if i == 0:
            ax1.plot(xx[:, 0], predict_values[i], color="C1", alpha=.1, label="Posterior Samples", linewidth=.5)
        else:
            ax1.plot(xx[:, 0], predict_values[i], color="C1", alpha=.1, linewidth=.5)

    ax1.plot(xx[:, 0], predict_mean, "--", color="C1", label="Posterior Mean")
    ax1.fill_between(xx[:, 0], predict_mean + predict_var, predict_mean - predict_var, color="C1",
                     label="1 $\sigma$", alpha=.5)

    ax1.set_ylim(-3.2, 3.2)
    ax1.set_xlim(min(xx), max(xx))
    ax1.legend(loc=4)
    return fig, ax1

def posterior(x, n=3):
    """
    :param np.ndarray x:
    :param int n: 
    :return:
    :rtype: np.ndarray
    """
    x = preprocess_array_format(x)
    x = preprocess(x)
    pred = [model(x).numpy().reshape(-1) for _ in range(n)]
    pred = inverse_y_transform(pred)
    pred = np.array(pred)
    return pred


# -

apply_input = False
n_epoch = 1000
batch_size = 50
freq_print_loss=10
freq_plot=50
n_samples=100
data_name='art2'

model = build_model(x_train.shape[1], 512, 0.5, apply_input=apply_input)

output_dir = "data/{data_name}/".format(**locals())
# 画像の出力先作成
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

# +
x_transformer = Transformer(scaling=True)
y_transformer = Transformer(scaling=True)

X, y = preprocess(x_train, y_train)

N = X.shape[0]

model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_squared_error'])

from sklearn.metrics import mean_squared_error


# +
list_loss = []

for e in range(n_epoch+1):
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        _x = X[idx]
        _y = y[idx]
        model.train_on_batch(_x, _y)

    y_pred = model(X)
    l = mean_squared_error(y, y_pred)
    
    if e % freq_print_loss == 0:
        print("epoch: {e}\tloss:{l}".format(**locals()))
    
    if e % freq_plot == 0:
        fig, ax = plot_posterior(x_test, X, y, n_samples=n_samples)
        ax.set_title("epoch:{0:04d}".format(e))
        fig.tight_layout()
        file_path = os.path.join(output_dir, "epoch={e:04d}.png".format(**locals()))
        fig.savefig(file_path, dpi=150)
        plt.close("all")
    list_loss.append([e, l])

save_logloss(list_loss, model.__str__())

# +
from glob import glob
import os
from PIL import Image

def make_anime(files, name='anime'):
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save(name+'.gif', save_all=True, \
        append_images=images[1:], optimize=True, duration=10 , loop=0)

l1_images = glob(os.path.join('./data/art2/',  "*.png"))
make_anime(l1_images, 'anime-keras')
# -

# keras version
# <img src='./anime-keras.gif' width=400>
# -
# chainer version

# <img src='./anime.gif' width=400></img>
