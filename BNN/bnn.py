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

# ! pip install chainer==1.24.0

from PIL import Image
from glob import glob
from chainer.optimizer import WeightDecay
from chainer import Variable
import os
import chainer.links as L
import chainer.functions as F
from chainer import Chain
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import numpy as np
from chainer import optimizers


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
        return np.sort(np.random.uniform(-1.5, 1.5, size=self.n_samples)
                       ).astype(np.float32).reshape(-1, 1)

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

plt.plot(x_train)

plt.plot(y_train)

plt.scatter(x_train, y_train)

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

# +
model_params = {
    "input_dim": input_dim,
    "output_dim": output_dim,
    "mask_type": 'dropout',
    "activate": 'relu',
    "hidden_dim": 512
}

train_params = {
    "n_epoch": 1000,
    "X": x_train,
    "y": y_train,
    "x_test": x_test,
    "data_name": 'art2'
}


# -

class PreprocessMixin(object):
    """
    特徴量の前処理を行うMixinクラス
    """

    def preprocess(self, X, y=None):
        """
        入力変数の変換
        :param np.ndarray X:
        :param np.ndarray y:
        :return: 変換後の変数のタプル
        :rtype: tuple of (numpy.ndarray, numpy.ndarray)
        """
        x_transformed = self.x_transformer.transform(X)
        if y is None:
            return x_transformed

        y_transformed = self.y_transformer.transform(y)
        return x_transformed, y_transformed

    def inverse_y_transform(self, y):
        """
        予測値の逆変換
        :param y:
        :return:
        """
        return self.y_transformer.inverse_transform(y)


class BNN(Chain):
    """
    ベイジアンニューラルネットの重み学習を行うクラス
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=512,
            activate="relu",
            mask_type="gaussian",
            prob=.5,
            lengthscale=10.):
        """
        :param int input_dim: 入力層の次元数
        :param int output_dim: 出力層の次元数
        :param int hidden_dim: 隠れ層の次元数
        :param str activate: 活性化関数
        :param str mask_type:
            変数へのマスクの種類を表すstring.
            "dropout", "gaussian", Noneのいずれかを指定
        :param float prob:
            dropoutの確率を表すfloat.
            0.のときdropoutをしないときに一致します.
            [0, 1) の小数
        :param float lengthscale:
            初期のネットワーク重みの精度パラメータ. 大きい値になるほど0に近い値を取ります.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activate_name = activate
        self.activate = self._get_function(activate)
        self.mask_type = mask_type
        self.lengthscale = lengthscale

        super().__init__(
            l1=L.Linear(
                input_dim,
                hidden_dim,
                initial_bias=np.random.normal(
                    scale=1. / lengthscale,
                    size=(hidden_dim)),
                initialW=np.random.normal(
                    scale=1. / lengthscale,
                    size=(
                        hidden_dim,
                        input_dim))),
            l2=L.Linear(
                hidden_dim,
                hidden_dim,
                initial_bias=np.random.normal(
                    scale=1. / lengthscale,
                    size=(hidden_dim)),
                initialW=np.random.normal(
                    scale=1. / lengthscale,
                    size=(
                        hidden_dim,
                        hidden_dim))),
            l3=L.Linear(
                hidden_dim,
                hidden_dim,
                initial_bias=np.random.normal(
                    scale=1. / lengthscale,
                    size=(hidden_dim)),
                initialW=np.random.normal(
                    scale=1. / lengthscale,
                    size=(
                        hidden_dim,
                        hidden_dim))),
            l4=L.Linear(
                hidden_dim,
                output_dim,
                initial_bias=np.random.normal(
                    scale=1. / lengthscale,
                    size=(output_dim)),
                initialW=np.random.normal(
                    scale=1. / lengthscale,
                    size=(
                        output_dim,
                        output_dim))))

        self.mask = Mask(name=mask_type, prob=prob)

    def _get_function(self, s):
        """
        文字列からそれに対応する関数を取得

        :param str s: 関数を表す文字列
        :return:
        """
        if s == "relu":
            f = F.relu
        elif s == "sigmoid":
            f = F.sigmoid
        elif s == "tanh":
            f = F.tanh
        else:
            print("対応する関数が見つかりません")
            def f(x): return x
        return f

    def __call__(self, x, apply_input=False, apply_hidden=True):
        """
        ネットワークの出力を作成

        :param Variable x: 入力ベクトル
        :param bool apply_hidden:
            隠れ層に対してマスクをかけるかのフラグ.
            True のときm `mask` によって生成されたマスクを隠れ層に掛ける
        :param bool apply_input:
            入力層に対してマスクをかけるかのフラグ.
            True にすると学習が不安定になることが観測されているため, 学習時には False が推奨
        :return: 出力
        :rtype: Variable
        """
        x1 = self.mask.apply(x, apply_input)
        h1 = self.activate(self.l1(x1))

        h1 = self.mask.apply(h1, apply_hidden)
        h2 = self.activate(self.l2(h1))

        h2 = self.mask.apply(h2, apply_hidden)
        h3 = self.activate(self.l3(h2))

        h3 = self.mask.apply(h3, apply_hidden)
        h4 = self.l4(h3)
        return h4

    def __str__(self):
        """
        ネットワークの条件をいい感じの文字列で表現する
        :return: ネットワーク条件の文字
        :rtype: str
        """
        s = "hidden={0.hidden_dim}_activate={0.activate_name}_{0.mask}".format(
            self)
        return s


class Mask(object):
    """
    入力変数に数値を掛けて摂動を与える関数クラス
    """

    def __init__(self, name="dropout", prob=.5):
        """
        :param str name: マスクに用いる手法の名前. "dropout", "gaussian", None のいずれかを指定
        :param float prob: マスクの確率
        """

        self.prob = self._check_prob(prob)
        self.name = name
        if name == "dropout":
            self.mask_generator = self._dropout_mask
        elif name == "gaussian":
            self.mask_generator = self._gaussian_mask
        elif name is None or name.lower() == "none":
            self.mask_generator = self._none_mask
        else:
            raise NameError(
                "name: {name} に該当するmask関数が見当たりません. ".format(
                    **locals()))

    def __repr__(self):
        s = "maskname={0.name}_prob={0.prob}".format(self)
        return s

    def _check_prob(self, prob):
        if prob >= 1.:
            prob = 1.
        elif prob < 0:
            prob = 0
        return prob

    def _dropout_mask(self, size):
        z = np.random.binomial(
            1, self.prob, size=size).astype(
            np.float32) * self.prob ** -1
        return z

    def _gaussian_mask(self, size):
        sigma = self.prob / (1. - self.prob)
        z = np.random.normal(loc=1., scale=sigma, size=size).astype(np.float32)
        return z

    def _none_mask(self, size):
        return np.array([1.] * size).astype(np.float32)

    def _make(self, size):
        z = self.mask_generator(size)
        return Variable(np.diag(z))

    def apply(self, h, do_mask=True):
        """
        ベクトルにマスクを掛ける関数
        :param Variable h: マスクされる変数
        :return: masked variable
        :rtype: Variable
        """
        if do_mask is False:
            return h

        size = h.shape[1]
        z = self._make(size)
        z = F.matmul(h, z)
        return z


class Transformer(object):
    """
    変数変換の実行クラス
    初めて変数が与えられたとき, 変換と同時にスケーリングのパラメータを学習し保存します。
    二回目以降は、一度目で学習したパラメータを用いて変換を行います。
    """

    def __init__(self, transform_log=False, scaling=False):
        """
        コンストラクタ
        :param bool transform_log: 目的変数をログ変換するかのbool.
        :param bool scaling:
        """
        self._is_fitted = False
        self.transform_log = transform_log
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
        * log変換 -> scaling変換 [-1,+1]

        :param np.ndarray x:
        :rtype: np.ndarray
        """
        x_trains = x[:]
        if self.transform_log:
            x_trains = np.log(x)

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

        if self.transform_log:
            x_inv = np.exp(x_inv)

        return x_inv


class BNNEstimator(BaseEstimator, PreprocessMixin):
    """
    Bayesian Neural Network を訓練し可視化を行うクラス
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=512,
            activate="relu",
            mask_type="gaussian",
            prob=.5,
            lengthscale=10.,
            optimizer="adam",
            weight_decay=4 * 10 ** -5,
            apply_input=False,
            x_scaling=True,
            y_scaling=True):
        """
        :param int input_dim: 入力層の次元数
        :param int output_dim: 出力層の次元数
        :param int hidden_dim: 隠れ層の次元数
        :param str activate: 活性化関数
        :param str mask_type:
            変数へのマスクの種類を表すstring.
            "dropout", "gaussian", None のいずれかを指定
        :param float prob:
            dropoutの確率を表す [0, 1) の小数.
            0.のときdropoutをしないときに一致します.
        :param float lengthscale:
            初期のネットワーク重みの精度パラメータ. 大きい値になるほど0に近い値を取ります.
        :param str optimizer: optimizer を指す string.
        :param float weight_decay:
            勾配の減衰パラメータ.
            目的関数に対して, 指定した重みの加わったL2ノルム正則化と同じ役割を果たします.
        :param bool apply_input:
            入力次元に対して mask を適用するかどうかを表す bool.
        :param bool x_scaling:
            入力変数を正規化するかを表す bool.
        :param bool y_scaling:
            目的変数を正規化するかを表す bool.
        """

        self.model = BNN(
            input_dim,
            output_dim,
            hidden_dim,
            activate,
            mask_type,
            prob,
            lengthscale)
        self.weight_decay = weight_decay
        self.apply_input = apply_input
        self.x_transformer = Transformer(scaling=x_scaling)
        self.y_transformer = Transformer(scaling=y_scaling)

        if optimizer == "adam":
            self.optimizer = optimizers.Adam()
        self.conditions = str(self.model)

    def fit(
            self,
            X,
            y,
            x_test=None,
            data_name=None,
            n_epoch=1000,
            batch_size=20,
            freq_print_loss=10,
            freq_plot=50,
            n_samples=100):
        """
        モデルのパラメータチューニングの開始
        :param np.ndarray X:
        :param np.ndarray y:
        :param np.ndarray | None x_test:
        :param str | None data_name:
        :param int n_epoch:
        :param int batch_size:
        :param int freq_print_loss:
        :param int freq_plot:
        :param int n_samples: 事後分布プロットの際の事後分布のサンプリング数.
        :return: self
        """
        conditions = self.conditions
        output_dir = "data/{data_name}/{conditions}".format(**locals())
        # 画像の出力先作成
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

        X, y = self.preprocess(X, y)
        if x_test is not None:
            x_test = self.x_transformer.transform(x_test)

        N = X.shape[0]

        # Variable 型への変換
        X = Variable(preprocess_array_format(X))
        y = Variable(preprocess_array_format(y))
        if x_test is not None:
            x_test = Variable(preprocess_array_format(x_test))

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(WeightDecay(self.weight_decay))
        list_loss = []

        for e in range(1, n_epoch + 1):
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i: i + batch_size]
                _x = X[idx]
                _y = y[idx]
                self.model.zerograds()
                loss = F.mean_squared_error(self.model(
                    _x, apply_input=self.apply_input), _y)
                loss.backward()
                self.optimizer.update()

            l = F.mean_squared_error(self.model(X, False, False), y).data
            if e % freq_print_loss == 0:
                print("epoch: {e}\tloss:{l}".format(**locals()))

            if e % freq_plot == 0:
                fig, ax = self.plot_posterior(
                    x_test, X.data, y.data, n_samples=n_samples)
                ax.set_title("epoch:{0:04d}".format(e))
                fig.tight_layout()
                file_path = os.path.join(
                    output_dir, "epoch={e:04d}.png".format(
                        **locals()))
                fig.savefig(file_path, dpi=150)
                plt.close("all")
            list_loss.append([e, l])

        save_logloss(list_loss, self.model.__str__())

    def plot_posterior(
            self,
            x_test,
            x_train=None,
            y_train=None,
            n_samples=100):
        model = self.model
        if x_test is None:
            xx = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
        else:
            xx = self.x_transformer.inverse_transform(x_test)

        x_train, y_train = self.x_transformer.inverse_transform(
            x_train), self.inverse_y_transform(y_train)
        predict_values = self.posterior(xx, n=n_samples)

        predict_mean = predict_values.mean(axis=0)
        predict_var = predict_values.var(axis=0)
        tau = (1. - model.mask.prob) * self.model.lengthscale ** 2. / \
            (2 * len(x_train) * self.weight_decay)
        predict_var += tau ** -1

        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train[:, 0], y_train[:, 0], "o", markersize=6.,
                 color="C0", label="Training Data Points", fillstyle="none")

        for i in range(100):
            if i == 0:
                ax1.plot(xx[:, 0], predict_values[i], color="C1",
                         alpha=.1, label="Posterior Samples", linewidth=.5)
            else:
                ax1.plot(xx[:, 0], predict_values[i],
                         color="C1", alpha=.1, linewidth=.5)

        ax1.plot(xx[:, 0], predict_mean, "--",
                 color="C1", label="Posterior Mean")
        ax1.fill_between(xx[:, 0], predict_mean +
                         predict_var, predict_mean -
                         predict_var, color="C1", label="1 $\sigma$", alpha=.5)

        ax1.set_ylim(-3.2, 3.2)
        ax1.set_xlim(min(xx), max(xx))
        ax1.legend(loc=4)
        return fig, ax1

    def posterior(self, x, n=3):
        """
        :param np.ndarray x:
        :param int n:
        :return:
        :rtype: np.ndarray
        """
        x = preprocess_array_format(x)
        x = self.preprocess(x)
        x = Variable(x)
        pred = [self.model(x, apply_input=False,
                           apply_hidden=True).data.reshape(-1) for _ in range(n)]
        pred = self.inverse_y_transform(pred)
        pred = np.array(pred)
        return pred


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


clf = BNNEstimator(**model_params)


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


clf.fit(**train_params)

# +


def make_anime(files, name='anime'):
    images = list(map(lambda file: Image.open(file), files))
    images[0].save(name + '.gif',
                   save_all=True,
                   append_images=images[1:],
                   optimize=True,
                   duration=10,
                   loop=0)


l1_images = glob(
    os.path.join(
        './data/art2/hidden=512_activate=relu_maskname=dropout_prob=0.5',
        "*.png"))
make_anime(l1_images, 'anime')
# -

# <img src='./anime.gif' width=400></img>
