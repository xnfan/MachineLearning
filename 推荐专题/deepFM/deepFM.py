import numpy as np
import tensorflow as tf

import keras
import pandas as pd
from keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras import backend as K
import gc
from keras.layers import Input, Embedding, Dense,Flatten, Concatenate, Activation, Reshape, BatchNormalization, \
     Dropout, Add, RepeatVector, Multiply, Lambda, Subtract
from keras.models import Model
from keras.optimizers import Adam
import time

# 对离散型变量进行编码 (LabelEncoder:[1,2,6] -> [0,1,2], OneHotEncode:[1] -> [0,1,0])
def category_encode(train_np, test_np=None, one_hot=True):
    lb_enc = LabelEncoder()
    train_all = np.vstack((train_np, test_np))
    train_all = np.unique(train_all)
    lb_enc.fit(train_all.reshape(-1, 1))
    train_np = lb_enc.transform(train_np.reshape(-1, 1))
    test_np = lb_enc.transform(test_np.reshape(-1, 1))
    if one_hot:
        oh_enc = OneHotEncoder()
        train_all = np.vstack((train_np.reshape(-1, 1), test_np.reshape(-1, 1)))
        train_all = np.unique(train_all)
        oh_enc.fit(train_all.reshape(-1, 1))
        sparse_train_np = oh_enc.transform(train_np.reshape(-1, 1))
        sparse_test_np = oh_enc.transform(test_np.reshape(-1, 1))
        return sparse_train_np, sparse_test_np
    return train_np, test_np

def log_loss(y_true, y_pred):
    log_loss = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return log_loss


class DeepFM():
    def __init__(self, params):
        self.dnn_layers = params.get("dnn_layers", 4)
        self.emb_dims = params.get("emb_dims", 4)
        self.dnn_dims = params.get("dnn_dims", 8)
        self.label_col = params.get("label_col", "label")
        self.category_col = params.get("category_col", [])  # possible values of each feature
        self.continue_col = params.get("continue_col", [])
        self.learning_rate = params.get("learning_rate", 0.01)
        self.batch_size = params.get("batch_size", 64)
        self.epochs = params.get("epochs", 10)
        self.seed = params.get("random_seed", 2021)
        self.regularize = params.get("regularize", keras.regularizers.l2)
        self.opt_reg_param = params.get("opt_rag_param", 0.15)
        self.layer_reg_param = params.get("layer_reg_param", 0.2)
        self.dropout = params.get("dropout", 0.9)
        self.continue_use_emb = params.get("continue_use_emb", True)
        self.model_name = params.get("model_name", "deepFM")
        self.model = None

    def build_model(self):
        dnn_layers = self.dnn_layers
        emb_dim = self.emb_dims
        dnn_dim = self.dnn_dims
        category_col = self.category_col
        continue_col = self.continue_col
        lr = self.learning_rate
        seed = self.seed
        reg = self.regularize
        layer_reg_param = self.layer_reg_param
        dropout = self.dropout
        continue_use_emb = self.continue_use_emb

        np.random.seed(seed)
        inputs = []
        flatten_layers = []

        # ------second order term------- 离散型和连续型变量统一编码为隐向量，维度一致
        for category_index, num in enumerate(category_col):
            inputs_c = Input(shape=(1,), dtype='int32', name='input_%d' % category_index)
            inputs.append(inputs_c)
            embed_c = Embedding(
                num,
                emb_dim,
                input_length=1,
                name='embed_%d' % category_index,
                embeddings_regularizer=reg(layer_reg_param)
            )(inputs_c)

            flatten_c = Reshape((emb_dim,))(embed_c)
            flatten_layers.append(flatten_c)
        inputs_dict = []

        for continue_feature in continue_col:
            inputs_c = Input(shape=(1,), dtype='float', name='input_sec_%s' % continue_feature)
            inputs.append(inputs_c)
            inputs_c = BatchNormalization(name='BN_%s' % continue_feature)(inputs_c)
            inputs_dict.append(inputs_c)
            if continue_use_emb:
                inputs_c = Dense(emb_dim)(inputs_c)
            else:
                inputs_c = RepeatVector(emb_dim)(inputs_c)
                inputs_c = Flatten()(inputs_c)
            flatten_layers.append(inputs_c)

        sum_features_emb = Add()(flatten_layers)
        sum_square_features_emb = Multiply()([sum_features_emb, sum_features_emb])  # 和的平方
        square_list = []
        for layer in flatten_layers:
            square_feature_emb = Lambda(lambda x: x**2)(layer)
            square_list.append(square_feature_emb)
        square_sum_features_emb = Add()(square_list)  # 平方的和
        y_second_order = Subtract()([sum_square_features_emb, square_sum_features_emb])
        y_second_order = Lambda(lambda x: x * 0.5)(y_second_order)
        y_second_order = Dropout(dropout, seed=seed)(y_second_order)

        # ----first order------
        fm_layers = []
        for category_index, num in enumerate(category_col):
            embed_c = Embedding(
                num,
                1,
                input_length=1,
                name='linear_%s' % category_index,
                embeddings_regularizer=reg(layer_reg_param)
            )(inputs[category_index])
            flatten_c = Flatten()(embed_c)
            fm_layers.append(flatten_c)

        for index, _ in enumerate(continue_col):
            inputs_c = Dense(1)(inputs[len(category_col)+index])
            fm_layers.append(inputs_c)
        y_first_order = Add()(fm_layers)
        y_first_order = BatchNormalization()(y_first_order)
        y_first_order = Dropout(dropout, seed=seed)(y_first_order)

        # deep
        y_deep = Concatenate()(flatten_layers)  # None * (F*K)
        for index in range(dnn_layers):
            y_deep = Dense(dnn_dim)(y_deep)
            y_deep = Activation('relu', name='deep_%d' % index)(y_deep)
            y_deep = Dropout(rate=dropout, seed=seed)(y_deep)

        concat_input = Concatenate(axis=1)([y_first_order, y_second_order, y_deep])

        outputs = Dense(1, activation='sigmoid', name='output')(concat_input)
        self.model = Model(inputs=inputs, outputs=outputs, name='model')
        solver = Adam(lr=lr, decay=0.1)

        self.model.compile(optimizer=solver, loss='binary_crossentropy', metrics=[auc, log_loss])
        plot_model(self.model, to_file='DeepFM.png', rankdir='LR')
        return self.model

    def fit(self, X, y, val_X, val_y, batch_size=None, epochs=None):
        if not batch_size:
            batch_size = self.batch_size
        if not epochs:
            epochs = self.epochs
        his = self.model.fit(X, y, batch_size=batch_size, validation_data=(val_X, val_y), epochs=epochs)
        self.model.save(self.model_name)
        return his

    def predict(self, test_X):
        model = keras.models.load_model(self.model_name, custom_objects={"auc": auc, "log_loss": log_loss})
        y_pred = model.predict(test_X)
        return y_pred

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)


if __name__ == '__main__':
    df_all = pd.read_csv("../ctr_data.csv")
    df_train = df_all[:int(len(df_all)*0.8)]
    df_test = df_all[int(len(df_all)*0.8):]
    print(df_train.head())
    print(df_test.head())

    all_columns = df_train.columns.tolist()

    params = {"label_col": "click",
              "continue_col": ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"],
              "dnn_layers": 2,
              "emb_dims": 8,
              "dnn_dims": 32,
              "learning_rate": 0.01,
              "batch_size": 64,
              "epochs": 25,
              "random_seed": 2021,
              "regularize": keras.regularizers.l2,
              "opt_reg_param": 0.15,
              "layer_reg_param": 0.2,
              "dropout": 0.8,
              "continue_use_emb": True,
              "model_name": "deepFM"
              }

    all_features = all_columns.copy()
    all_features.remove(params.get("label_col"))
    continue_columns = params.get("continue_col", [])
    print(continue_columns, all_columns)

    col_index = []
    for col in all_features:
        col_index.append(all_columns.index(col))

    target_col = all_columns.index(params.get("label_col"))

    category_columns = all_features.copy()
    [category_columns.remove(x) for x in continue_columns]
    max_features = {}
    for i in range(len(category_columns)):
        max_features[category_columns[i]] = (df_all[category_columns[i]].unique().shape[0])

    # del df_all
    # gc.collect()

    max_features_df = pd.DataFrame(data=np.array([list(max_features.keys()), list(max_features.values())]).T,
                                   columns=['ids', 'max_features'], index=range(len(max_features)))

    max_features = pd.merge(pd.DataFrame(category_columns, columns=['ids']), max_features_df, on=['ids'])

    max_features.max_features = max_features.max_features.astype(int)
    max_features = max_features.max_features.tolist()
    params["category_col"] = max_features

    for i in category_columns:
        df_train[i], df_test[i] = category_encode(df_train[i].values.reshape(-1, 1),
                                           df_test[i].values.reshape(-1, 1), one_hot=False)

    train_x, train_y = df_train[all_features], df_train[params.get('label_col')]
    test_x, test_y = df_test[all_features], df_test[params.get('label_col')]
    del df_test
    del df_train
    gc.collect()

    X = train_x.T.values
    y = train_y.values
    X = [np.array(X[i, :]) for i in range(X.shape[0])]
    validation_data = (test_x.T.values, test_y.values)
    val_X, val_y = validation_data
    val_X = [np.array(val_X[i, :]) for i in range(val_X.shape[0])]

    del train_x
    del validation_data
    gc.collect()

    deepFM = DeepFM(params)
    deepFM.build_model()
    his = deepFM.fit(X, y, val_X, val_y, epochs=15)

    pd.DataFrame(his.history).to_csv("%s_his.csv" % (time.strftime('%Y-%m-%d', time.localtime(time.time()))))

    y_pred = deepFM.predict(val_X)

    from sklearn.metrics import roc_auc_score, log_loss

    print(roc_auc_score(val_y, y_pred))
    print(log_loss(val_y, y_pred))