import glob
import os
import pathlib
import pprint
import random
import re

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, url_for
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

UPLOAD_FOLDER = "./uploads"
GRAPH_FOLDER = "./graphs"
p_temp = pathlib.Path("C:/Users/SatoshiHoriuchi/Downloads/houses/")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/fileUpLoad", methods=["GET", "POST"])
def fileUpLoad():
    send_data = request.files["send_data"]
    send_data.save(os.path.join(app.config["UPLOAD_FOLDER"], send_data.filename))
    # データを可視化
    df = pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], send_data.filename))
    header = df.columns  # DataFrameのカラム名の1次元配列のリスト
    record = df.values.tolist()  # DataFrameのインデックスを含まない全レコードの2次元配列のリスト
    # データの要約
    description = df.describe()

    # 要約のdfに行indexの列を付与して見やすい形に
    description["index"] = description.index
    # 列の並び替え
    col_order = [(len(description.columns) - 1)]
    for i in range(len(description.columns) - 1):
        col_order.append(i)

    description = description.reindex(columns=description.columns[col_order])
    # htmlに渡す形に変形
    header_desc = description.columns
    record_desc = description.values.tolist()

    return render_template(
        "lookData.html",
        header=header,
        record=record,
        header_desc=header_desc,
        record_desc=record_desc,
    )


# 確認用
# @app.route("/lookData/<filename>", methods=["GET"])
# def lookData(filename):
#     df = pd.read_csv(filename)
#     header = df.columns  # DataFrameのカラム名の1次元配列のリスト
#     record = df.values.tolist()  # DataFrameのインデックスを含まない全レコードの2次元配列のリスト
#     return render_template("lookData.html", header=header, record=record)


@app.route("/regression", methods=["GET", "POST"])
def regression():
    df = pd.read_csv(list(p_temp.glob("**/*.csv"))[0])
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    train_size = train.shape[0]
    test_size = test.shape[0]
    return render_template(
        "regression.html", train_size=train_size, test_size=test_size
    )


@app.route("/lightgbm", methods=["GET", "POST"])
def lightgbm():
    df = pd.read_csv(list(p_temp.glob("**/*.csv"))[0])
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    # 訓練データ
    x_train = train.drop("y", axis=1)
    y_train = train["y"]
    # テストデータ
    x_test = test.drop("y", axis=1)
    y_test = test["y"]

    import lightgbm as lgb

    lgb_reg = lgb.LGBMRegressor()
    lgb_reg.fit(x_train, y_train)
    # y_train_pred = lgb_reg.predict(x_train)
    y_test_pred = lgb_reg.predict(x_test)

    import matplotlib.pyplot as plt

    plt.scatter(y_test, y_test_pred)
    plt.title("LightGBM 結果(テストデータ)", fontname="MS Gothic")
    plt.xlabel("実績値", fontname="MS Gothic")
    plt.ylabel("予測値", fontname="MS Gothic")
    plt.savefig(
        GRAPH_FOLDER + "/lightgbm_" + str(random.randint(10000000, 99999999)) + ".png"
    )
    plt.show()

    return render_template("lightgbm.html")
