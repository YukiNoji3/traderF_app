# 必要なモジュールのインポート
import joblib
import torch
import torch.nn.functional as F
#import MeCab
from janome.tokenizer import Tokenizer
import pickle
from wtforms import Form, StringField, SubmitField, validators
#from animal import transform, Net # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template
#from sklearn.feature_extraction.text import CountVectorizer
from tradeF import Net # tradeF.py から前処理とネットワークの定義を読み込み

#vectorizerの読み込み
with open("./cnt_vec.pickle", "rb") as f:
    vectorizer = pickle.load(f)

#janomeを使った分かち書きのための関数
def tokenize(text):
    t = Tokenizer()
    # 改行コードを除去
    text = text.replace('\n', '')
    tokens = t.tokenize(text)
    result = ' '.join([token.surface for token in tokens])
    return result

#テキストの前処理関数
def text_to_tensor(text):
  wakati_text = tokenize(text)
  bow = vectorizer.transform([wakati_text]).toarray()
  tensor = torch.tensor(bow, dtype=torch.float32)
  return tensor

#テキストの推測関数
def predict(input_tensor):
    # 推論モードに切り替え
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('./tweet_judge.pt', map_location=torch.device('cpu'))) 
    
    # 推論の実行
    with torch.no_grad():
        y = net(input_tensor.to().unsqueeze(0))

    # 推論ラベルを取得
    y = torch.argmax(F.softmax(y, dim=-1))

    return y

#推論したラベルから買い候補かそうでないかを返す関数
def Judge_Tweet(label):
    if label == 0:
        return '買い候補'
    elif label == 1:
        return '買わない候補'

#Flaskのインスタンス化
app = Flask(__name__)

#入力フォームの設定を行うクラス
class InputForm(Form):
    TweetText = StringField('ここにツイートを入力', [validators.DataRequired()])#Tweetの入力データの設定
    submit = SubmitField('判定')#html側で表示するsubmitボタンの設定

#URLにアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    #WTFormsで構築したフォームをインスタンス化
    inputForm = InputForm(request.form)
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        #条件に当てはまらない場合
        if inputForm.validate() == False:
            return render_template('index.html', forms=inputForm)
        #条件に当てはまる場合、推論を実行
        else:
            #tweet = str(request.form['TweetText'])
            tweet = request.form['TweetText']
            tensor = text_to_tensor(tweet)
            pred = predict(tensor)
            Judge_Result_ = Judge_Tweet(pred)
            return render_template('result.html', Judge_Result=Judge_Result_)
        #return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=inputForm)

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
