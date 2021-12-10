from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./model.pkl", "rb"))
tfidf = pickle.load(open("./tfidf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    
    text = request.form.get('news')
    corpus = []
    corpus.append(text)
    corpus = tfidf.transform(corpus)
    pred = model.predict(corpus)
    
    labels = ['Real', 'Fake']
    return render_template("index.html", pred_text=f"{text} is {labels[pred[0]]} news!")
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
   
