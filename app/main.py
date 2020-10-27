import os
from flask import Flask, request, jsonify, render_template
from urllib import parse

from torch_utils import preprocessing, get_predictions

UPLOAD_FOLDER = './'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def parse_urlargs(url):
    query = parse.parse_qs(parse.urlparse(url).query)
    return {k:v[0] if v and len(v) == 1 else v for k,v in query.items()}

def allowed_file(filename):
    """ wav only """
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    file = request.files.get("audio")
    word = form.get("word")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    
    spectrogram, label = preprocessing(file.filename, word)
    score = get_predictions(spectrogram, label)

    return render_template("score.html", score=score, word=word)


@app.route("/collect", methods=["POST"])
def collect():
    file = request.files.get("audio")
    form = request.form
    label = form.get("label")
    score = form.get("score") # input ex: 1;0;1;0
    return None