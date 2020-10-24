from flask import Flask, request, jsonify
from urllib import parse

from torch_utils import preprocessing, get_prediction

app = Flask(__name__)

def parse_urlargs(url):
    query = parse.parse_qs(parse.urlparse(url).query)
    return {k:v[0] if v and len(v) == 1 else v for k,v in query.items()}

def allowed_file(filename):
    """ wav only """
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        params = request.data.decode("utf-8").split("&")
        data = { "filename": params[0].split("=")[1], "word": params[1].split("=")[1] }
        
        if data is None or data["filename"] == "": return jsonify({ "error": "no file" })

        if not allowed_file(data["filename"]): return jsonify({ "error": "format not supported" })

        try:
            spectrogram, label = preprocessing("test/"+data["filename"], data["word"])
            score = prediction(spectrogram, label)

            return jsonify({ "result": score, "word": data["word"] })

        except:
            return jsonify({ "error": "error during prediction" })

    # load audio

    # preprocessing
    # predict 
    # return json data
    return jsonify({ "result": 1 })