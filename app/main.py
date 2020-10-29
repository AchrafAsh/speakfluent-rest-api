import os
import csv
from flask import Flask, request, jsonify, render_template
from urllib import parse
from torch_utils import preprocessing, get_predictions

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        form = request.form
        file = request.files.get("audio")
        filename = file.filename + ".wav"
        file.save(filename)

        word = form.get("word")

        spectrogram, label = preprocessing(filename, word)
        score = get_predictions(spectrogram, label)
        print(f"score: {score}")
        response = jsonify({"score": score, "word": word})

        return _corsify_actual_response(response)
    else:
        raise RuntimeError(
            "Weird - don't know how to handle method {}".format(request.method))


@app.route("/collect", methods=["POST", "OPTIONS"])
def collect():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        form = request.form
        file = request.files.get("audio")
        filename = file.filename + ".wav"
        file.save(filename)

        word = form.get("word")
        score = '0'
        for i in range(1, len(word)):
            score += '|0'

        # update data.csv
        with open('data.csv', 'a', newline='\n') as data_csv:
            spamwriter = csv.writer(data_csv, delimiter=' ')
            spamwriter.writerow([filename, word, 'train', score])

        response = jsonify({"message": "thank you!"})

        return _corsify_actual_response(response)
    else:
        raise RuntimeError(
            "Weird - don't know how to handle method {}".format(request.method))


@app.route("/labels", methods=["GET", "OPTIONS"])
def get_samples():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "GET":  # The actual request following the preflight
        # get labels from data.csv
        samples = []
        with open('data.csv', 'r') as data_csv:
            reader = csv.DictReader(data_csv, delimiter=' ')
            for row in reader:
                print(row)
                score = list(map(int, row["score"].split('|')))
                samples.append(
                    {"word": row["word"], "filename": row["filename"], "score": score})

        response = jsonify({"samples": samples})

        return _corsify_actual_response(response)
    else:
        raise RuntimeError(
            "Weird - don't know how to handle method {}".format(request.method))


@app.route("/evaluate", methods=["POST", "OPTIONS"])
def collect():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        form = request.form
        filename = form.get("filename")
        word = form.get("word")
        score = form.get("score")
        print(f"filename: {filename}, word: {word}, score: {score}")

        # update score in data.csv
        with open('data.csv', 'a', newline='\n') as data_csv:
            spamwriter = csv.writer(data_csv, delimiter=' ')
            spamwriter.writerow([filename, word, 'train', score])

        response = jsonify({"message": "thank you!"})

        return _corsify_actual_response(response)
    else:
        raise RuntimeError(
            "Weird - don't know how to handle method {}".format(request.method))


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
