from main import get_pred

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict-aplhabet', methods=["post"])
def predictData():
    img = request.files.get("alphabet")
    pred = get_pred(img)
    return jsonify({
        "prediction": pred,
    }), 200


if __name__ == '__main__':
    app.run(debug=True)