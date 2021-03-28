from flask import Flask, jsonify, request
from PneumoniaClassificator import PneumoniaClassificator

app = Flask(__name__)
model = PneumoniaClassificator()
model.set_model_architecture()
model.set_model_weights()


@app.route("/predict_pneumonia", methods=["POST"])
def index():
    data = request.data
    prediction = model.predict(data)
    return jsonify({"PneumoniaProbability": str(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
