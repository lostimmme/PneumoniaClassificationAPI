from flask import Flask, jsonify, request
from PneumoniaClassificator import PneumoniaClassificator

app = Flask(__name__)
model = PneumoniaClassificator().get_instance().set_model('weights/mobilenet_weights.h5')


@app.route("/predict_pneumonia", methods=["POST"])
def index():
    data = request.data
    prediction = model.predict(data)
    return jsonify({"ImageClass": str(prediction)})
