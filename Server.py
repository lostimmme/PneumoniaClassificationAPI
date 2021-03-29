from flask import Flask, jsonify, request
from PneumoniaClassificator import PneumoniaClassificator

app = Flask(__name__)
model = PneumoniaClassificator()  # Constructing object of classificator
model.set_model_architecture()  # setting architecture for model
model.set_model_weights()  # setting pretrained weights for model


@app.route("/predict_pneumonia", methods=["POST"])
def index():
    """In this route we take post request and return probability of image class.
    Returns
    -------
        :returns json_value: json_file
    """
    data = request.data  # parse data from post request
    prediction = model.predict(data)  # getting predictions
    return jsonify({"PneumoniaProbability": str(prediction)})  # return json {"PneumoniaProbability": some_value}


if __name__ == "__main__":
    app.run(debug=True)  # Running python server on localhost:5000 in debug mode
