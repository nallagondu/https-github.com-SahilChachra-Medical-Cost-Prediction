from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "rf_tuned.pkl"
try:
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    app.logger.error("Error loading model: %s", e)
    raise

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input
        features = [int(x) for x in request.form.values()]
        if len(features) != 6:
            raise ValueError("Expected 6 features")

        # Predict
        final = np.array(features).reshape((1, 6))
        pred = model.predict(final)[0]

        # Return prediction
        if pred < 0:
            return render_template('op.html', pred='Error calculating Amount!')
        else:
            return render_template('op.html', pred='Expected amount is {0:.3f}'.format(pred))
    except Exception as e:
        app.logger.error("Prediction failed: %s", e)
        return render_template('op.html', pred='Failed to predict. Please try again later.')

if __name__ == '__main__':
    app.run(debug=True)