from flask import Blueprint, render_template, flash, redirect, url_for, request
from app import limiter, app
from flask_login import current_user
import logging
import joblib
from forecast_prediction.forms import ForecastForm
from models.SKTIME_models.Regression_predict import regression_predict
from werkzeug.utils import secure_filename
import os


forecast_prediction_blueprint = Blueprint('forecast_prediction', __name__, template_folder='templates')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


@forecast_prediction_blueprint.route('/forecast', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def forecast_prediction():
    form = ForecastForm()
    if form.validate_on_submit():
        forecast_values = form.upload.data
        if forecast_values and allowed_file(forecast_values.filename):
            filename = secure_filename(forecast_values.filename)
            target = os.path.join(os.getenv('FORECAST_FOLDER'), filename)
            forecast_values.save(target)
            prediction = regression_predict(target)
            logging.warning('SECURITY - Prediction made by [%s, %s]',
                            current_user.email,
                            request.remote_addr)
            return render_template('forecast_prediction/forecast_prediction.html', form=form, filename=filename, forecast_values=prediction)
        # query the ML model for a prediction
        # model = joblib.load('models/SKTIME_models/LSTM_Model.pkl')
        # prediction = model.predict()
        # if prediction is not None:

        #     return render_template('forecast_prediction/forecast_prediction.html', prediction=prediction)
    return render_template('forecast_prediction/forecast_prediction.html', form=form)
