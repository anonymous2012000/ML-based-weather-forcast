from flask_wtf import FlaskForm
from wtforms import SubmitField


class RetrainModelForm(FlaskForm):
    submit = SubmitField('Retrain Model')
    cancel = SubmitField('Cancel')
