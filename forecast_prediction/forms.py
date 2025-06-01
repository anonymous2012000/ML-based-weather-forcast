from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed


class ForecastForm(FlaskForm):
    upload = FileField('Feature', validators=[DataRequired(), FileAllowed(['csv'])])
    submit = SubmitField()
