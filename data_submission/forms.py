from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileAllowed
from wtforms.validators import DataRequired


class DataUploadForm(FlaskForm):
    upload = FileField(validators=[DataRequired(), FileAllowed(['csv'])])
    submit = SubmitField()
