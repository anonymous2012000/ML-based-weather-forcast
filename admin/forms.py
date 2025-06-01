from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired


class AddClientForm(FlaskForm):
    username = StringField(validators=[DataRequired()])
    submit = SubmitField()
