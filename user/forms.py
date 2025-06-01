from flask_wtf import FlaskForm
from wtforms import StringField, EmailField, PasswordField, SubmitField, HiddenField
from wtforms.validators import DataRequired, Email, Length, EqualTo, NoneOf, ValidationError
import re


disallowed_chars = ['*', '?', '!', '\'', '^', '+', '%', '&', '/', '(', ')',
                    '=', '}', ']', '[', '{', '$', '#', '@', '<', '>']


def validate_password(self, field):
    p = re.compile(r'(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*\W)')
    if not p.match(field.data):
        raise ValidationError("Must contain 1 digit, at least one upper and lower case letter and a special character")


# Form to take in user registration data
class RegisterForm(FlaskForm):
    username = StringField(validators=[DataRequired(), NoneOf(disallowed_chars)])
    email = EmailField(validators=[DataRequired(), Email()])
    # phone_number = StringField(validators=[DataRequired()])
    password = PasswordField(validators=[DataRequired(), Length(min=6, max=20), validate_password])
    confirm_password = PasswordField(validators=[DataRequired(),
                                                 EqualTo('password', message='Passwords must be equal!')])
    wallet_address = HiddenField(validators=[DataRequired()])
    signature = HiddenField()
    submit = SubmitField()


# TODO: Implement login form
class LoginForm(FlaskForm):
    email = StringField(validators=[DataRequired(), Email()])
    password = PasswordField(validators=[DataRequired()])
    wallet_address = HiddenField(validators=[DataRequired("Connect wallet required")])
    signature = HiddenField()
    submit = SubmitField()
