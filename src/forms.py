from flask_wtf import Form
from wtforms import StringField, PasswordField, validators
from wtforms.validators import DataRequired


class RegisterForm(Form):
    """
    Register Form
    """
    username = StringField(label="username", validators=[validators.Length(min=4, max=25)])
    email = StringField(label="E-mail", validators=[validators.Length(min=4, max=25)])
    password = PasswordField(label='Password', validators=[
                    DataRequired()
                ])


class LoginForm(Form):
    """
    Login Form
    """
    email = StringField(label="E-mail", validators=[validators.Length(min=4, max=25)])
    password = PasswordField(label='Password', validators=[
                    DataRequired()
                ])
