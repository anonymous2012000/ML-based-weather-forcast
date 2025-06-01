import time

from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import os
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from flask_talisman import Talisman
from web3 import Web3
import logging
import threading
from flask_login import LoginManager, current_user
from urllib.parse import urlparse, urljoin


# custom logging filter to only write security logs to the output log file
class SecurityFilter(logging.Filter):
    def filter(self, record):
        return 'SECURITY' in record.getMessage()


# configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# set the path of the log file
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'admin/logs')
log_filepath = os.path.join(log_dir, 'security-events.log')

# handler logs events to an output log file
file_handler = logging.FileHandler(log_filepath, 'a')
file_handler.setLevel(logging.WARNING)
# add the custom filter to the file handler
file_handler.addFilter(SecurityFilter())
# format the logs to include time and the event message
formatter = logging.Formatter('%(asctime)s : %(message)s', '%m/%d/%Y %I:%M:%S %p')

# add the format to the file handler
file_handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(file_handler)

# load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# rate limiter throws 429 error if user exceeds limits
limiter = Limiter(get_remote_address,
                  app=app,
                  default_limits=["200 per day", "50 per hour"],
                  )

# app configurations
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # used for validating sessions
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_ECHO'] = os.getenv('SQLALCHEMY_ECHO')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['FORECAST_FOLDER'] = os.getenv('FORECAST_FOLDER')
app.config['GANACHE_URL'] = os.getenv('GANACHE_URL')
app.config['CONTRACT_OWNER_ADDR'] = os.getenv('CONTRACT_OWNER_ADDR')
app.config['CONTRACT_ADDR'] = os.getenv('CONTRACT_ADDR')
app.config['ABI'] = os.getenv('ABI')
app.config['TRAINING_DATA'] = os.getenv('TRAINING_DATA')
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH')
app.config['GANACHE_URL'] = os.getenv('GANACHE_URL')

db = SQLAlchemy(app)

# Custom Content Security Policy to whitelist bootstrap CDN libraries
csp = {
    'default-src': [
        '\'self\'',
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css'
    ],
    'script-src': [
        '\'self\'',
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js',
        'https://cdnjs.cloudflare.com/ajax/libs/web3/4.4.0/web3.min.js',
    ]
}
# protect against XSS attacks
talisman = Talisman(app, content_security_policy=csp)


@app.route('/')
@limiter.limit("5 per minute")
def index():
    return render_template('index.html')


# RBAC Wrapper Function to manage user access to views.
# This function is used as a decorator to specify which roles
# can access each view function.
def roles_required(*roles):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if current_user.is_anonymous:
                logging.warning('SECURITY - Invalid Anonymous Access Attempt '
                                '[%s, accessed: %s, previous page: %s]',
                                request.remote_addr,
                                request.full_path,
                                request.referrer)
                flash('Please login to access this page')
                return redirect(url_for('user.login'))
            if current_user.role.role_name not in roles:
                logging.warning('SECURITY - Invalid User Access Attempt '
                                '[%s, %s, %s, %s, accessed: %s, previous page: %s]',
                                current_user.id,
                                current_user.email,
                                current_user.role.role_name,
                                request.remote_addr,
                                request.full_path,
                                request.referrer)
            if current_user.role.role_name not in roles:
                return render_template('error_pages/403.html')
            return f(*args, **kwargs)
        return wrapped
    return wrapper


# Utility function to ensure redirects go only to safe URLs, protecting against open redirect vulnerabilities.
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return (test_url.scheme in ('http', 'https') and
            ref_url.netloc == test_url.netloc)


# import blueprints (placing here avoids cyclic import error)
from user.views import user_blueprint
from data_submission.views import data_submission_blueprint
from forecast_prediction.views import forecast_prediction_blueprint
from admin.views import admin_blueprint
from update_model.views import update_model_blueprint
from test_model.views import test_model_blueprint
from retrain_model.views import retrain_model_blueprint

# register blueprints
app.register_blueprint(user_blueprint)
app.register_blueprint(data_submission_blueprint)
app.register_blueprint(forecast_prediction_blueprint)
app.register_blueprint(admin_blueprint)
app.register_blueprint(update_model_blueprint)
app.register_blueprint(test_model_blueprint)
app.register_blueprint(retrain_model_blueprint)

# initialise the LoginManager to manage user sessions
login_manager = LoginManager()
login_manager.login_view = 'user.login'  # redirect anonymous user to the login page
login_manager.init_app(app)

# avoid circular import error
from db_models import User


# the user_loader callback reloads a user from the database given their user ID stored in the session
@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


@app.errorhandler(403)
def forbidden(error):
    return render_template('error_pages/403.html'), 403


if __name__ == '__main__':
    app.run(debug=True)
