from flask import Blueprint, render_template, flash, redirect, url_for, request, jsonify, session
from user.forms import RegisterForm, LoginForm
from app import db, limiter, roles_required
from db_models import User, Role
import logging
import secrets
from eth_account import Account
from eth_account.messages import encode_defunct
from flask_login import login_user, current_user, logout_user, login_required
import bcrypt
from datetime import datetime


user_blueprint = Blueprint('user', __name__, template_folder='templates')


# generate a nonce for the user to sign in frontend
@user_blueprint.route('/gen_nonce', methods=['GET', 'POST'])
def gen_nonce():
    wallet_address = request.form.get('wallet_address')
    if not wallet_address:
        flash('Wallet address is required')
        return 400
    nonce = secrets.token_hex(16)
    session['nonce'] = nonce
    return nonce


# verify signature of the user from frontend
@user_blueprint.route('/verify_signature', methods=['GET', 'POST'])
def verify_signature():
    signature = request.form.get('signature')
    wallet_address = request.form.get('wallet_address')
    nonce = session.get('nonce')

    message = f"sign this message to verify wallet and register account: {nonce}"
    encoded_message = encode_defunct(text=message)
    recover_address = Account.recover_message(encoded_message, signature=signature)
    if recover_address != wallet_address:
        flash('Invalid signature')
        return render_template('index.html'), 400
    return render_template('index.html'), 200


@user_blueprint.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    form = RegisterForm()
    # handle form submission
    if form.validate_on_submit():
        # check if user already exists
        username = User.query.filter_by(username=form.username.data).first()
        if username:
            flash('Username already exists')
            return render_template('user/register.html', form=form)

        email = User.query.filter_by(email=form.email.data).first()
        if email:
            flash('Email already exists')
            return render_template('user/register.html', form=form)

        # get user role from db
        user_role = Role.query.filter_by(role_name='user').first()
        if not user_role:
            flash('Internal server error registering user')
            return 500
        # add new user to the database
        new_user = User(username=form.username.data,
                        email=form.email.data,
                        password=form.password.data,
                        wallet_address=form.wallet_address.data,
                        role=user_role)
        db.session.add(new_user)
        db.session.commit()

        # log user registration
        logging.warning('SECURITY - User Registered [%s, %s, %s]',
                        new_user.id,
                        new_user.email,
                        request.remote_addr)

        # send user to login page
        return redirect(url_for('user.login'))
    # reload the page if the form is not valid
    return render_template('user/register.html', form=form)


# Login function to authenticate users
@user_blueprint.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if current_user.is_authenticated:
            flash("Already logged in")
            return redirect(url_for('index'))

        # Query the database for the user.
        user = User.query.filter_by(email=form.email.data).first()
        if form.wallet_address.data != user.wallet_address:
            flash('Incorrect wallet address!')
            logging.warning('SECURITY - Invalid Login Attempt [%s, %s]',
                            form.email.data,
                            request.remote_addr)
            return render_template('user/login.html', form=form)
        # Validate that the user exists and that the password is correct using bcrypt.
        if not user and bcrypt.checkpw(form.password.data.encode('utf-8'), user.password):
            flash('Login Unsuccessful. Please check your username and password.', 'danger')
            logging.warning('SECURITY - Invalid Login Attempt [%s, %s]',
                            form.email.data,
                            request.remote_addr)
            return render_template('user/login.html', form=form)

        login_user(user)
        logging.warning('SECURITY - User Logged In [%s, %s, %s]',
                        current_user.id,
                        current_user.email,
                        request.remote_addr)
        user.last_login = user.current_login
        user.current_login = datetime.now()
        db.session.add(user)
        db.session.commit()

        flash('Welcome, ' + current_user.username + '!')
        if current_user.role.role_name == 'admin':
            return redirect(url_for('admin.admin_dashboard'))
        else:
            return redirect(url_for('user.profile'))

    return render_template('user/login.html', form=form)


# logout a user and log the event
@user_blueprint.route('/logout')
@login_required
def logout():
    logging.warning('SECURITY - User Logged Out [%s, %s, %s]',
                    current_user.id,
                    current_user.email,
                    request.remote_addr)
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('user.login'))


# user profile page
@user_blueprint.route('/profile')
@login_required
def profile():
    return render_template('user/profile.html')
