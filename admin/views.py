import os.path
from web3 import Web3
from web3.exceptions import InvalidAddress, TimeExhausted
from flask import Blueprint, render_template, url_for, redirect, flash, request
from admin.forms import AddClientForm
from app import roles_required, app
from db_models import User, change_role
from flask_login import current_user
import logging

admin_blueprint = Blueprint('admin', __name__, template_folder='templates')


@admin_blueprint.route('/admin/dashboard', methods=['GET', 'POST'])
@roles_required('admin')
def admin_dashboard():
    # read contents of security-events.log into the template
    base_dir = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    log_path = os.path.join(log_dir, 'security-events.log')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = f.readlines()
        print("Here are the logs: ", logs)
        return render_template('admin/admin_dashboard.html', logs=logs)
    else:
        logs = []
        print("Here are not the logs: ", logs)
    return render_template('admin/admin_dashboard.html', logs=logs)


@admin_blueprint.route('/admin/add_client', methods=['GET', 'POST'])
@roles_required('admin')
def add_client():
    form = AddClientForm()
    if form.validate_on_submit():
        new_client = User.query.filter_by(username=form.username.data).first()

        if not new_client:
            flash('User does not exist, enter correct username')
            return render_template('admin/add_client.html', form=form)

        client_addr = new_client.wallet_address
        print(client_addr)
        # update user to client in the database
        change_role(new_client.username, 'client')

        if (client_addr[0:2] == '0x') and (len(client_addr) == 42):
            # set up web3 connection to contract
            w3 = Web3(Web3.HTTPProvider(app.config['GANACHE_URL']))
            contract = w3.eth.contract(address=app.config['CONTRACT_ADDR'], abi=app.config['ABI'])
            try:
                w3.eth.default_account = current_user.wallet_address
            except InvalidAddress:
                flash('Your wallet address is not registered in Ganache', 'danger')
                return redirect(url_for('index'))

            # add client to smart contract
            tx = contract.functions.addClient(client_addr).transact()
            try:
                w3.eth.wait_for_transaction_receipt(tx, 20)
                flash('Client submitted')
            except TimeExhausted:
                flash('Timeout')

            # log the event
            logging.warning('SECURITY - Client Added [%s] Added by [%s, %s]',
                            new_client.username,
                            current_user.username,
                            request.remote_addr)

            return redirect(url_for('index'))
        flash('Invalid client address format')
        return render_template('admin/add_client.html', form=form)
    return render_template('admin/add_client.html', form=form)
