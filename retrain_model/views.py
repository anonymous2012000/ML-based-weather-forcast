import os

import ipfshttpclient
import web3.exceptions
from flask import Blueprint, flash, redirect, url_for, request, render_template
from app import limiter, app, roles_required
from web3 import Web3
from web3.exceptions import InvalidAddress, TimeExhausted
from flask_login import current_user
from models.SKTIME_models import Regression
from retrain_model.forms import RetrainModelForm
import logging

retrain_model_blueprint = Blueprint('retrain_model', __name__, template_folder='templates')


@retrain_model_blueprint.route('/retrain_model', methods=['GET', 'POST'])
@roles_required('client')
@limiter.limit("5 per minute")
def retrain_model():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('filename'))
    form = RetrainModelForm()
    if form.validate_on_submit():
        if form.submit.data:
            # set up web3 connection to contract
            w3 = Web3(Web3.HTTPProvider(app.config['GANACHE_URL']))
            contract = w3.eth.contract(address=app.config['CONTRACT_ADDR'], abi=app.config['ABI'])
            try:
                w3.eth.default_account = current_user.wallet_address
            except InvalidAddress:
                flash('Your wallet address is not registered in Ganache', 'danger')
                return render_template('index.html')

            # retrain the model on submitted data
            model_path = Regression.retrain(filepath)

            # TODO: store new model on IPFS
            ipfs_hash = Regression.store_model_ipfs(model_path)

            # send transaction
            tx = contract.functions.storeHash(ipfs_hash).transact()
            # receive response
            try:
                w3.eth.wait_for_transaction_receipt(tx, 20)
                flash('IPFS hash successfully stored')
                # log the event
                logging.warning('SECURITY - Model retrained and IPFS hash stored [%s], By [%s, %s, %s]',
                                ipfs_hash,
                                current_user.email,
                                current_user.role.role_name,
                                request.remote_addr)
            except TimeExhausted:
                flash('Timeout occurred sending transaction', 'danger')
            return render_template('index.html')
        elif form.cancel.data:
            if os.path.exists(filepath):
                os.remove(filepath)
                flash("Retraining cancelled, the uploaded file has been deleted")
                return redirect(url_for('index'))
            else:
                flash("Retraining cancelled")
                return redirect(url_for('index.html'))
        else:
            return redirect(url_for('index'))
    # render the template with the form
    return render_template('retrain_model/retrain_model.html', form=form, filename=request.args.get('filename'))
