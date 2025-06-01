import joblib
import io
import web3.exceptions
from flask import Blueprint, render_template, flash, session, redirect, url_for
from fontTools.misc.eexec import encrypt
from app import limiter, app
from web3 import Web3
from web3.exceptions import InvalidAddress
from flask_login import current_user
import json
import requests
import random

test_model_blueprint = Blueprint('test_model', __name__, template_folder='templates')


@test_model_blueprint.route('/test_model', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def test_model():
    # new model available on IPFS to test and vote on

    # set up web3 connection to contract
    w3 = Web3(Web3.HTTPProvider(app.config['GANACHE_URL']))
    contract = w3.eth.contract(address=app.config['CONTRACT_ADDR'], abi=app.config['ABI'])
    try:
        w3.eth.default_account = current_user.wallet_address
    except InvalidAddress:
        flash('Your wallet address is not registered in Ganache', 'danger')
        return redirect(url_for('index'))

    # listen for a new model
    event_filter = contract.events.NewHash.create_filter(from_block='earliest')
    new_events = event_filter.get_all_entries()

    # new model available to test
    if len(new_events) > 0:
        for event in new_events:

            # below code functionality from: [1]
            event = Web3.to_json(event)
            event = json.loads(event)
            ipfs_hash = event['args']['hash']
            # end of code functionality from: [1]

            if ipfs_hash not in session['hashes'].keys():
                session['hashes'][ipfs_hash] = 1

                # TODO: use ipfs hash to pull new model from ipfs
                ipfs_url = f"https://ipfs.io/ipfs/{ipfs_hash}"  #Assuming ipfs_hash == cid
                response = requests.get(ipfs_url)
                if response.status_code == 200:
                    print("File downloaded from IPFS successfully")

                encrypted_model = joblib.load(io.BytesIO(response.content))
                print("Model is ready to use")

                # TODO: decrypt model
                # TODO: test model and generate score (0-100)
                model_score = random.randint(0, 100)

                # send transaction
                tx = contract.functions.castMLScore(ipfs_hash, model_score).transact()

                # receive response
                try:
                    w3.eth.wait_for_transaction_receipt(tx, 20)
                except web3.exceptions.TimeExhausted:
                    flash('Timeout')

    # redirect to index
    return redirect(url_for('index'))

# References
# [1] O. Brown, ‘07_Subscribing_To_ContractEvents.py’, 2023, Console Cowboys. Accessed: Mar. 16, 2025. [Online]. Available: https://github.com/cclabsInc/Python-SmartContact-BlockchainExploitation/blob/main/Part1_Manual_Interactions/Contract_Interactions/07_Subscribing_To_ContractEvents.py
