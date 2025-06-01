import web3.exceptions
from flask import Blueprint, redirect, url_for, session, flash
from app import limiter, app
from web3 import Web3
from web3.exceptions import InvalidAddress
from flask_login import current_user
import json

update_model_blueprint = Blueprint('update_model', __name__, template_folder='templates')


@update_model_blueprint.route('/update_model', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def update_model():
    # new model available on IPFS to set as primary

    # set up web3 connection to contract
    w3 = Web3(Web3.HTTPProvider(app.config['GANACHE_URL']))
    contract = w3.eth.contract(address=app.config['CONTRACT_ADDR'], abi=app.config['ABI'])
    try:
        w3.eth.default_account = current_user.wallet_address
    except InvalidAddress:
        flash('Your wallet address is not registered in Ganache', 'danger')
        return redirect(url_for('index'))

    # listen for a model decision
    event_filter = contract.events.VoteDecided.create_filter(from_block='earliest')
    new_events = event_filter.get_all_entries()
    if len(new_events) > 0:

        # below code functionality from: [1]
        event = new_events[len(new_events - 1)]
        event = Web3.to_json(event)
        event = json.loads(event)
        if event['args']['decision'] == 'true':
            ipfs_hash = event['args']['hash']
            # end of code functionality from: [1]

            if (ipfs_hash not in session['hashes'].keys()) or (session['hashes'][ipfs_hash] == 1):

                # TODO: use ipfs hash to pull new model from ipfs
                # TODO: decrypt model
                # TODO: set model as primary
                pass
                session['hashes'][ipfs_hash] = 2

    return redirect(url_for('index'))

# References
# [1] O. Brown, ‘07_Subscribing_To_ContractEvents.py’, 2023, Console Cowboys. Accessed: Mar. 16, 2025. [Online]. Available: https://github.com/cclabsInc/Python-SmartContact-BlockchainExploitation/blob/main/Part1_Manual_Interactions/Contract_Interactions/07_Subscribing_To_ContractEvents.py