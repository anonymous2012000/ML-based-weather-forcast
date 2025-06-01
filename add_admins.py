from web3 import Web3
from web3.exceptions import TimeExhausted
from dotenv import load_dotenv
import os

load_dotenv()

# set up web3 connection to contract
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

contract_address = os.getenv('CONTRACT_ADDR')
abi = os.getenv('ABI')
contract = w3.eth.contract(address=contract_address, abi=abi)

# switch to owner
w3.eth.default_account = w3.eth.accounts[0]

# set up admins
tx = contract.functions.addPrivileged([w3.eth.accounts[i] for i in range(5)]).transact()
try:
    w3.eth.wait_for_transaction_receipt(tx, 20)
    print('Successfully added admins')
except TimeExhausted:
    print('Failed to add admins')

print('Successfully added the following admin accounts')
for i in range(5):
    print(w3.eth.accounts[i])

