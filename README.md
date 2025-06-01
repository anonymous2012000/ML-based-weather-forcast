# Decentralized-Weather-Forecast-System

## Flask Backend Setup
 ### Flask app
 1. Recommended to create a virtual environment in your local repo
 2. Set the python interpreter for the virtual environment
 3. Set the templating language to Jinja2
 4. Install all requirements in requirements.txt
 ### Local version of user account database initialisation
 1. access the python console/shell within the project root directory
 2. enter the following commands
 ```pythonconsole
 >>> from app import db
 >>> from db_models import init_db
 >>> init_db()
 ```
 
 ## Ganache, truffle and MetaMask setup (initally from Benito, edited by Elliot)
 1. Install WSL
 2. Install Node.js
 3. Run:
 ```commandline
 npm install -g truffle
 ```
 4. Run:
 ```commandline
 truffle version
 ```
 5. Install Ganache for Windows
 6. Create a folder in the local filesystem called 'deployment' or similar:
 ```commandline
 mkdir deployment
 ```
 7. Cd into the folder
 ```commandline
 cd deployment
 ```
 8. Run:
 ```commandline
 truffle init
 ```
 9. Observe that new folders in the 'deployment' folder have been created.
 10. Copy and paste the smart_contract.sol file into the 'contracts' folder.
 11. Create a .js file in the 'migrations' folder and paste the following:
 ```
 const Smart = artifacts.require("Smart");
 
 module.exports = function(deployer) {
   deployer.deploy(Smart);
 };
 ```
 12. Go to the 'config.js' file in the 'deployment' folder and ensure the following information is present and un-commented (it must match the RPC and network ID shown in Ganache):
 ```
 module.exports = {
   networks: {
     development: {
       host: "127.0.0.1",
       port: 7545,
       network_id: "5777"
     }
   },
 
   compilers: {
     solc: {
       version: "0.8.13",
     }
   }
 };
 ```
 13. Add the config.js file to the workspace in Ganache.
 13. Ensure you are inside the 'deployment' folder and run the following command:
 ```commandline
 truffle migrate
 ```
 14. Note down the contract address to use for sending transactions and receiving events.
 15. Install the MetaMask browser extension.
 16. Add a custom network to MetaMask with the following details:
 ```commandline
 Name : Truffle
 Default RPC URL : http://127.0.0.1:7545
 Chain ID : 1337
 Currency Symbol : ETH
 ```
 17. Import accounts from Ganache into MetaMask using their private keys.
 
 ## Connecting with a Local IPFS Node
 1. Install IPFS Kubo.
 2. Add ipfs.exe to the system PATH.
 3. Open Command Prompt and go to the directory where ipfs.exe is located.
 4. Run:  
 ```commandline
 ipfs daemon
 ```
 5. Update the url variable in the store_model_ipfs() function in Regression.py with the port number where the server is listening for incoming API requests. The output will show as:
 ```commandline
 RPC API server listening on /ip4/127.0.0.1/tcp/<port>
 ```
 6. Run the Regression.py.

## ML model datasets and processing

### Old Source Data Archive
This folder contains data sources that were once planned to be used in the project. For some reason (which will be discussed in the documentation) we opted to not use this data in the final model

### Source data:
This folder contains the source datasets as they were before being modified in the project. This folder maintains these as original files should errors occur, and to reference back to the original dataset. The link to the kaggle notebook from which this data was sourced is below:
https://www.kaggle.com/code/wesleytan/weather-prediction-knn-and-moving-average/notebook

### File verification
This file ensures that all csvs follow the same format so that problems dont arise during preprocessing

### Data preprocessing
This file ensures the dataset itself is ready for use in the machine learning model(s). This involves checking for empty values, verifying data types, encoding to ML ready format etc.


# Important note!!!
CSVs currently are ugly. Need to delete all shape columns as they are used to create graphs that are not necessary to this. Also need to change data & type so that its maybe 1 or 2 d.p. or integer values.