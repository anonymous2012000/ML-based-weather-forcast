// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

/** 
 * @title Smart
 * @author Weaam Alrbeiqi, Elliot A. J. Hurley, Benito Vincent
 * Main smart contract that handles external communication and checks
 */
contract Smart {
    // Smart contract varibles:
    // Rating for each client
    mapping(address => uint) ratings;
    // Hash of primary model
    bytes main;
    // List of which client pushed what model
    mapping(bytes => address) clientModels;
    // Stores the owner of the smart contract
    address owner;
    // Tracks whether privileged users have been added yet
    bool privilegedAdded;
    // List of clients, true if client, false otherwise
    mapping(address => bool) clients;
    // List of admins, true if admin, false otherwise
    mapping(address => bool) admins;
    // Number of clients registered
    uint noOfClients;
    // Tracks model scores from clients
    mapping(bytes => mapping(address => uint)) modelScores;
    // Tracks all clients that gave a score on a model
    mapping(bytes => address[]) clientVotedScore;
    // List of final scores for each model
    mapping(bytes => uint) finalModelScores;

    // Constructor

    constructor() {
        owner = msg.sender;
        privilegedAdded = false;
        main = new bytes(32);
    }

    // Events:

    /**
    * Indicates to other clients a new model has been submitted
    * @param hash - Hash signature of the newly submitted model
    */
    event NewHash(bytes hash);

    /**
    * Indicates to clients that a majority has been reached on a vote
    * @param hash - Hash signature of the model in question
    * @param decision - Decision reached
    */
    event VoteDecided(bytes hash, bool decision);

    // Modifiers:

    /**
    * Modifies a function to be owner-only
    */
    modifier mustBeOwner {
        require(owner == msg.sender, "You must be an owner to access this function");
        _;
    }

    /**
    * Modifies a function to be client-only
    */
    modifier mustBeClient {
        require(clients[msg.sender], "You are not a registered client, contact a system admin");
        _;
    }

    // Functions:

    /**
    * Adds a new client to the client list if not already present, caller must be an admin
    */
    function addClient(address newClient) external {
        if (!clients[newClient] && admins[msg.sender]){
            clients[newClient] = true;
            noOfClients++;
        }
    }

    /**
    * Sets the first primary model hash, caller must be owner
    * @param hash - Hash of the primary model
    */
    function setMain(bytes calldata hash) mustBeOwner external {
        main = hash;
    }

    /**
    * Allows the smart contract owner to add 5 privileged users but only once
    * @param usersToAdd - List of clients that want privileged access
    */
    function addPrivileged(address[] calldata usersToAdd) mustBeOwner external {
        // Check the owner is the one calling the function and that the function hasn't been called before
        require(privilegedAdded == false, "Admins already added");
        require(usersToAdd.length == 5, "Invalid number of admins to add (must be 5)");
        privilegedAdded = true;
        // Iterate through each user in the list
        for (uint i = 0; i < usersToAdd.length; i++){
            // Adds users
            clients[usersToAdd[i]] = true;
            admins[usersToAdd[i]] = true;
            noOfClients++;
            // Sets the user's reputation to the max
            ratings[usersToAdd[i]] = 100;
        }
    }

    /** 
    * Takes in any data that is sent to the smart contract
    * @param hash - IPFS Hash data for a model
    */
    function storeHash(bytes calldata hash) mustBeClient external {
        require(clientModels[hash] == 0x0000000000000000000000000000000000000000, "Model already submitted");
        // Keep track of who submitted the new model
        clientModels[hash] = msg.sender;
        // Emit message to client
        emit NewHash(hash);
    }

    /**
    * Allows clients to give their score on a ML model
    * @param hash - Hash value of model in question
    * @param score - Score submitted by user on the model
    */
    function castMLScore(bytes calldata hash, uint score) mustBeClient external {
        require(finalModelScores[hash] == 0, "Model already given a final score");
        require(modelScores[hash][msg.sender] == 0, "You have already voted for this model");
        require(ratings[msg.sender] >= 10, "You do not have enough reputation to vote");
        // Add score to list of scores for that model
        modelScores[hash][msg.sender] = score;
        // Add client to list of clients that voted for that model
        clientVotedScore[hash].push(msg.sender);
        // Calculate total reputation between clients
        uint totalRep = 0;
        for (uint i = 0; i < clientVotedScore[hash].length; i++){
            totalRep += ratings[clientVotedScore[hash][i]];
        }
        // Check if the total reputation of clients exceeds a certain value
        if (totalRep > (noOfClients * 5)){
            uint finalScore = 0;
            // Calculate final score based on a weighted sum of client-submitted model scores (weights are based on client reputations)
            for (uint i = 0; i < clientVotedScore[hash].length; i++){
                finalScore += (modelScores[hash][clientVotedScore[hash][i]] * (ratings[clientVotedScore[hash][i]] / totalRep));
            }
            finalModelScores[hash] = finalScore;
            // Check if submitted model achieved a better score than the current one
            if (finalScore > finalModelScores[main]){
                main = hash;
                // Emit result (Tell clients to update to the newest model and that the vote is over for this hash)
                emit VoteDecided(hash, true);
            } else {
                // Emit result (Tell clients to ignore model and that the vote is over for this hash)
                emit VoteDecided(hash, false);
            }
            // Calculate new reputation for model submitter
            calculateReputation(hash);
        }
    }

    /**
    * Calculate reputation score to give to a user based off their model performance
    * @param model - Hash value of model
    */
    function calculateReputation(bytes memory model) private {
        // Get values and store for use later
        uint mlScore = finalModelScores[model];
        address modelOwner = clientModels[model];
        // Either issue postive or negative reputation based off the model score
        // Score 50 = +0 rep, Score 0 = -10 rep, Score 100 = +10 rep
        int finalUserScore = int(ratings[modelOwner]) + (int(mlScore) - 50) / 5;
        // Makes sure the final reputation score of user doesn't leave the boundries
        if (finalUserScore > 100) {
            finalUserScore = 100;
        }
        else if (finalUserScore < 0) {
            finalUserScore = 0;
        }
        ratings[modelOwner] = uint256(finalUserScore);
    }
    
}
