// // This file is used to connect a metamask wallet with the application in the frontend.
// // source for JS code: https://devdojo.com/omarthedev/metamask-discord
window.walletAddress = null;

const connectWallet = document.getElementById('connectWallet');
// Hidden input field in register form to get wallet address
const walletInput = document.getElementById('wallet_address');
const walletBalance = document.getElementById('walletBalance')

// retrieves the wallet balance from the ethereum API
async function getWalletBalance() {
    const balance = await window.ethereum.request({
        method: 'eth_getBalance',
        params: [window.walletAddress, 'latest']
    })
    .catch((e) => {
        console.error(e.message)
    })
    if (!balance) { return }

    walletBalance.innerText = 'ETH Balance: ' + parseFloat(balance) / Math.pow(10, 18)
}

// Check if MetaMask is installed
function checkInstalled() {
    if (typeof window.ethereum === 'undefined') {
        alert("Please install MetaMask to continue.");
        return false;
    }
    connectWallet.addEventListener('click', connectWalletWithMetaMask);
}

// Connect MetaMask and fetch the wallet address
async function connectWalletWithMetaMask() {
    try {
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
        if (!accounts || accounts.length === 0) return;

        // add web3.js to ensure wallet addresses from metamask are checksummed
        const web3 = new Web3(window.ethereum);
        window.walletAddress = web3.utils.toChecksumAddress(accounts[0]);

        // Fetch nonce from backend
        const response = await fetch('/gen_nonce', {
            method: 'POST',
            body: new URLSearchParams({ wallet_address: window.walletAddress }),
        });
        const nonce = await response.text();
        if (!nonce) {
            alert("Failed to fetch nonce from server");
            return;
        }

        // message to sign
        const message = `sign this message to verify wallet and register account: ${nonce}`;

        // Sign the nonce with MetaMask
        const signature = await window.ethereum.request({
            method: 'personal_sign',
            params: [message, window.walletAddress],
        });

        // Send the signature to the backend
        const response2 = await fetch('/verify_signature', {
            method: 'POST',
            body: new URLSearchParams({ wallet_address: window.walletAddress, signature   }),
        });
        const data = await response2.text();
        if (!data) {
            alert("Failed to verify signature with server");
            return;
        }

        // Update hidden input field to submit with the form
        walletInput.value = window.walletAddress;
        console.log("Wallet Address Set:", walletInput.value);

        connectWallet.innerText = 'Disconnect';
        connectWallet.removeEventListener('click', connectWalletWithMetaMask);
        setTimeout(() => {
        connectWallet.addEventListener('click', signOutOfMetaMask)
        }, 200 );
        await getWalletBalance();
    } catch (error) {
        console.error("MetaMask Connection Failed", error);
    }
}

// disconnect the wallet. Does not remove the wallet from the site
// only clears the wallet address and balance
function signOutOfMetaMask() {
    window.walletAddress = null;
    walletBalance.innerText = '';
    walletInput.value = '';  // Clear hidden input

    connectWallet.innerText = 'Connect';
    connectWallet.removeEventListener('click', signOutOfMetaMask);
    connectWallet.addEventListener('click', connectWalletWithMetaMask);
}

// Initialize MetaMask check on page load
window.addEventListener('DOMContentLoaded', () => {
    checkInstalled();
});