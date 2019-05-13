/*global chrome*/
let ifr = document.createElement('iframe');
ifr.setAttribute('allow', 'microphone; camera'); //necessary for cross-origin frames that request permissions
ifr.src = chrome.runtime.getURL('popup.html');
document.body.appendChild(ifr);