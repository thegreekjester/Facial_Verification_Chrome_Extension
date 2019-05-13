/*global chrome*/
if('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices){
    navigator.mediaDevices.getUserMedia({video: true})
        .then(function(){
            chrome.runtime.sendMessage({success: true})
        })

  }