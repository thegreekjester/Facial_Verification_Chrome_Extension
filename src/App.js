/*global chrome*/
import React from 'react';
import './App.css';

class App extends React.Component {
    componentDidMount() {
      chrome.tabs.executeScript(null, {
        file: "contentScript.js"
      });
      chrome.runtime.onMessage.addListener(
        function (request, sender, sendResponse) {
          if (request.success){
            // Grab elements, create settings, etc.
            var video = document.getElementById('video');

          // Get access to the camera!
          if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Not adding `{ audio: true }` since we only want video now
            navigator.mediaDevices.getUserMedia({
              video: true
            }).then(function (stream) {
              //video.src = window.URL.createObjectURL(stream);
              video.srcObject = stream;
              video.play();
            });
          }
        }

        });
  }
  render() {
    return (
      <div className="App">
        <video id="video" width="80%" height="80%" autoplay></video>
        <button id="snap">Snap Photo</button>
        <canvas id="canvas" width="640" height="480"></canvas>
      </div>
    );
  }
}

export default App;
