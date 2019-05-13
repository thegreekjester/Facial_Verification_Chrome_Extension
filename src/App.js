/*global chrome*/
import React from 'react';
import './App.css';

class App extends React.Component {
  constructor(props){
    super(props);
    this.state = {

    }
  }
    componentDidMount() {
      chrome.tabs.executeScript(null, {
        file: "contentScript.js"
      });
      this.grabCamera()
  }

  grabCamera(){
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

  takePicture(){
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var video = document.getElementById('video');
    context.drawImage(video, 0, 0, 640, 480);
    var image = new Image();
    image.src = canvas.toDataURL("image/png");
    this.setState({img:image})
  }

  render() {
    if(this.state.image){
      return (
        <div className="App">
          <img src={this.state.image} alt="image_for_app"></img>
      </div>
      );

    }else{

    return (
      <div className="App">
        <video id="video" width="80%" height="80%" autoplay></video>
        <button onClick={() => this.takePicture()} id="snap">Snap Photo</button>
        <canvas id="canvas" width="640" height="480"></canvas>
      </div>
    );
  }
}
}

export default App;
