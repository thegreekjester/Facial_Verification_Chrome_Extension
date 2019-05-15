/*global chrome*/
import React from 'react';
import './App.css';
import axios from 'axios'

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  componentDidMount() {
    var t = this;
    chrome.tabs.executeScript(null, {
      file: "contentScript.js"
    });
    chrome.runtime.onMessage.addListener(
      function (request, sender, sendResponse) {
        if (request.success) {
          t.grabCamera()
        }
      });
  }

  grabCamera() {
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

  takePicture() {
    var t = this;
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var video = document.getElementById('video');
    context.drawImage(video, 0, 0, 350, 300);
    var image = canvas.toDataURL();
    this.setState({ img: image })
  }

  sendPhoto(){
    let data = new FormData();
    data.append('image', this.state.img)
    const config = {
        headers: {
            'content-type': 'multipart/form-data'
        }
    }
    axios.post('http://localhost:5000', data, config)
  }

  whatToRender() {
    if (this.state.img) {
      return (
        <div>
          <img src={this.state.img} alt='da_image_yo'></img>
          <button onClick={() => this.sendPhoto()} id="keep">Keep Photo</button>
          <button onClick={() => { this.setState({ img: undefined }, ()=>{this.grabCamera()})}} id="retake">Retake</button>
        </div>
      );
    } else {
      return (
        <div>
          <video id="video" width="80%" height="80%" autoplay></video>
          <button onClick={() => this.takePicture()} id="snap">Snap Photo</button>
          <canvas id="canvas" width="350" height="300"></canvas>
        </div>
      )
    }
  }

  render() {
    return (
      <div className='App'>
        {this.whatToRender()}
      </div>
    )
  }
}

export default App;
