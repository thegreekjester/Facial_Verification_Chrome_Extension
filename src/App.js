// use npm run build to create optimized build for the chrome extension

/*global chrome*/
import React from 'react';
import './App.css';
import axios from 'axios'
var t;
var mediaRecorder;


class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {}
    // creating global instance of "this"
    t = this;
  }
  componentDidMount() {
    // onload of the react app, the contentScript.js is run
    // this takes care of injecting the iframe that asks for webcam permissions
    chrome.tabs.executeScript(null, {
      file: "contentScript.js"
    });
    // this captures messages sent from the iframe that is loaded onto the current page
    chrome.runtime.onMessage.addListener(
      function (request, sender, sendResponse) {
        if (request.success) {
          t.grabCamera()
        }
      });
  }

  /***************************** Camera *********************************/

  grabCamera() {
    // Grab elements, create settings, etc.
    var video = document.getElementById('video');
    // Get access to the camera!
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // Not adding `{ audio: true }` since we only want video now
      // This line returns a promise (the stream)
      navigator.mediaDevices.getUserMedia({
        video: true,
        audio:false
      }).then(function (stream) {
        // this stream is the webcam stream and we assign it to the video element
        video.srcObject = stream;
        video.play();

        /******* Media Recorder stuff *******/
        chrome.extension.getBackgroundPage().console.log('in media recorder area')
        var buffer = [] // video content array
        var options = {
          mimeType: 'video/webm'
        }
        mediaRecorder = new MediaRecorder(stream, options)
        // this function runs once data is available to write from the mediaRecorder object
        mediaRecorder.ondataavailable = (e) => {
          chrome.extension.getBackgroundPage().console.log('in ondata')
          // push the event data from the mediaRecorder obj into the buffer array
          buffer.push(e.data)
        }

        // when the media recorder is stopped, the following function runs
        mediaRecorder.onstop = (e) => {
          chrome.extension.getBackgroundPage().console.log('length of buffer array', buffer)

          // creating new blob (binary large obj) defining it as an webm file
          let blob = new Blob(buffer, {type:'video/webm'});
          chrome.extension.getBackgroundPage().console.log('this is the blob', blob)

          // clean up buffer array
          buffer = []
          // convert blob into object URL (can be used as video src)
          let videoURL = URL.createObjectURL(blob)
          t.setState({ blobURL: videoURL, video: blob })
        }
      })
      .catch((e)=>{
        chrome.extension.getBackgroundPage().console.log('this is your error', e)
      })
    }
  }

  // this function captures the frame and prints it to the canvas to be saved in local state
  takePicture() {
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var video = document.getElementById('video');
    context.drawImage(video, 0, 0, 350, 260);
    var image = canvas.toDataURL();
    this.setState({ img: image })
  }


  // this grabs the b64 encoded image from state and sends it to server 
  sendPhoto() {
    let data = new FormData();
    data.append('image', this.state.img)
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    }
    axios.post('http://localhost:5000', data, config)
    .then(function (response) {
      chrome.extension.getBackgroundPage().console.log(response.data);
    })
    .catch(function (error) {
      chrome.extension.getBackgroundPage().console.log(error);
    });
  }


  /********************************** Video *****************************************/

  // click handler for the record button
  videoPowerButton() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      // media recorder is in state, has been recording already (stop button)
      chrome.extension.getBackgroundPage().console.log('was already recording')
      let video = document.getElementById('video')
      let track = video.srcObject.getTracks()
      chrome.extension.getBackgroundPage().console.log('these are thet tracks', track)
      track.forEach((tr)=>{
        tr.stop()
      })
      chrome.extension.getBackgroundPage().console.log('this is stream', track)
      mediaRecorder.stop()
      chrome.extension.getBackgroundPage().console.log('stopped it', mediaRecorder.state)

    } else if (mediaRecorder) {
      // media recorder is in state but hasn't recorded yet (first button press)
      mediaRecorder.start()
      chrome.extension.getBackgroundPage().console.log('started the recording', mediaRecorder.state)
    } else {
      chrome.extension.getBackgroundPage().console.log('in else for some reason')

      //media recorder is not loaded into state for some reason
      return
    }
  }

  // takes video blob and sends it to the server
  sendVideo() {
    chrome.extension.getBackgroundPage().console.log('sending video')
    let data = new FormData();
    data.append('video', this.state.video)
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    }
    axios.post('http://localhost:5000/video', data, config)
      .then((response) =>{
        chrome.extension.getBackgroundPage().console.log(response.data)
        // setting the local state with the response from the server
        t.setState({vidResponse:response.data})
        
      })
      .catch((error) => {
        chrome.extension.getBackgroundPage().console.log(error)
      })
  }
  /*************************** General *******************************************/

  whatToRender() {
    chrome.extension.getBackgroundPage().console.log('new render')
    // renders if there IS an image captured and in local state
    if (this.state.img) {
      return (
        <div className='container'>
          {this.state.imgResponse === 'nothing found'? <h2 style={{'color':'#7CFC00'}}>Could not recognize you</h2>:<h2>{this.state.imgResonse}</h2> }
          <img src={this.state.img} alt='da_image_yo'></img>
          <button onClick={() => this.sendPhoto()} id="keep">Keep Photo</button>
          <button onClick={() => { this.setState({ img: undefined }, () => { this.grabCamera() }) }} id="retake">Retake</button>
        </div>
      );
    } else if (this.state.blobURL) {
      chrome.extension.getBackgroundPage().console.log('hello brah', this.state.video)
      return (
        <div className='container'>
          {this.state.vidResponse && <h2 style={{'color':'#7CFC00'}}>Succesfully trained the model on you!</h2>}
          <video id='recording' width='100%' height='100%' src={this.state.blobURL} controls></video>
          <button onClick={() => this.setState({ blobURL: '' }, () => { this.grabCamera() })}>Retake Video</button>
          <button onClick={() => this.sendVideo()}>Train Model</button>
        </div>
      );
    } 
    else {
      //renders if there is no image or video captured
      return (
        <div className='container'>
          <video id="video" width="100%" height="100%" autoplay></video>
          <button onClick={() => this.takePicture()} id="snap">Snap Photo</button>
          <button onClick={() => this.videoPowerButton()}>Record Video</button>
          <canvas id="canvas" height='260' width='350' style={{'display':'none'}}></canvas>
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
