## Facial Verification Log-in Chrome Extension
Extension to enable facial verification for account login within your Chrome browser. Supply your siamese model with a short video recording and enjoy a "Face ID" kind of authentication right in your browser!  

## Motivation
I thought of creating this extension after first witnessing the iPhone X's Face ID feature first hand. Having an interest in computer vision I was amazed with being able to use your face as a way to sign in. When thinking about my day-to-day, I was constantly using/forgetting passwords while browsing the internet, so this seemed like a great project to take on

 
## Screenshots

**Introduce your facial features to the model for later verification:**

![Training](readme_files/training.gif=250x200)

**Verification:**

![Verification](readme_files/verification.gif=250x200)

## Tech/framework used

- [React](https://reactjs.org/)
- [Keras](https://keras.io)
- [OpenCV](https://opencv.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Flask](https://www.fullstackpython.com/flask.html)
- [Chrome API](https://developer.chrome.com/extensions/api_index)
- [Chrome Extensions](https://developer.chrome.com/extensions)

## Packages:
- [Base64](https://docs.python.org/3/library/base64.html)
- [Numpy](https://numpy.org/)
- [PIL](https://www.pythonware.com/products/pil/)
- [Axios](https://www.npmjs.com/package/axios)
- [Papaparse](https://www.npmjs.com/package/papaparse)


## Features

- Facial verification
- Facial descriptor via siamese model
- Automatically applies username and password to logins

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation

## Set up

1. **pip install all relevant packages (listed above in tech/frameworks)**

2. **Grab your Chrome Passwords by exporting it to CSV**

- https://www.cyclonis.com/how-export-passwords-csv-file-from-google-chrome/ 

3. **Put Chrome Passwords CSV into src folder**

4. **Go to ./server/flask_server.py and add your name**

```python

person = 'user' # replace with your name
model = None # do not change
graph = None # do not change

```

4. **Go to ./src/App.js and update the Chrome Passwords csv path for the variable data**
```javascript
/*global chrome*/
import React from 'react';
import './App.css';
import axios from 'axios';
import data from 'insert_chrome_passwords_csv_here';
import Papa from 'papaparse';
var t;
var mediaRecorder;
```

## Usage

1. **After Set up, run "npm run build" to create out the new build folder**

2. **Go to "chrome://extensions/" in your browser and upload the build folder you just created as a new extension**

3. **Go to ./server and run "export FLASK_APP=flask_server.py" in the terminal**

4. **Start Flask server by running "flask run" in terminal**

5. **Train the Neural Network by Capturing a video of you, then press "train model"; You should receive a message saying model is successfully trained**

6. **Go to a website that your chrome browser has saved the passwords for, then open up extension and take a photo. Then Click "keep photo". If the model recognized you, the user name and password should populate the login**

7. **Congrats! You now can sign in with your facial features!!**

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests
Describe and show how to run the tests with code examples.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Peter Katsos]()