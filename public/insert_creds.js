var userArray = ['User', 'user', 'email', 'Email']
var passArray = ['password', 'Password']
var password;
var username;
userArray.forEach(function(str){
    var element = document.querySelector(`input[placeholder*="${str}"]`)
    if(element !== null){
        username = element;
        console.log('we found it')
    }
})

passArray.forEach(function(str){
    var element = document.querySelector(`input[placeholder*="${str}"]`)
    if(element !== null){
        password = element;
        console.log('we found it password')
    }
})

username.value = config.username

password.value = config.password
