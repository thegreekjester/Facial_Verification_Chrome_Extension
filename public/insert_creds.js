var userArray = ['User', 'user', 'email', 'Email', 'id', 'ID']
var passArray = ['pass', 'Pass']
var password;
var username;
for (var i=0; i<userArray.length; i++){
    var str = userArray[i];
    var element = document.querySelector(`input[placeholder*="${str}"]`)
    if(element !== null){
        username = element;
        break;
    }
}

for (var j=0; j<passArray.length; j++){
    var str = passArray[j]
    var element = document.querySelector(`input[placeholder*="${str}"]`)
    if(element !== null){
        password = element;
        break;
    }
}


username.value = config.username

password.value = config.password
