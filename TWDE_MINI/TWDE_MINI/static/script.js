function getImgNum(){
    var imgNum = document.getElementsByClassName("pfp_block").length;
    return imgNum;
}

function getClickedId(ID){
    console.log("The ID is: " + ID + " The num is "+ getImgNum()); 
    return ID;
}

function showcase(ID){
    // Obtain the element of transparent background
    var showcase_bg = document.getElementsByClassName("showcase_bg");
    var showcase_container = document.getElementsByClassName("showcase_container");

    // Obtian and copy the image that we going to show
    var img = document.getElementById(ID);
    var img_copy = img.cloneNode(true);
    img_copy.removeAttribute("onclick");
    img_copy.removeAttribute("id");
    img_copy.removeAttribute("class");
    img_copy.id = "show";

    showcase_bg[0].style.display = 'block';
    showcase_container[0].appendChild(img_copy);   
}

function closeShowcase(){
    // Obtain the element of transparent background
    var showcase_bg = document.getElementsByClassName("showcase_bg");
    var img = document.getElementById("show");

    // Set showcase to invisible
    showcase_bg[0].style.display = 'none';

    // remove image
    img.parentElement.removeChild(img);
}

function menu(){
    // Menu button animation
    var menu_button = document.getElementsByClassName("Menu_button")[0];
    var bar1 = document.getElementById("bar1");
    var bar2 = document.getElementById("bar2");
    var bar3 = document.getElementById("bar3");

    // show and hide menu
    var function_buttons = document.getElementsByClassName("Menu_function");
    var icons = document.getElementsByClassName("icon");
    
    if (menu_button.id == "open"){
        // Remove opening class if opened
        function_buttons[0].classList.remove("opening");
        function_buttons[1].classList.remove("opening");
        function_buttons[2].classList.remove("opening");
        bar1.classList.remove("opening");
        bar2.classList.remove("opening");
        bar3.classList.remove("opening");
        
        // set icons to invisible
        for (var i = 0; i < icons.length; i++){
            icons[i].style.display = "none";
        }
        menu_button.id = "close";
    } else { 
        // Apply opening  class if closed     
        function_buttons[0].classList.add("opening");
        function_buttons[1].classList.add("opening");
        function_buttons[2].classList.add("opening");
        bar1.classList.add("opening");
        bar2.classList.add("opening");
        bar3.classList.add("opening");

        menu_button.id = "open";
        for (var i = 0; i < icons.length; i++){
            icons[i].style.display = "block";
        }
    }
}


function request_image(main_container){
    // Link the main container
    
    // Create an image element
    var img_container = document.createElement('div');
    img_container.className = 'pfp_block';
    var img = document.createElement('img');

    // Request an generated image from flask
    let image_request = new XMLHttpRequest();
    image_request.open('GET', '/image');
    image_request.send();
    image_request.onload = function(){
        img.src = "data:image/jpeg;base64," + image_request.response;
        img.className = 'pfp';
        img.id = main_container.childElementCount;
        img.onclick = function(){
            showcase(this.id);
        }

        img_container.appendChild(img);
        main_container.appendChild(img_container);
        if(play_button.childNodes[1].id == "pause"){
            img_container.onload = request_image(main_container);   
        }        
    }
}

function start_n_pause(ID){
    var function_button = document.getElementById(ID);
    if(play_button.childNodes[1].id == "play"){
        play_button.childNodes[1].id = "pause";
    }else{
        play_button.childNodes[1].id = "play";
    }
    
    if(play_button.childNodes[1].id == "play"){
        function_button.children[0].src = "static/icons/play.png";
    }else{
        function_button.children[0].src = "static/icons/pause.png";
    }

    if(play_button.childNodes[1].id == "pause"){
        request_image(main); 
    }
}

function clear(){
    main = document.getElementById("main_container");
    console.log(main);
    main.clear;
}



var main;

window.onload = function init(){
    main = document.getElementById("main_container");
    var play_button = document.getElementById("play_button");
    console.log(play_button.childNodes[1].id);
    if(play_button.childNodes[1].id == "pause"){
        request_image(main); 
    }    
}
