// general function for toggling blocks
function show_block(prefix) {
    return function() {

        // reset other blocks
        for (var p of ["video", "read", "pnp"]) {

            // de-highlight buttons
            document.getElementById(p + "_button").classList.remove("clicked");

            // hide thumbnails
            document.getElementById(p + "_thumbnails").style.display = "none";
        }
        
        // highlight button
        document.getElementById(prefix + "_button").classList.add("clicked");

        // show thumbnails
        document.getElementById(prefix + "_thumbnails").style.display = "block";
    }
}

// set jqueries
for (var prefix of ["video", "read", "pnp"]) {
    $("#" + prefix + "_button").click(show_block(prefix));
}

// subscribe submit button
$("#subscribe_button").click(
    function(){

        // make success message visible
        var box = document.getElementById("subscribe_success_box");
        box.style.display = "block";
        
        // position message
        var msg = document.getElementById("subscribe_success_message");
        const xy = $("#subscribe_form").offset();
        const width = document.getElementById("subscribe_form").clientWidth;
        msg.style.top = xy.top + "px";
        msg.style.left = xy.left + "px";
        msg.style.width = width + "px";
        msg.style.position = "absolute";
        msg.style.opacity = 0;

        // fade-out subscribe options
        $("#subscribe_form").fadeTo(1000, 0);

        // fade-in sucess message
        $("#subscribe_success_message").fadeTo(1000, 1);

        // send email to server
        const url = "https://titansapi.eastus.cloudapp.azure.com/subscribe/"
            + encodeURIComponent(document.getElementById("subscribe_email").value);
        fetch(url);
    }
);

// questions link
$("#questions_link").click(
    function() {

        // fade-out prompt
        $("#questions_prompt").fadeTo(1000, 0);

        // fade-in box
        var form = document.getElementById("questions_form");
        form.style.display = "block";
        form.style.opacity = 0;
        $("#questions_form").fadeTo(1000, 1);
    }
);

// questions submit button
$("#questions_button").click(
    function(){

        // make success message visible
        var box = document.getElementById("questions_success_box");
        box.style.display = "block";
        
        // position message
        var msg = document.getElementById("questions_success_message");
        const xy = $("#questions_form").offset();
        const width = document.getElementById("questions_form").clientWidth;
        msg.style.top = xy.top + "px";
        msg.style.left = xy.left + "px";
        msg.style.width = width + "px";
        msg.style.position = "absolute";
        msg.style.opacity = 0;

        // fade-out subscribe options
        $("#questions_form").fadeTo(1000, 0);

        // fade-in sucess message
        $("#questions_success_message").fadeTo(1000, 1);

        // send questions to server
        const url = "https://titansapi.eastus.cloudapp.azure.com/comment/"
            + "?comment="
            + encodeURIComponent(document.getElementById("questions_box").value)
            + "&email="
            + encodeURIComponent(document.getElementById("questions_email").value);
        fetch(url);
    }
);
