// click cover to play video
$('#video_cover').click(function(){
    
    // get elements
    var primary_video = document.getElementById("primary_video");
    var video_cover = document.getElementById("video_cover");
    var body = document.getElementById("body");

    // set height and display
    primary_video.style.display = "block";
    const height = document.getElementById("video_cover").clientHeight + "px";
    for (const element of [primary_video, body]) {
        element.style.height = height
    }

    // hide cover
    $('#video_cover').fadeOut(1);

    // play video
    primary_video.play();
})
