// fade in after intro movie
document.getElementById('loading_video').addEventListener(
    'ended',
    function() {
        
        // preserve body height
        const body_height = document.getElementById('loading_video').clientHeight + "px";
        for (const element of ['body', 'video_cover']) {
            document.getElementById(element).style.height = body_height;
        }

        // fade-out loading screen
        $("#loading_video").fadeOut(1000);

        // fade-in header/footer
        $("#header").fadeTo(1000, 1);
        $("#footer").fadeTo(1000, 1);

        // make video cover image visible
        document.getElementById("video_cover").style.display = "block";
    },
    false
);
