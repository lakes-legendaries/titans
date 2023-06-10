// ====================================================================================
// Camera controls
// ====================================================================================

// ====================================================================================
// Interface
var camera = {
	get_x0   : function() {}, // return left-most  x of visible windpw
	get_xmax : function() {}, // return right-most x of visible window
	get_midpt: function() {}, // return center     x of visible window
	reset    : function() {}, // to play table
	toggle   : function() {}, // play table <-> buy piles
	setup    : function() {}, // phaser plugin
	update   : function() {}, // phaser plugin
	at_left  : function() {return camera.to_x == camera.left}, // returns true if at default position
	at_right : function() {return camera.to_x == camera.right}, // returns true if at default position
	to_right : function() {camera.to_x = camera.right;},       // move camera to the buy piles (right)
	active   : function() {}, // return whether camera is moving
}

camera.get_x0 = function() {
	return env.physics.cameras.main.scrollX;
}

camera.get_xmax = function() {
	return env.physics.cameras.main.scrollX + env.physics.cameras.main.width;
}

camera.get_midpt = function() {
	return env.physics.cameras.main.scrollX + env.physics.cameras.main.width / 2;
}

camera.reset = function() {
	camera.to_x = camera.left;
}

camera.toggle = function() {
	camera.to_x = camera.to_x == camera.left? camera.right: camera.left;
}

camera.setup = function() {
	
	// initial config
	camera.to_x  = camera.left;
	
	// build
	camera.button = env.add.sprite(camera.button_pos, 'camera');
	camera.button.setInteractive();
	camera.button.on('pointerdown', () => {camera.toggle()});
}

camera.update = function() {
	let dist = camera.to_x - env.physics.cameras.main.scrollX;
	if (dist == 0) {return;}
	if (Math.abs(dist) < camera.speed) {
		env.physics.cameras.main.scrollX = camera.to_x;
		return;
	}
	env.physics.cameras.main.scrollX += (dist > 0? 1: -1) * camera.speed;
	camera.button.setFrame(camera.to_x == camera.right? camera.button_frame.to_left: camera.button_frame.to_right);
}

camera.active = function() {
	return camera.to_x != env.physics.cameras.main.scrollX;
}

// ====================================================================================
// Backend
camera.left  = 0;
camera.right = env.scene.x - env.window.x;
camera.speed = 30;
camera.button_frame = {to_right: 0, to_left: 1};
camera.button_pos = {x: 1872.5, y: 980};