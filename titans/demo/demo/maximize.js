// ====================================================================================
// Maximize -- button to go fullscreen
// ====================================================================================

// ====================================================================================
// Interface
var maximize = {
	setup : function(scale) {}, // phaser plugin
	update: function()      {}, // phaser plugin
	scale : null,               // access to the scale controller
}

maximize.setup = function(scale) {
	maximize.scale  = scale;
	maximize.button = env.add.sprite(maximize.button_loc, 'full screen button');
	maximize.button.setInteractive().on('pointerdown', () => {maximize.scale.toggleFullscreen();});
}

maximize.update = function() {
	maximize.button.setFrame(maximize.scale.isFullscreen? 1: 0);
}

// ====================================================================================
// Backend
maximize.button_loc = {x: 1582, y: 934};