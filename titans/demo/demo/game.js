// ====================================================================================
// Phaser object for controlling the game
// ====================================================================================

// ====================================================================================
// Backend
var game = {
	phaser : null,          // Phaser.Game object
	preload: function() {}, // Load game assets
	setup  : function() {}, // Initialize game
	update : function() {}, // Update game
	config : null,          // Game configuration
}

game.preload = function() {
	env    .preload(this);
	instr  .preload();
	loading.preload();
}

game.setup = function() {
	age       .setup();
	env       .setup();
	camera    .setup();
	log       .setup();
	maximize  .setup(game.phaser.scale);
	zone      .setup();
	subv      .setup();
	ref       .setup();
	card      .setup();
	highlight .setup();
	move      .setup();
	stats     .setup();
	tut       .setup();
	full_card .setup();
	instr     .setup();
	controller.setup();
	button    .setup();
	input     .setup();
	debug     .setup();
	loading   .setup();
}

game.update = function() {
	loading.update();
	if (!loading.done) {return;}
	camera    .update();
	maximize  .update();
	move      .update();
	stats     .update();
	tut       .update();
	full_card .update();
	instr     .update();
	controller.update();
	button    .update();
	debug     .update();
}

game.config = {
	type   : Phaser.AUTO,
	width  : env.window.x,
	height : env.window.y,
	physics: {default: 'arcade'},
	parent: "body",
	scene  : {
		preload: game.preload,
		create : game.setup,
		update : game.update,
		pack: {files: [
			{type: 'image', key: 'cover0', url: 'demo-assets/Cover 0.png'},
			{type: 'image', key: 'cover1', url: 'demo-assets/Cover 1.png'},
			{type: 'image', key: 'cover2', url: 'demo-assets/Cover 2.png'},
			{type: 'image', key: 'cover3', url: 'demo-assets/Cover 3.png'},
			{type: 'image', key: 'cover4', url: 'demo-assets/Cover 4.png'},
			{type: 'image', key: 'cover5', url: 'demo-assets/Cover 5.png'},
			{type: 'image', key: 'cover6', url: 'demo-assets/Cover 6.png'}
		]}
	},
	scale: {
		mode      : Phaser.Scale.FIT,
		autoCenter: Phaser.Scale.CENTER_BOTH,
	}
}

game.phaser = new Phaser.Game(game.config);