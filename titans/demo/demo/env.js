// ====================================================================================
// Environment -- background, and access to the physics engine
// ====================================================================================

// ====================================================================================
// Interface
var env = {
	
	// access to physics engine
	physics: null,                         // set equal to `this` variable of phaser#preload
	
	// phaser plugins
	preload: function(phasers_this) {},    // preload assets
	setup  : function()             {},    // draw background
	
	// create images & sprites
	add    : {                             // create objects
		image : function(coords, name) {}, // add image  to scene
		sprite: function(coords, name) {}, // add sprite to scene
	},
	
	// basic object manipulation
	place   : function(sprite, coords) {}, // instantaneously move sprite to coords
	to_front: function(sprite)         {}, // move sprite to front
	
	// coordinates & sizes
	nowhere: {x: 2000, y: 2000},           // default, out-of-bounds location
	scene  : {x: 3440, y: 1080},           // scene size (full background)
	window : {x: 1920, y: 1080},           // window size (viewable portion)
	
	// image object for the background
	background: null,
}

env.preload = function(phasers_this) {
	
	// Save pointer to physics engine
	env.physics = phasers_this;
	
	// Load background
	env.physics.load.image('background', 'demo-assets/Background.png');
	
	// Load cards
	env.physics.load.image      ('ref sheet'   , 'demo-assets/Reference Sheet.png');
    env.physics.load.image      ('ref sprite'  , 'demo-assets/Reference Sprite.png');
	env.physics.load.spritesheet('full cards A', 'demo-assets/Full Cards A.png', {frameWidth: 500, frameHeight: 700});
	env.physics.load.spritesheet('full cards B', 'demo-assets/Full Cards B.png', {frameWidth: 500, frameHeight: 700});
	env.physics.load.spritesheet('half cards'  , 'demo-assets/Card Sprites.png'  , {frameWidth: 250, frameHeight: 200});
	
	// Load buttons
	env.physics.load.image      ('log button'        , 'demo-assets/Log Button.png');
	env.physics.load.image      ('help button'       , 'demo-assets/Help Button.png');
	env.physics.load.image      ('restart button'    , 'demo-assets/Restart Button.png');
	env.physics.load.spritesheet('action buttons'    , 'demo-assets/Action Buttons.png'      , {frameWidth: 250, frameHeight: 100});
	env.physics.load.spritesheet('camera'            , 'demo-assets/Camera Toggles.png'      , {frameWidth:  75, frameHeight: 175});
	env.physics.load.spritesheet('tutorial buttons'  , 'demo-assets/Tutorial Buttons.png'    , {frameWidth: 250, frameHeight: 100});
	env.physics.load.spritesheet('full screen button', 'demo-assets/Full Screen Button.png', {frameWidth:  75, frameHeight:  75});
	env.physics.load.spritesheet('title buttons'     , 'demo-assets/Title Buttons.png'       , {frameWidth: 600, frameHeight: 150});
	
	// Load text
	env.physics.load.spritesheet('age text', 'demo-assets/Age Text.png'      , {frameWidth: 350, frameHeight: 170});
	env.physics.load.spritesheet('delta'   , 'demo-assets/Delta.png'         , {frameWidth: 100, frameHeight:  67});
	env.physics.load.spritesheet('numbers' , 'demo-assets/Numbers.png'       , {frameWidth:  50, frameHeight:  67});
	
	// Load victory screen
	env.physics.load.spritesheet('victory' , 'demo-assets/Victory Screen.png', {frameWidth: 1100, frameHeight: 650});
	
	// Tutorial instructions
	env.physics.load.spritesheet('main instructions a'     , 'demo-assets/Main Instructions A.png'     , {frameWidth:  1100, frameHeight: 650});
	env.physics.load.spritesheet('main instructions b'     , 'demo-assets/Main Instructions B.png'     , {frameWidth:  1100, frameHeight: 650});
	env.physics.load.spritesheet('first buy instructions', 'demo-assets/First Buy Instructions.png', {frameWidth: 412.5, frameHeight: 250});
	env.physics.load.spritesheet('later buy instructions'  , 'demo-assets/Later buy instructions.png', {frameWidth:  1100, frameHeight: 650});
	env.physics.load.spritesheet('final instructions'      , 'demo-assets/Final Instructions.png'    , {frameWidth:   650, frameHeight: 300});
	env.physics.load.image      ('super-effective'         , 'demo-assets/Super Effective Chain.png');
	
	// Shaders
	env.physics.load.spritesheet('vertical shaders' , 'demo-assets/Vertical Shaders.png' , {frameWidth:  300, frameHeight: 1000});
	env.physics.load.image      ('horizontal shader', 'demo-assets/Horizontal Shader.png');
	env.physics.load.image      ('single shader', 'demo-assets/Single Shader.png');
	
	// Tutorial arrows
	env.physics.load.image('play arrow'        , 'demo-assets/Play arrow.png');
	env.physics.load.image('buy arrow'         , 'demo-assets/Buy arrow.png');
	env.physics.load.image('ref arrow'         , 'demo-assets/Ref arrow.png');
	env.physics.load.image('log arrow'         , 'demo-assets/Log arrow.png');
	env.physics.load.image('explanation arrow' , 'demo-assets/Explanation arrow.png');
	env.physics.load.image('top card arrow'    , 'demo-assets/Top card arrow.png');
	env.physics.load.image('surge arrow'       , 'demo-assets/Surge arrow.png');
	env.physics.load.spritesheet('camera arrow', 'demo-assets/Camera arrows.png', {frameWidth: 700, frameHeight: 500});
}

env.setup = function() {
	let mid_pt = {x: env.scene.x/2, y: env.scene.y/2}
	env.background = env.add.image(mid_pt, 'background');
}

env.add.image = function(coords, name) {
	return env.physics.add.image(coords.x, coords.y, name);
}

env.add.sprite = function(coords, name) {
	return env.physics.physics.add.sprite(coords.x, coords.y, name);
}

env.place = function(sprite, coords) {
	// move sprite
	if (sprite.body != null) {
		sprite.body.reset(coords.x, coords.y);
	// move image
	} else {
		sprite.x = coords.x;
		sprite.y = coords.y;
	}
}

env.to_front = function(sprite) {
	env.physics.children.bringToTop(sprite);
}