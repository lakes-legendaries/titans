// ====================================================================================
// Ref -- reference card
// ====================================================================================

// ====================================================================================
// Interface
var ref = {
	setup : function() {}, // phaser plugin -- load sheet, place button, & add mouse-over action
	show  : function() {}, // show ref sheet
	hide  : function() {}, // hide ref sheet (called whenever full_card is shown or hidden)
	active: function() {return ref.sheet.visible;}, // whether is currently active
}

ref.setup = function() {
	// make button
	ref.button = env.add.image(ref.button_pos, 'help button');
	ref.button.setInteractive();
	ref.button.on('pointerdown', () => {ref.toggle()});
    
    // make sprite
    ref.sprite = env.add.image(ref.sprite_pos, 'ref sprite');
    ref.sprite.setInteractive();
    ref.sprite.on('pointerdown', () => {ref.toggle()});
	
	// make sheet
	ref.sheet = env.add.image(ref.sheet_pos_L, 'ref sheet');
	ref.sheet.visible = false;
}

ref.show = function() {
	ref.sheet.visible = true;
	env.to_front(ref.sheet);
	env.place(ref.sheet, camera.at_left()? ref.sheet_pos_L: ref.sheet_pos_R);
}

ref.hide = function() {
	ref.sheet.visible = false;
}

// ====================================================================================
// Backend
ref.button_pos  = {x: 1582, y:  54};
ref.sheet_pos_L = {x: 1350, y: 540};
ref.sheet_pos_R = {x: 3150, y: 540};
ref.sprite_pos  = {x: 3280, y: 910};

ref.toggle = function() {
	if (ref.sheet.visible) {ref.hide();}
	else {ref.show();}
}