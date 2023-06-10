// ====================================================================================
// Show the full card for any given sprite
// ====================================================================================

// ====================================================================================
// Interface
var full_card = {
	// phaser plugins
	setup       : function() {}, // attach hover-over functions to card sprites
	update      : function() {}, // bring showing to front
	
	// turn off
	hide        : function() {}, // hide all full-card sprites
	
	// meta-data
	showing     : [-1, -1],      // last 2 cards shown, -1 is the default value
	showing_subv: null,          // subversion frame currently being shown
	active      : function() {return full_card[0].visible || full_card[1].visible;}, // true if a full card is currently being shown
	
	// trigger to set
	game_over   : false,
}

full_card.setup = function() {
	
	// Make full card sprites
	full_card[0] = env.add.sprite(full_card.pos, 'full cards A'); // 1st half of cards
	full_card[1] = env.add.sprite(full_card.pos, 'full cards B'); // 2nd half of cards
	full_card[2] = env.add.sprite(full_card.pos, 'full cards A'); // used for subversions, so first half
	full_card[3] = env.add.sprite(full_card.pos, 'full cards A'); // used for subversions, so first half
	
	// Add hover-action for cards
	for (let card_num = 0; card_num < card.num; card_num++) {
		card[card_num].sprite.on('pointerover', () => {if (!touch.using()) {full_card.show(card_num);}});
		card[card_num].sprite.on('pointerout' , () => {if (!full_card.touch) {full_card.hide();}});
		card[card_num].sprite.on('pointerdown', () => {full_card.show(card_num)});
	}
	
	// Add hover-action for subversions
	for (let s = 0; s < subv.num; s++) {
		for (let type = 0; type < subv.type.num; type++) {
			subv[s][type].on('pointerover', () => {if (!touch.using()) {full_card.show(s, type)}});
			subv[s][type].on('pointerout' , () => {if (!full_card.touch) {full_card.hide()}});
			subv[s][type].on('pointerdown', () => {full_card.show(s, type)});
		}
	}
	
	// Add help-hide for other objects
	env.background.setInteractive().on('pointerdown', () => {full_card.hide()});
	tut.button[0] .setInteractive().on('pointerdown', () => {full_card.hide()});
	tut.button[1] .setInteractive().on('pointerdown', () => {full_card.hide()});
	
	// Hide created objects
	full_card.hide();
	
	// Set game as ongoing
	full_card.game_over = false;
}

full_card.update = function() {
	for (let c = 3; c >= 0; c--) {
		if (!full_card[c].visible) {continue;}
		env.to_front(full_card[c]);
	}
}

full_card.hide = function() {
	full_card[0].visible = false;
	full_card[1].visible = false;
	full_card[2].visible = false;
	full_card[3].visible = false;
	full_card.showing = [-1, -1];
	full_card.showing_subv = null;
	ref.hide();
	log.active = false;
}

// ====================================================================================
// Backend
full_card.pos = {x: 200, y: env.window.y/2, ymindist: 300, xoffset: 350, from_edge: 300, from_instr_box: 700, sub_offset: {x: 430, y: 262, from_edge: 300}}

full_card.show_associated_subversion = function(subversion, sprite_num = 2) {
	
	// Show 2 subversions
	if (subversion == abilities.subvert_cave_in) {
		full_card.show_associated_subversion(abilities.subvert_harmless, 2);
		full_card.show_associated_subversion(abilities.subvert_mindless, 3);
		return;
	}
	
	// Show subversion
	full_card[sprite_num].x = full_card[0].x + full_card.pos.sub_offset.x;
	full_card[sprite_num].y = full_card[0].y + (sprite_num == 2? -1: 1) * full_card.pos.sub_offset.y;
	full_card[sprite_num].setFrame(subversion == 7? 7: subversion - abilities.subvert_cave_in).setScale(0.75);
	full_card[sprite_num].visible = true;
	env.to_front(full_card[sprite_num]);
}

full_card.show = function(card_num, subversion_frame) {
	
	// see if skip
	if (tut.block_full)  {return;}
	if (camera.active()) {return;}
	if (card[card_num].sprite.alpha < 1) {return;}
	if (full_card.game_over && (zone.find(card_num).place == zone.buy_top || zone.find(card_num).place == zone.buy_other)) {return;}
	
	// check for touch
	full_card.touch = touch.using();
	
	// Choose frame
	let frame_num;
	let is_subv = subversion_frame != null;
	if (!is_subv) { // showing a card
		if (card[card_num].sprite.frame.name != 0) { // faceup card
			frame_num = card[card_num].sprite_frame;
		} else if (zone.get_last(player.you, zone.deck) == card_num || zone.find(card_num).place == zone.play) {
			frame_num = 0;
		} else {return;} // not valid to show
	} else { // showing a subversion
		frame_num = 1 + subversion_frame;
	}
	
	// Hide subversions & ref card
	full_card[2].visible = false;
	full_card[3].visible = false;
	ref.hide();
	log.active = false;
	
	// Get full_card coordinates
	let half_coords = !is_subv? card[card_num].sprite: subv[card_num][subversion_frame];
	full_card[0].x = half_coords.x;
	
	// Chcek if a subversion is being shown
	let has_subv = 
		card[card_num].abilities[abilities.subvert_cave_in   ] > 0 || 
		card[card_num].abilities[abilities.subvert_harmless  ] > 0 || 
		card[card_num].abilities[abilities.subvert_mindless  ] > 0 || 
		card[card_num].abilities[abilities.subvert_traitorous] > 0 ;
	
	// Apply x-offset
	if (Math.abs(half_coords.y - full_card.pos.y) < full_card.pos.ymindist) {
		let sign  = half_coords.x < camera.get_midpt()? 1: -1;
		if (tut.active() && tut.stage.cur == tut.stage.buy && card[card_num].elem == card.elem.ice) {
			sign = 1;
		}
		full_card[0].x += sign * full_card.pos.xoffset;
		if (sign == -1 && has_subv) {
			full_card[0].x -= full_card.pos.xoffset;
		}
	}
		
	// Check bounds
	{
		// get bounds
		let x_min = camera.get_x0  () + full_card.pos.from_edge;
		let x_max = camera.get_xmax() - full_card.pos.from_edge
		if (!camera.at_left()) {
			x_min = camera.get_x0() + full_card.pos.from_instr_box;
		}
		
		// apply bounds
		if (full_card[0].x < x_min) {full_card[0].x = x_min;}
		if (full_card[0].x > x_max) {full_card[0].x = x_max;}
	}
	
	// Show corresponding subversions
	if (frame_num != 0 && !tut.block_instr) {
		for (let ability = abilities.subvert_cave_in; ability <= abilities.subvert_traitorous; ability++) {
			if (card[card_num].abilities[ability] > 0) {
				full_card.show_associated_subversion(ability);
			}
		}
		if (card[card_num].name == card.name.aurora_draco) {
			full_card.show_associated_subversion(7);
			full_card.show_associated_subversion(abilities.subvert_harmless, 3);
		}
	}
	
	// Show card
	full_card[1].x = full_card[0].x;
	full_card[1].y = full_card[0].y;
	let full_card_index = frame_num >= 12? 1: 0;
	let other_index     = full_card_index == 1? 0: 1;
	let rel_frame       = frame_num % 12;
	full_card[full_card_index].setFrame(rel_frame);
	full_card[full_card_index].visible = true;
	env.to_front(full_card[full_card_index]);
	full_card[other_index].visible = false;
	
	// Mark which full card is being shown
	full_card.showing[0] = full_card.showing[1];
	full_card.showing[1] = !is_subv? card_num: -1;
	full_card.showing_subv = subversion_frame;
}