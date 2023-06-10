// ====================================================================================
// Tutorial
// ====================================================================================

// ====================================================================================
// Interface
var tut = {
	
	// phaser plugins
	setup : function() {}, // phaser plugin; set active to false
	update: function() {}, // phaser plugin -- make sure bought cards don't come in front of instructions
	
	// begin tutorial (button function)
	start : function() {}, // start tutorial
	
	// enum for stage of tutorial
	stage : {
		cur   : 0,
		off   : 0,
		main  : 1,
		play  : 2,
		buy   : 3,
		battle: 4,
	},
	
	// meta-data
	active      : function() {return tut.stage.cur != tut.stage.off}, // return whether tutorial is on-going
	
	// instructions for the other classes
	block_instr : false, // whether instr can show card explanations
	block_full  : false, // whether full_card can be shown
	block_button: false, // whether to block button update
	forced_input:  null, // if not null, input will only respond to this card being chosen
}

tut.setup = function() {
	
	// turn tutorial off
	tut.stage.cur    = tut.stage.off;
	tut.block_instr  = false;
	tut.block_full   = false;
	tut.block_button = false;
	tut.forced_input = null;
	
	// create buttons  (to be moved around later)
	for (let b = 0; b < 2; b++) {
		tut.button[b] = env.add.sprite(env.nowhere, 'tutorial buttons').setInteractive();
		tut.button[b].on('pointerdown', () => {tut.button.click(b)});
	}
}

tut.update = function() {
	if (tut.active() && (tut.main.visible || tut.main2.visible)) {
		env.to_front(tut.main.visible? tut.main: tut.main2);
		env.to_front(tut.button[0]);
		env.to_front(tut.button[1]);
	}
	if (tut.active() && age.major() == 2 && age.minor() == age.step.buy && !camera.at_left() && tut.later_buys.visible) {
		tut.super_effective.setVisible(true);
	}
}

tut.start = function() {
	
	// set tutorial as active
	tut.stage.cur = tut.stage.main;
	
	// make main instruction
	tut.main  = env.add.sprite(tut.pos.main, 'main instructions a');
	tut.main2 = env.add.sprite(tut.pos.main, 'main instructions b');
	tut.main2.visible = false;
	
	// setup buttons for main instructions
	tut.button[0].setFrame(tut.button.frame.blank);
	tut.button[1].setFrame(tut.button.frame.next );
	for (let b = 0; b < 2; b++) {
		env.place   (tut.button[b], tut.pos.main.button[b]);
		env.to_front(tut.button[b]);
	}
	
	// create arrows
	tut.arrow[tut.arrow.frame.play       ] = env.add.image(tut.pos.arrow[tut.arrow.frame.play       ], 'play arrow');
	tut.arrow[tut.arrow.frame.camera     ] = env.add.image(tut.pos.arrow[tut.arrow.frame.camera     ], 'camera arrow');
	tut.arrow[tut.arrow.frame.buy        ] = env.add.image(tut.pos.arrow[tut.arrow.frame.buy        ], 'buy arrow');
	tut.arrow[tut.arrow.frame.ref        ] = env.add.image(tut.pos.arrow[tut.arrow.frame.ref        ], 'ref arrow');
	tut.arrow[tut.arrow.frame.log        ] = env.add.image(tut.pos.arrow[tut.arrow.frame.log        ], 'log arrow');
	tut.arrow[tut.arrow.frame.explanation] = env.add.image(tut.pos.arrow[tut.arrow.frame.explanation], 'explanation arrow');
	tut.arrow[tut.arrow.frame.top_card   ] = env.add.image(tut.pos.arrow[tut.arrow.frame.top_card   ], 'top card arrow');
	tut.arrow[tut.arrow.frame.full_screen] = env.add.image(tut.pos.arrow[tut.arrow.frame.full_screen], 'full screen arrow');
	tut.arrow[tut.arrow.frame.surge      ] = env.add.image(tut.pos.arrow[tut.arrow.frame.surge      ], 'surge arrow');
	for (let a = 0; a < tut.arrow.num; a++) {
		tut.arrow[a].visible = false;
	}
	
	// create additional instructions
	tut.first_buy       = env.add.sprite(tut.pos.first_buy      , 'first buy instructions').setVisible(false);
	tut.later_buys      = env.add.sprite(tut.pos.later_buys     , 'later buy instructions').setVisible(false);
	tut.final_instr     = env.add.sprite(tut.pos.final_instr    , 'final instructions'    ).setVisible(false);
	tut.super_effective = env.add.sprite(tut.pos.super_effective, 'super-effective'       ).setVisible(false);
	
	// create shaders
	tut.v_shade = env.add.sprite(env.nowhere, 'vertical shaders' );
	tut.h_shade = env.add.image (env.nowhere, 'horizontal shader');
	
	// block card explanations & full cards from being shown
	tut.block_instr = true;
	tut.block_full  = true;
	
	// show full screen arrow
	tut.fs_help = false;
	// if (!maximize.scale.isFullscreen) {
		// tut.fs_help = true;
		// tut.arrow[tut.arrow.frame.full_screen].visible = true;
		// tut.fs_dismiss = env.add.sprite(tut.pos.fs_dismiss, 'tutorial buttons').setInteractive().setFrame(tut.button.frame.no_thanks).on('pointerdown', () => {
			// tut.arrow[tut.arrow.frame.full_screen].destroy();
			// tut.fs_dismiss.destroy();
			// tut.fs_help = false;
		// });
		// maximize.button.on('pointerdown', () => {
			// tut.arrow[tut.arrow.frame.full_screen].destroy();
			// tut.fs_dismiss.destroy();
			// tut.fs_help = false;
		// });
	// } else {tut.fs_help = false;}
	
	// disable user input
	tut.forced_input = -1; // an unreachable value
    
    // turn off secondary reference sheet (sprite)
    ref.sprite.visible = false;
}

// ====================================================================================
// Backend - positions & enums

tut.button = {
	frame: {
		prev     : 0,
		next     : 1,
		play     : 2,
		restart  : 3,
		got_it   : 4,
		blank    : 5,
		no_thanks: 6,
	}
}
tut.arrow = {
	frame: {
		play       : 0,
		camera     : 1,
		buy        : 2,
		ref        : 3,
		log        : 4,
		explanation: 5,
		top_card   : 6,
		full_screen: 7,
		surge      : 8,
	},
	num: 9,
}
tut.pos = {
	main: {
		x: 815,
		y: 540,
		button: [
			{x:  615, y: 855},
			{x: 1015, y: 855},
		],
	},
	play_offset    : {x:    0, y: -150},
	buy_offset     : {x: -300, y:    0},
	first_buy      : {x: 3125, y:  540, button: {x: 3260, y: 680}},
	later_buys     : {x: 2865, y:  390, button: {x: 3250, y: 737}},
	super_effective: {x: 1950, y: 690},
	shader: {
		storm      : 2075,
		fire     : 2375,
		ice      : 2675,
		rock     : 2975,
		non_elem : 3275,
		warrior  :  165,
		beast    :  415,
		dragon   :  665,
		titan    :  915,
		default_x: 2524,
		default_y:  540,
	},
	battle: {
		x: 1350,
		y: 540,
		button: {x: 1730, y: 875},
	},
	final_instr: {
		x: 365,
		y: 540,
		button: {x:  525, y: 700},
	},
	end: {x: 675, y: 685},
	arrow: [
		env.nowhere,
		{x: 1675, y: 810},
		env.nowhere,
		{x: 1350, y: 235},
		{x: 1582, y: 400},
		{x: 1415, y: 550},
		{x: 1250, y: 580},
		{x: 1582, y: 800},
		{x: 1300, y: 832},
	],
	fs_dismiss: {x: 1750, y: 740},
}
tut.main_npanels   =  6;
tut.main2_npanels  =  6;
tut.main_stop      =  7;
tut.first_buy_stop = 14;
tut.main2_on       = false;
tut.main_frame     = function() {return tut.main.frame.name + tut.main2.frame.name + (tut.main2_on? 1: 0);}

// ====================================================================================
// Backend - basic functions & controllers

tut.show_play_arrow = function() {
	
	// decide on card that user is to play
	let play_choice = zone.get_last(player.you, zone.deck);
	for (let d = 0; d < zone.count(player.you, zone.hand); d++) {
		let card_num = zone.get(player.you, zone.hand, d);
		if (card[card_num].name == card.name.monk) {
			play_choice = card_num;
			break;
		}
	}
	
	// get coords for that card
	let coords = math.add_coords(card[play_choice].sprite, tut.pos.play_offset);
	
	// show play arrow
	tut.arrow[tut.arrow.frame.play].visible = true;
	env.place(tut.arrow[tut.arrow.frame.play], coords);
	env.to_front(tut.arrow[tut.arrow.frame.play]);
	
	// force this input
	tut.forced_input = play_choice;
}

tut.play = function() {
	
	// set stage
	tut.stage.cur = tut.stage.play;
	
	// show play arrow
	if (age.major() == 0) {
		tut.show_play_arrow();
	}
	
	// hide other arrows
	tut.arrow[tut.arrow.frame.camera].visible = false;
	tut.arrow[tut.arrow.frame.buy   ].visible = false;
	
	// hide buy instructions
	tut.first_buy .setVisible(false);
	tut.later_buys.setVisible(false);
	
	// show play instructions
	switch (age.major()) {
		case 1:
			tut.main2.setFrame(tut.main_stop - tut.main_npanels + 1).setVisible(true);
			tut.button[0].setFrame(tut.button.frame.got_it).setVisible(true);
			env.to_front(tut.main2);
			env.to_front(tut.button[0]);
			env.place(tut.button[0], tut.pos.main.button[1]);
			break;
		case 2:
			tut.main2.setFrame(tut.main_stop - tut.main_npanels + 2).setVisible(true);
			tut.button[0].setFrame(tut.button.frame.got_it).setVisible(true);
			env.to_front(tut.main2);
			env.to_front(tut.button[0]);
			env.place(tut.button[0], tut.pos.main.button[1]);
			break;
	}
	
	// block full if instructions are up
	tut.block_full = age.major() != 0;
	
	// get rid of full screen help
	if (tut.fs_help) {
		tut.arrow[tut.arrow.frame.full_screen].destroy();
		tut.fs_dismiss.destroy();
		tut.fs_help = false;
	}
}

tut.buy = function() {
	
	// set stage
	tut.stage.cur = tut.stage.buy;
	
	// hide play arrow
	tut.arrow[tut.arrow.frame.play].visible = false;
	
	// show camera arrow
	tut.arrow[tut.arrow.frame.camera].setVisible(true).setFrame(age.major());
	env.to_front(tut.arrow[tut.arrow.frame.camera]);
	camera.button.on('pointerdown', () => {tut.arrow[tut.arrow.frame.camera].visible = false});
	
	// process via age
	switch (age.major()) {
		case 0:
			tut.first_buy.setVisible(true);
			env.place(tut.button[0], tut.pos.first_buy.button);
			break;
		case 2:
			tut.later_buys.setFrame(1);
			//tut.super_effective.setVisible(true);
			// fall through
		case 1:
			tut.later_buys.setVisible(true);
			env.place(tut.button[0], tut.pos.later_buys.button);
			break;
	}
	
	// show button
	tut.button[0].setVisible(true).setFrame(tut.button.frame.got_it);
	env.to_front(tut.button[0]);
	
	// block full
	tut.block_full = true;
}

tut.battle = function() {
	
	// set stage
	tut.stage.cur = tut.stage.battle;
	
	// hide play / camera / buy arrows
	tut.arrow[tut.arrow.frame.play  ].visible = false;
	tut.arrow[tut.arrow.frame.camera].visible = false;
	tut.arrow[tut.arrow.frame.buy   ].visible = false;
	
	// show battle instructions
	tut.main2.setVisible(true).setFrame(tut.main_stop - tut.main_npanels + 3);
	env.to_front(tut.main2);
	env.place(tut.main2, tut.pos.battle);
	
	// show battle button
	env.place(tut.button[0], tut.pos.battle.button);
	env.to_front(tut.button[0]);
	tut.button[0].setFrame(tut.button.frame.got_it).setVisible(true);
	
	// block button updating
	tut.block_button = true;
	
	// block full card
	tut.block_full   = true;
}

// ====================================================================================
// Backend - process button clicks

tut.button.click = function(b) {
	switch (tut.stage.cur) {
		
		// ============================================================================
		// main stage (overview instructions) -- toggle main instructions
		case tut.stage.main: {
			let cur_frame = tut.main.frame.name + tut.main2.frame.name + (tut.main2.visible? 1: 0);
			// previous button clicked
			if (b == 0) {
				if (cur_frame > 0) {
					if (cur_frame == tut.main_npanels) {
						tut.main .visible =  true;
						tut.main2.visible = false;
						tut.main .setFrame(tut.main_npanels - 1);
					} else if (cur_frame > tut.main_npanels) {
						tut.main2.setFrame(cur_frame-1 - tut.main.n_panels);
					} else {
						tut.main .setFrame(cur_frame-1);
					}
					tut.button[1].setFrame(tut.button.frame.next);
					if (cur_frame-1 == 0) {
						tut.button[0].setFrame(tut.button.frame.blank);
					}
				}
			// next button clicked
			} else {
				if (cur_frame == tut.main_stop) {
					
					// clear out main instructions
					tut.main2    .visible = false;
					tut.button[0].visible = false;
					tut.button[1].visible = false;
					
					// get play instructions
					tut.play();
					
				} else {
					if (cur_frame + 1 == tut.main_npanels) {
						tut.main .visible = false;
						tut.main2.visible =  true;
						tut.main2.setFrame(0);
					} else if (cur_frame + 1 > tut.main_npanels) {
						tut.main2.setFrame(cur_frame+1 - tut.main_npanels);
					} else {
						tut.main .setFrame(cur_frame+1);
					}
					tut.button[0].setFrame(tut.button.frame.prev);
					if (cur_frame+1 == tut.main_stop) {
						tut.button[1].setFrame(tut.button.frame.play);
					}
				}
			}
			break;
		}
		
		// ============================================================================
		// play: clear instructions, show play arrow
		case tut.stage.play:
			tut.main .visible = false;
			tut.main2.visible = false;
			tut.button[0].visible = false;
			tut.show_play_arrow();
			tut.block_full = false;
			break;
		
		// ============================================================================
		// buy: advance / clear instructions
		case tut.stage.buy:
			switch (age.major()) {
				case 0:
					
					// advance instructions
					let cur_frame = tut.first_buy.frame.name;
					if (cur_frame < tut.first_buy_stop) {
						tut.first_buy.setFrame(cur_frame + 1);
					} else {
						// hide instructions
						tut.first_buy.setVisible(false);
						tut.button[0].setVisible(false);
						tut.block_full = false;
						
						// show buy arrow
						let card_num = 59;
						let coords   = math.add_coords(card[card_num].sprite, tut.pos.buy_offset);
						tut.arrow[tut.arrow.frame.buy].visible = true;
						env.place(tut.arrow[tut.arrow.frame.buy], coords);
						env.to_front(tut.arrow[tut.arrow.frame.buy]);
						tut.forced_input = card_num;
					}
					
					// hide shaders
					tut.v_shade.visible = false;
					tut.h_shade.visible = false;
					
					// show shaders
					switch (tut.first_buy.frame.name) {
						case 2:
							env.place(tut.v_shade, {x: tut.pos.shader.storm, y: tut.pos.shader.default_y});
							tut.v_shade.setFrame(0).setVisible(true);
							break;
						case 3:
							env.place(tut.v_shade, {x: tut.pos.shader.fire, y: tut.pos.shader.default_y});
							tut.v_shade.setFrame(1).setVisible(true);
							break;
						case 4:
							env.place(tut.v_shade, {x: tut.pos.shader.ice, y: tut.pos.shader.default_y});
							tut.v_shade.setFrame(2).setVisible(true);
							break;
						case 5:
							env.place(tut.v_shade, {x: tut.pos.shader.rock, y: tut.pos.shader.default_y});
							tut.v_shade.setFrame(3).setVisible(true);
							break;
						case 7:
							env.place(tut.h_shade, {x: tut.pos.shader.default_x, y: tut.pos.shader.warrior});
							tut.h_shade.setVisible(true);
							break;
						case 8:
							env.place(tut.h_shade, {x: tut.pos.shader.default_x, y: tut.pos.shader.beast});
							tut.h_shade.setVisible(true);
							break;
						case 9:
							env.place(tut.h_shade, {x: tut.pos.shader.default_x, y: tut.pos.shader.dragon});
							tut.h_shade.setVisible(true);
							break;
						case 10:
							env.place(tut.h_shade, {x: tut.pos.shader.default_x, y: tut.pos.shader.titan});
							tut.h_shade.setVisible(true);
							break;
						case 11:
							env.place(tut.v_shade, {x: tut.pos.shader.non_elem, y: tut.pos.shader.default_y});
							tut.v_shade.setFrame(4).setVisible(true);
							break;
					}
					
					// bring instructions & button to front
					env.to_front(tut.first_buy);
					env.to_front(tut.button[0]);
					
					break;
					
				case 1: {
					
					// hide instructions
					tut.later_buys.visible = false;
					tut.button[0] .visible = false;
					tut.block_full = false;
					
					// show buy arrow
					let card_num = 71;
					let coords   = math.add_coords(card[card_num].sprite, tut.pos.buy_offset);
					tut.arrow[tut.arrow.frame.buy].visible = true;
					env.place(tut.arrow[tut.arrow.frame.buy], coords);
					env.to_front(tut.arrow[tut.arrow.frame.buy]);
					tut.forced_input = card_num;
					
					break;
				}
					
				case 2: {
					
					// hide instructions
					tut.later_buys.visible = false;
					tut.button[0] .visible = false;
					tut.super_effective.visible = false;
					tut.block_full = false;
					
					// show buy arrow
					let card_num = 83;
					let coords   = math.add_coords(card[card_num].sprite, tut.pos.buy_offset);
					tut.arrow[tut.arrow.frame.buy].visible = true;
					env.place(tut.arrow[tut.arrow.frame.buy], coords);
					env.to_front(tut.arrow[tut.arrow.frame.buy]);
					tut.forced_input = card_num;
					
					break;
				}
			}
			break;
		
		// ============================================================================
		// battle instructions
		case tut.stage.battle: {
			
			// ========================================================================
			// battle overview
			if (tut.main2.visible) {
				
				// get frame
				let cur_frame = tut.main2.frame.name;
				
				// show next frame
				if (cur_frame == tut.main2_npanels - 2) {
					
					tut.main2.setFrame(cur_frame + 1);
					
					// allow button updating
					tut.block_button = false;
					
				// show final instructions
				} else {
					
					// hid main instructions
					tut.main2.setVisible(false);
					
					// allow full card
					tut.block_full   = false;
					
					// show final instructions
					tut.final_instr.setVisible(true);
					env.to_front(tut.final_instr);
					
					// move button
					env.place(tut.button[0], tut.pos.final_instr.button);
					env.to_front(tut.button[0]);
				}
				
			// ========================================================================
			// post-battle final instructions
			} else {
				
				// get frame
				let cur_frame = tut.final_instr.frame.name;
				
				// reset the game?
				if (cur_frame == 5) {
					game.setup();
					return;
				}
				
				// show next frame
				tut.final_instr.setFrame(cur_frame + 1);
				
				// show restart button
				if (cur_frame == 4) {
					tut.button[0].setFrame(tut.button.frame.restart);
				}
				
				// show arrows
				switch (cur_frame) {
					case 0: {
						let sprite = tut.arrow[tut.arrow.frame.top_card];
						sprite.setVisible(true);
						env.to_front(sprite);
						break;
					}
					case 1: {
						tut.arrow[tut.arrow.frame.top_card].setVisible(false);
						{
							let sprite = tut.arrow[tut.arrow.frame.ref];
							sprite.setVisible(true);
							env.to_front(sprite);
						}
						{
							let sprite = tut.arrow[tut.arrow.frame.log];
							sprite.setVisible(true);
							env.to_front(sprite);
						}
						break;
					}
					case 2: {
						tut.arrow[tut.arrow.frame.ref].setVisible(false);
						tut.arrow[tut.arrow.frame.log].setVisible(false);
						let sprite = tut.arrow[tut.arrow.frame.explanation];
						sprite.setVisible(true);
						env.to_front(sprite);
						tut.block_instr = false;
						break;
					}
					case 3: {
						tut.arrow[tut.arrow.frame.explanation].setVisible(false);
						let sprite = tut.arrow[tut.arrow.frame.surge];
						sprite.setVisible(true);
						env.to_front(sprite);
						break;
					}
					case 4: {
						tut.arrow[tut.arrow.frame.surge].setVisible(false);
						break;
					}
				}
			}
			break;
		}
	}
}