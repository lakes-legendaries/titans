// ====================================================================================
// Process user input (b)
// ====================================================================================

// ====================================================================================
// Interface
var button = {
	setup       : function() {}, // draw buttons
	update      : function() {}, // change which buttons are shown
	queue_update: function() {}, // update() once the move queue is empty
}

button.setup = function() {
	// Top button
	button.top = env.add.sprite(button.position.top, "action buttons");
	button.top.setInteractive();
	button.top.on('pointerdown', () => {button.click.top()});
	button.top.setFrame(button.frame.tutorial);
	
	// Bottom button
	button.bot = env.add.sprite(button.position.bot, "action buttons");
	button.bot.setInteractive();
	button.bot.on('pointerdown', () => {button.click.bot()});
	button.bot.setFrame(button.frame.blank);
}

button.update = function() {
	
	// Check if update should be performed
	if (!button.perform_update || move.in_progress() || tut.block_button) {return;}
	button.perform_update = false;
	
	// Perform update
	if (age.major() < age.battle) {
		switch (age.minor()) {
			case age.step.play:
				let allow_surge = age.major() == 0 && stats.temples(player.you) < 3 && stats.num_surges(player.you) > 0;
				button.top.setFrame(!allow_surge? button.frame.blank: button.frame.surge);
				break;
			case age.step.subvert_cave_in:
				button.top.setFrame(button.frame.dont_subvert);
				break;
			case age.step.sacrifice:
				button.top.setFrame(button.frame.dont_sacrifice);
				break;
			case age.step.purify:
				button.top.setFrame(button.frame.dont_purify);
				break;
			case age.step.buy:
				button.top.setFrame(tut.active()? button.frame.blank: button.frame.dont_buy);
				break;
			case age.step.substitute_out:
				button.top.setFrame(button.frame.dont_substitute);
				break;
			default:
				button.top.setFrame(button.frame.blank);
				break;
		}
		button.bot.setFrame(button.frame.blank);
	} else {
		button.top.setFrame(tut.active()? button.frame.blank: button.frame.discard_none);
		button.bot.setFrame(tut.active()? button.frame.blank: button.frame.discard_all);
	}
}

button.queue_update = function() {
	button.perform_update = true;
}

// ====================================================================================
// Backend
button.frame = {
	tutorial          :  0,
	surge             :  1,
	discard_selected  :  2,
	discard_all       :  3,
	dont_buy          :  4,
	dont_subvert      :  5,
	dont_sacrifice    :  6,
	dont_purify       :  7,
	dont_substitute   :  8,
	blank             :  9,
	discard_none      : 10,
	sacrifice_selected: 11,
}
button.position = {
	top: {x: 1730, y:  934},
	bot: {x: 1730, y: 1026},
}
button.perform_update = false;

button.click = {
	top: function(force = false) {
		
		// Don't allow double-clicks or update-pending clicks
		if (controller.bot_clicked || controller.top_clicked) {return;}
		if (button.perform_update) {return;}
		if (controller.game_over) {return;}
		
		// Tutorial
		if (button.top.frame.name == button.frame.tutorial) {
			button.top.setFrame(button.frame.blank);
			tut.start();
			return;
		}
		
		// Skip if in tutorial
		if (tut.active()) {return;}
		
		// Blank -- no action
		if (button.top.frame.name == button.frame.blank) {return;}
		
		// Do nothing while move is in progress
		if (!force && move.in_progress()) {return;}
		
		// Add to log
		switch (button.top.frame.name) {
			case button.frame.dont_buy:
				log.add({event: log.event.decline_buy, person: player.you});
				break;
			case button.frame.dont_subvert:
				log.add({event: log.event.decline    , person: player.you, ability: "Subvert: Cave In"});
				break;
			case button.frame.dont_sacrifice:
				log.add({event: log.event.decline    , person: player.you, ability: "Sacrifice"});
				break;
			case button.frame.dont_purify:
				log.add({event: log.event.decline    , person: player.you, ability: "Purify"});
				break;
			case button.frame.dont_substitute:
				log.add({event: log.event.decline    , person: player.you, ability: "Substitute"});
				break;
		}
		log.active = false;
		
		// Surge
		if (button.top.frame.name == button.frame.surge) {
			actions.surge(player.you);
			button.update();
			return;
		}
		
		// Sacrifice selected
		if (button.top.frame.name == button.frame.sacrifice_selected) {
			for (let d = zone.count(player.you, zone.hand)-1; d >= 0; d--) {
				let card_num = zone.get(player.you, zone.hand, d);
				if (highlight.active(card_num)) {
					actions.sacrifice(card_num);
				}
			}
			highlight.clear();
		}
		
		// Discard selected
		if (button.top.frame.name == button.frame.discard_selected) {
			for (let d = zone.count(player.you, zone.hand)-1; d >= 0; d--) {
				let card_num = zone.get(player.you, zone.hand, d);
				if (highlight.active(card_num)) {
					
					// Log discard
					log.add({event: log.event.shuffle_discard, person: player.you, count: 1, card_num: card_num});
					
					// Perform discard
					actions.discard(player.you, card_num, {simultaneous: true});
				}
			}
			highlight.clear();
		}
		
		// Process click
		controller.top_clicked = true;
		controller.advance();
	},
	
	bot: function(force = false) {
		
		// Don't allow double-clicks or update-pending clicks
		if (controller.bot_clicked || controller.top_clicked) {return;}
		if (button.perform_update) {return;}
		if (controller.game_over) {return;}
		
		// Do nothing while move is in progress
		//if (!force && move.in_progress()) {return;}
		
		// Skip if in tutorial
		if (tut.active()) {return;}
		
		// Blank -- no action
		if (button.bot.frame.name == button.frame.blank) {return;}
		
		// Process click
		controller.bot_clicked = true;
		
		// Add to log
		log.add({event: log.event.shuffle_discard, person: player.you, count: -1});
		log.active = false;
		
		// Discard all
		move.combine_zone(player.you, zone.hand, zone.disc);
		highlight.clear();
		controller.advance();
		
		// Hide help
		full_card.hide();
	},
}