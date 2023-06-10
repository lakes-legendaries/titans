// ====================================================================================
// Controller: Governs game actions
// ====================================================================================

// ====================================================================================
// Interface
var controller = {
	setup          : function() {}, // Each player draws their starting hand
	update         : function() {}, // Phaser plugin - check for end of game
	advance        : function() {}, // Process next game action
	hand_size0     : 6,             // # of cards in starting hand
	top_clicked    : null,          // whether the top    button has recently been clicked
	bot_clicked    : null,          // whether the bottom button has recently been clicked
	used_cave_in   : null,          // set to `true` when user uses cave in ability
	game_over      : null,          // whether the game is over
}

controller.setup = function() {	
	controller.summon_used         = 0;
	controller.substitute_used     = 0;
	controller.substitute_limit    = 0;
	controller.opp_subversions     = [];
	for (let person = 0; person < player.num; person++) {
		actions.draw(person, controller.hand_size0);
	}
	controller.top_clicked     = false;
	controller.bot_clicked     = false;
	controller.game_over       = false;
}

controller.update = function() {
	if (!controller.game_over && stats.game_over()) {
		controller.end_of_game();
		return;
	}
}

controller.advance = function() {
	
	// Process end of current age / step
	if (age.major() < age.battle) {
		
		// Get ability stats
		let ability_stats = controller.ability_stats();
		
		// Zero-out substitute count (if clicked don't substitute)
		if (controller.top_clicked && age.minor() == age.step.substitute_out) {
			controller.substitute_limit = 0;
		}
		
		// Discard Cavern's Defender from play
		if (age.minor() == age.step.subvert_cave_in && controller.used_cave_in) {
			for (let d = zone.count(player.you, zone.play) - 1; d >= 0; d--) {
				let card_num = zone.get(player.you, zone.play, d);
				if (card[card_num].name != card.name.caverns_defender) {continue;}
				actions.discard(player.you, card_num);
				break;
			}
		}
		controller.used_cave_in = false;
		
		// Return for additional input
		if (!controller.top_clicked && ability_stats.conditions_met) {
			switch (age.minor()) {
				
				// Play cards until you reach your summon limit
				case age.step.play:
					if (controller.summon_used++ < stats.count(player.you, abilities.summon)) {return;}
					break;
				
				// Decrement `Now` ability, return if more still existing
				case age.step.subvert_harmless:
				case age.step.subvert_mindless:
				case age.step.subvert_traitorous:
				case age.step.subvert_cave_in:
				case age.step.flash:
				case age.step.sacrifice:
				case age.step.purify:
					stats.decrement(player.you, ability_stats.ability, 1);
					if (stats.count(player.you, ability_stats.ability) > 0) {return;}
					break;
				
				// Same as above decrement, but w/ backtracking the age step
				case age.step.substitute_in:
					if (++controller.substitute_used < controller.substitute_limit) {
						age.set_minor(age.step.substitute_out);
						return;
					}
					break;
			}
		}
		
		// Opponent acts
		switch (age.minor()) {
			
			// basic ability processing
			case age.step.play:
			case age.step.flash:
			case age.step.haunt:
			case age.step.sacrifice:
			case age.step.purify:
			case age.step.buy:
			case age.step.substitute_in:
				ai.process_ability(ability_stats.ability, stats.count(player.opp, ability_stats.ability));
				break;
			
			// subversion processing (opp does all subversions at once, after you do all of yours)
			case age.step.subvert_cave_in:
				for (let a = 0; a < abilities.num_subvert(); a++) {
					ai.process_ability(abilities.first_subvert() + a, controller.opp_subversions[a]);
				}
				break;
			
			// opponent does nothing
			case age.step.subvert_harmless:
			case age.step.subvert_mindless:
			case age.step.subvert_traitorous:
			case age.step.substitute_out:
				break;
		}
		
		// Zero-out `Now` & `This Age` abilities
		switch (age.minor()) {
			case age.step.subvert_harmless:
			case age.step.subvert_mindless:
			case age.step.subvert_traitorous:
			case age.step.subvert_cave_in:
			case age.step.flash:
			case age.step.haunt:
			case age.step.sacrifice:
			case age.step.purify:
			case age.step.buy:
				for (let person = 0; person < player.num; person++) {
					stats.decrement(person, ability_stats.ability, -1);
				}
				break;
		}

		// Reveal played cards
		let any_revealed = false;
		switch (age.minor()) {
			case age.step.play:
			case age.step.flash:
			case age.step.substitute_in:
				any_revealed = actions.reveal_played();
				break;
		}
		
		// Backtrack on the ability stack
		if (age.minor() == age.step.flash && any_revealed) {
			age.set_minor(age.step.play);
		}
	
	// Process end of turn
	} else {
		controller.end_of_turn();
	}
	
	// Advance age
	camera.reset();
	age   .advance();
	stats .queue_update();
	button.queue_update();
	full_card.hide();
	
	// Resolve buttons
	skip_ahead = age.minor() == age.step.substitute_in && controller.top_clicked;
	controller.top_clicked = false;
	controller.bot_clicked = false;
	
	// Log
	if (age.minor() == age.step.play) {
		log.add({event: log.event.new_age, age: age.major()});
		if (age.major() == age.battle) {
			log.add({event: log.event.battle, person: stats.winner(), capture: stats.capture()});
		}
	}
	
	// Process beginning of next age
	if (age.major() < age.battle) {
		
		// Get ability stats
		let ability_stats = controller.ability_stats();
		
		// Process start-of-the-age
		if (age.minor() == age.step.play) {
			
			// Reset counts
			controller.summon_used      = 0;
			controller.substitute_used  = 0;
			controller.substitute_limit = 0;
		
			// Activate Start-Of-Each-Age abilities
			for (let person = 0; person < player.num; person++) {
				actions.draw        (person, stats.count(   person , abilities.draw   ));
				actions.discard_rand(person, stats.count(+(!person), abilities.discard));
			}
			
			// Zero-out This-Age abilities
			for (let person = 0; person < player.num; person++) {
				stats.decrement(person, abilities.protect, -1);
			}
			
			// AI Surge
			ai.surge();
		}
		
		// Get counts
		switch (age.minor()) {
			
			// opponent's subversion count
			case age.step.subvert_harmless:
				
				// get count & decrement abilities
				for (let a = 0; a < abilities.num_subvert(); a++) {
					let ability = abilities.first_subvert() + a;
					controller.opp_subversions[a] = stats.count(player.opp, ability);
					stats.decrement(player.opp, ability, -1);
				}
				
				// mark cards subvertible / not subvertible
				for (let card_num = 0; card_num < card.num; card_num++) {
					card[card_num].ai_subvertible = zone.contains(player.you, zone.play, card_num);
				}
				
				// finished
				break;
			
			// your substitute count
			case age.step.substitute_out:
				if (age.major() == age.substitute && controller.substitute_limit == 0) {
					controller.substitute_limit = ability_stats.count;
					instr.num_sub = controller.substitute_limit;
				}
				break;
		}
		
		// Process haunt
		if (age.minor() == age.step.haunt) {
			for (let person = 0; person < player.num; person++) {
				actions.haunt(+(!person), stats.count(person, abilities.haunt));
			}
		}
		
		// Log Protecting against Subvert
		if (ability_stats.count > 0 && stats.count(player.opp, abilities.protect)) {
			switch (age.minor()) {
				case age.step.subvert_harmless:
					log.add({event: log.event.protect, person: player.opp, ability: "Subvert: Harmless"});
					break;
				case age.step.subvert_mindless:
					log.add({event: log.event.protect, person: player.opp, ability: "Subvert: Mindless"});
					break;
				case age.step.subvert_traitorous:
					log.add({event: log.event.protect, person: player.opp, ability: "Subvert: Traitorous"});
					break;
				case age.step.subvert_cave_in:
					log.add({event: log.event.protect, person: player.opp, ability: "Subvert: Cave In"});
					break;
			}
		}
		
		// Skip ahead if we're not waiting for user input
		if (!ability_stats.conditions_met || skip_ahead) {
			controller.advance();
		}
	}
	
	// Show tutorial
	if (tut.active()) {
		if (age.major() == age.battle) {
			tut.battle();
		} else if (age.minor() == age.step.play) {
			tut.play();
		} else if (age.minor() == age.step.buy) {
			tut.buy();
		}
	}
}

// ====================================================================================
// Backend
controller.victory_loc         = {x: 2675, y: 540};
controller.restart_loc         = {x: 2675, y: 850};

controller.ability_stats = function() {
	
	// Get corresponding ability & count
	let ability = abilities.from_age();
	let count   = age.minor() == age.step.buy? stats.energy(player.you): stats.count(player.you, ability);
	
	// Check if conditions are met for ability going active
	let conditions_met = false;
	switch (age.minor()) {
		
		// Any cards in your hand or top of your deck
		case age.step.play:
		case age.step.flash:
			conditions_met = zone.count(player.you, zone.deck) + zone.count(player.you, zone.hand) > 0;
			break;
		
		// Any cards in opponent's play that have not yet been subverted with the given ability
		case age.step.subvert_harmless:
		case age.step.subvert_mindless:
		case age.step.subvert_traitorous:
			if (stats.count(player.opp, abilities.protect) > 0) {conditions_met = false; break;}
			conditions_met = controller.any_opp_cards_in_play_not_subverted_with(ability);
			break;
		case age.step.subvert_cave_in:
			if (stats.count(player.opp, abilities.protect) > 0) {conditions_met = false; break;}
			conditions_met = controller.any_opp_cards_in_play_not_subverted_with(abilities.subvert_harmless)
				|| controller.any_opp_cards_in_play_not_subverted_with(abilities.subvert_mindless);
			break;
		
		// Always true
		case age.step.haunt:
			conditions_met = false; // auto-advance for haunt (we never want user input)
			break;
			
		// Any cards in your hand	
		case age.step.sacrifice:
			conditions_met = zone.count(player.you, zone.hand) > 0;
			break;
		
		// Any cards in play that are subverted
		case age.step.purify:
			conditions_met = false;
			for (let person = 0; person < player.num && !conditions_met; person++) {
				for (let d = 0; d < zone.count(person, zone.play); d++) {
					let card_num = zone.get(person, zone.play, d);
					if (card.num_subv(card_num) > 0) {
						conditions_met = true;
						break;
					}
				}
			}
			break;
		
		// Any cards buy-able
		case age.step.buy:
			conditions_met = zone.count(player.none, zone.buy_top) > 0;
			break;
		
		// Any cards in your play, and is third age
		case age.step.substitute_out:
			conditions_met = age.major() == age.substitute && zone.count(player.you, zone.play) > 0;
			break;
		
		// Any cards in your hand or top of your deck, and is third age, and you substituted out
		case age.step.substitute_in:
			conditions_met = age.major() == age.substitute && (zone.count(player.you, zone.deck) + zone.count(player.you, zone.hand) > 0) && !controller.top_clicked;
			break;
	}
	
	// Additional conditions
	switch (age.minor()) {
		
		// std -- ability must be active
		case age.step.subvert_harmless:
		case age.step.subvert_mindless:
		case age.step.subvert_traitorous:
		case age.step.subvert_cave_in:
		case age.step.flash:
		case age.step.haunt:
		case age.step.sacrifice:
		case age.step.purify:
		case age.step.buy:
		case age.step.substitute_out:
			conditions_met = conditions_met && count > 0;
			break;
		
		// substitute in -- don't go off active count: use controller variables instead
		case age.step.substitute_in:
			conditions_met = conditions_met && controller.substitute_used < controller.substitute_limit;
			break;
		
		// no additional conditions for play -- you get one free
		case age.step.play:
			break;
	}
	
	// Return stats
	return {ability: ability, count: count, conditions_met: conditions_met};
}

controller.any_opp_cards_in_play_not_subverted_with = function(ability) {
	for (let card_num of zone.get(player.opp, zone.play)) {
		let type = subv.from_ability(ability);
		if (card[card_num].subv[type] == null) {
			return true;
		}
	}
	return false;
}

controller.end_of_turn = function() {
	
	// Return traitorous cards
	for (let person = 0; person < player.num; person++) {
		for (let d = 0; d < zone.count(person, zone.play); d++) {
			let card_num      = zone.get(person, zone.play, d);
			let subv_card_num = card[card_num].subv[subv.type.traitorous];
			if (subv_card_num == null) {continue;}
			actions.purify(card_num, null);
			log.add({event: log.event.shuffle_traitorous, person: +(!person), card_num: card_num});
		}
	}
	
	// Clear out subversions & abilities
	subv.reset();
	card.discard();
	
	// Opponent discards
	ai.end_of_turn();
	
	// Doubly make sure all subversions are gone
	for (let s = 0; s < subv.num; s++) {
		for (let type = 0; type < subv.type.num; type++) {
			subv[s][type].visible = false;
		}
	}
	
	// All cards -> deck
	move.combine_zone(null, zone.play, zone.disc);
	move.combine_zone(null, zone.disc, zone.deck);
	zone.shuffle();
	move.organize(null, zone.deck, {bring_to_front: true, simultaneous: false});
	
	// Organize hand, draw cards
	move.organize(null, zone.hand);
	for (let person = 0; person < player.num; person++) {
		actions.draw(person, controller.hand_size0 - zone.count(person, zone.hand));
	}
}

controller.end_of_game = function() {
	
	// reset camera
	camera.to_right();
	
	// mark game over
	controller.game_over = true;
	instr     .game_over = true;
	full_card .game_over = true;
	
	// show victory screen
	controller.victory_screen = env.add.sprite(controller.victory_loc, 'victory');
	controller.victory_screen.setFrame(stats.temples(player.you) == 0? 1: 0);
	
	// show restart button
	controller.restart_button = env.add.image(controller.restart_loc, 'restart button');
	controller.restart_button.setInteractive();
	controller.restart_button.on('pointerdown', () => {game.setup();});
}