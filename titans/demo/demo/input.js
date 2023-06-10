// ====================================================================================
// Process user input (card clicks)
// ====================================================================================

// ====================================================================================
// Interface
var input = {
	setup      : function() {}, // adds input controls to card sprites
	using_touch: function() {}, // true if touch input is being used
}

input.setup = function() {
	for (let card_num = 0; card_num < card.num; card_num++) {
		card[card_num].sprite.setInteractive();
		card[card_num].sprite.on('pointerdown', () => {input.select(card_num)});
	}
}

// ====================================================================================
// Backend
input.last_touch = -1;
input.select = function(card_num) {
	
	// Can't select while object are moving, or if is blocked
	if (move.in_progress() && !move.fading()) {return false;}
	if (controller.game_over) {return false;}
	
	// Process touch input
	if (touch.using() && full_card.showing[0] != card_num) {return false;}
	
	// Skip, if in tutorial and we're forcing a different input
	if (tut.forced_input != null) {
		if (tut.forced_input == card_num) {
			tut.forced_input = null;
		} else {return false;}
	}
	
	// Find card location
	let loc = zone.find(card_num);
	if (loc == null) {return false;} // for debugging -- can't choose a sacrificed card
	
	// Process via age of the game
	if (age.major() < age.battle) {
		switch (age.minor()) {
			
			// play card
			case age.step.play :
			case age.step.flash:
			case age.step.substitute_in: {
				
				// See if card can be played
				if (loc.person != player.you) {return false;}
				if (loc.place != zone.hand && zone.get_last(player.you, zone.deck) != card_num) {return false;}
				
				// Play card
				actions.play(player.you, card_num);
				
				// Log, if substitute
				if (age.minor() == age.step.substitute_in) {
					log.add({event: log.event.substitute, card_num: input.subbed_out, card_num2: card_num, hide: true})
				}
				
				// Return
				break;
			
			}
			
			// subvert card
			case age.step.subvert_harmless:
			case age.step.subvert_mindless:
			case age.step.subvert_traitorous:
			case age.step.subvert_cave_in: {
				
				// See if card can be subverted
				if (loc.person != player.opp || loc.place != zone.play) {return false;}
				if (age.minor() == age.step.subvert_cave_in && card[card_num].age_played != age.major()) {return false;}
				
				// See if card already has the given subversion
				switch (age.minor()) {
					case age.step.subvert_harmless:
						if (card[card_num].subv[subv.type.harmless] != null) {return false;}
						break;
					case age.step.subvert_mindless:
						if (card[card_num].subv[subv.type.mindless] != null) {return false;}
						break;
					case age.step.subvert_traitorous:
						if (card[card_num].subv[subv.type.traitorous] != null) {return false;}
						break;
					case age.step.subvert_cave_in:
						if (card[card_num].subv[subv.type.harmless] != null && card[card_num].subv[subv.type.mindless] != null) {return false;}
						break;
				}
				
				// Subvert card
				switch (age.minor()) {
					case age.step.subvert_harmless:
						actions.subvert(card_num, subv.type.harmless);
						break;
					case age.step.subvert_mindless:
						actions.subvert(card_num, subv.type.mindless);
						break;
					case age.step.subvert_traitorous:
						actions.subvert(card_num, subv.type.traitorous);
						break;
					case age.step.subvert_cave_in:
						controller.used_cave_in = true;
						actions.subvert(card_num, subv.type.mindless);
						actions.subvert(card_num, subv.type.harmless);
						break;
				}
				
				// Add to log
				log.add({event: subv.to_event(subv.from_age(age.minor())), person: player.you, card_num: card_num});
				
				// Return
				break;
			
			}
			
			// sacrifice card
			case age.step.sacrifice: {
				
				// See if card can be sacrificed
				if (loc.person != player.you || loc.place != zone.hand) {return false;}
				
				// Turn highlight off, if on
				if (highlight.active(card_num)) {
					highlight.clear(card_num);
				// Turn highlight on, if off and highlight limit not exceeded
				} else if (highlight.num_active() < stats.count(player.you, abilities.sacrifice)) {
					highlight.add  (card_num);
				}
				
				// Update button
				button.top.setFrame(highlight.any_active()? button.frame.sacrifice_selected: button.frame.dont_sacrifice);
				
				// Return -- we don't advance controller here
				return false;
			}
			
			// purify card
			case age.step.purify: {
				
				// See if card can be purified
				if (loc.place != zone.play || card.num_subv(card_num) == 0) {return false;}
				
				// Purify
				actions.purify(card_num, player.you);
				break;
			
			}
			
			// buy card
			case age.step.buy: {
				
				// See if card can be bought
				if (loc.person != player.none)  {return false;}
				if (loc.place  != zone.buy_top) {return false;}
				if (stats.energy(player.you) < card[card_num].cost) {return false;}
				
				// Mark zodiac bought (for help message)
				if (card[card_num].name == card.name.zodiac_the_eternal) {
					instr.jace_read = true;
				}
				
				// Buy card
				actions.buy(player.you, card_num);
				break;
			
			}
			
			// substitute card out of play
			case age.step.substitute_out: {
				
				// See if card can be substituted
				if (loc.person != player.you || loc.place != zone.play) {return false;}
				
				// Save card to add to log later
				input.subbed_out = card_num;
				
				// Substitute
				actions.discard(player.you, card_num);
				break;
			}
		}
		
		// input processed, throw back to controller
		controller.advance();
	
	// after battle - discard cards
	} else {
		
		// Skip if tutorial
		if (tut.active()) {return false;}
		
		// Check if card is in your hand
		if (loc.person != player.you || loc.place != zone.hand) {return false;} // card isn't in your hand
		
		// Toggle for discarding
		highlight.toggle(card_num);
		
		// Update button
		button.top.setFrame(highlight.any_active()? button.frame.discard_selected: button.frame.discard_none);
	}
	
	// hide full card
	full_card.hide();
	return true;
}