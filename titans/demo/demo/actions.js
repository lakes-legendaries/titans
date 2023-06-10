// ====================================================================================
// Basic player actions
// ====================================================================================

// ====================================================================================
// Interface
var actions = {
	
	// methods
	buy          : function(person, card_num )        {}, // person buys card_num
	draw         : function(person, num_cards)        {}, // person draws num_cards from their deck
	discard      : function(person, card_num , props) {}, // person discards card_num
	discard_rand : function(person, num_cards)        {}, // person discards a random num_cards from their hand
	haunt        : function(person, num_times)        {}, // person gets haunted num_times
	play         : function(person, card_num )        {}, // person plays card_num
	purify       : function(card_num)                 {}, // remove all subversions from card_num
	reveal_played: function()                         {}, // flip played cards faceup, add abilities
	sacrifice    : function(card_num)                 {}, // sacrifice card_num from the game
	subvert      : function(card_num, type)           {}, // subvert card with type of subversion
	surge        : function(person)                   {}, // use surge
	
	// trackers
	just_played  : Array(card.num).fill(false)   , // [card_num] == true if just played (i.e. before abilities activate)
}

actions.buy = function(person, card_num) {
	
	// Buy card
	move.card(person, zone.disc, card_num);
	
	// Move the next card to the top of the buy pile
	if (card[card_num].name == card[card_num-1].name) {
		zone.change(player.none, zone.buy_top, card_num-1);
	}
	
	// Organize
	move.organize(person, zone.disc);
	
	// Log
	log.add({event: log.event.buy, person: person, card_num: card_num});
}

actions.draw = function(person, num_cards) {
	if (num_cards <= 0) {return;}
	move.organize(person, zone.hand);
	let c;
	for (c = 0; c < num_cards && zone.count(person, zone.deck) > 0; c++) {
		let card_num = zone.get_last(person, zone.deck);
		move.card(person, zone.hand, card_num);
	}
	move.organize(person, zone.hand);
	
	// Log
	if (age.major() > 0 && c > 0) {
		if (age.major() == age.battle) {
			log.add({event: log.event.shuffle_draw, person: person, count: c});
		} else {
			log.add({event: log.event.draw        , person: person, count: c});
		}
	}
}

actions.discard = function(person, card_num, props) {
	
	// check if is traitorous
	let is_traitorous = card[card_num].subv[subv.type.traitorous] != null;
	
	// reset abilities & remove subversions
	card.discard(card_num);
	
	// move card
	move.card(is_traitorous? +(!person): person, zone.disc, card_num, props);
}

actions.discard_rand = function(person, num_cards) {
	if (num_cards <= 0) {return;}
	let c;
	for (c = 0; c < num_cards && zone.count(person, zone.hand) > 0; c++) {
		let card_num = zone.get_rand(person, zone.hand);
		move.card(person, zone.disc, card_num);
	}
	
	// Log
	if (c > 0) {
		log.add({event: log.event.discard, person: +(!person), count: c});
	}
}

actions.haunt = function(person, num_times) {
	
	// Skip if not haunting
	if (num_times == 0) {return;}
	
	// Skip if opponent is protected
	if (stats.count(person, abilities.protect) > 0) {
		log.add({event: log.event.protect, person: person, ability: "Haunt"});
		return;
	}
	
	// Find ghost on top of pile
	let card_num;
	for (card_num = card.ordinal.ghost + card.count(card.name.ghost) - 1; card_num >= card.ordinal.ghost && !zone.contains(player.none, zone.buy_top, card_num); card_num--) {}
	
	// Haunt
	let num_haunted;
	for (num_haunted = 0; num_haunted < num_times && card_num >= card.ordinal.ghost; card_num--, num_haunted++) {
		
		// Haunt
		actions.subvert(card_num, subv.type.harmless, true); // attach subversion to ghost
		move.card(person, zone.play, card_num, {flip_facedown: false, simultaneous: true});
		
		// Move next ghost to top of pile
		if (card_num-1 >= card.ordinal.ghost) {
			zone.change(player.none, zone.buy_top, card_num-1);
		}
	}
	
	// Log
	log.add({event: log.event.haunt, person: +(!person), count: num_haunted});
	
	// Organize
	move.organize(person, zone.play);
}

actions.play = function(person, card_num) {
	move.card(person, zone.play, card_num);
	move.organize(person, zone.play);
	actions.just_played[card_num] = true;
	full_card.hide();
	card[card_num].age_played = age.major();
}

actions.purify = function(card_num, person) {
	
	// Do each subversion
	for (let type = 0; type < subv.type.num; type++) {
		
		// Remove subversion
		let subv_card_num = card[card_num].subv[type];
		if (subv_card_num == null) {continue;}
		move.fade_out(subv[subv_card_num][type]);
		card[card_num].subv[type] = null;
		
		// Return traitorous card
		if (type == subv.type.traitorous) {
			let loc = zone.find(card_num);
			zone.change  (+(!loc.person), zone.play, card_num);
			move.organize(+(!loc.person), zone.play);
		}
	}
	
	// Log
	if (person != null) {
		log.add({event: log.event.purify, person: person, card_num: card_num});
	}
}

actions.reveal_played = function() {
	
	// keep track of if any are flipped faceup
	let any_revealed = false;
	
	// Flip cards faceup
	move.add_pause(10);
	for (let person = 0; person < player.num; person++) {
		let num_played = 0;
		for (let card_num of zone.get(person, zone.play)) {
			if (actions.just_played[card_num]) {
				move.card(person, zone.play, card_num, {flip_faceup: true, flip_facedown: false, skip_move: true, simultaneous: true});
				card.play(card_num);
				actions.just_played[card_num] = false;
				any_revealed = true;
				num_played++;
				
				// Log
				log.unhide();
				if (age.minor() == age.step.play && num_played == 1) { // log normal play
					log.add({event: log.event.play, person: person, card_num: card_num});
				} else if (age.minor() != age.step.substitute_in) { // log flash & summon
					log.add({event: age.minor() == age.step.play? log.event.summon: log.event.flash, person: person, card_num: card_num});
				}
			}
		}
	}
	move.add_pause(1);
	
	// Update powers
	stats.update();
	
	// return if any were flipped faceup
	return any_revealed;
}

actions.sacrifice = function(card_num) {
	// Add to log
	log.add({event: log.event.sacrifice, person: zone.find(card_num).person, card_num: card_num});
	
	// Process
	move.fade_out(card_num);
	zone.remove(card_num);
}

actions.subvert = function(card_num, type, skip_fade) {
	
	// Skip if card is already subverted w/ type
	if (card[card_num].subv[type] != null) {return;}
	
	// Get subversion card
	subv_card_num = subv.next(type);
	
	// Mark subverted
	card[card_num].subv[type] = subv_card_num;
	
	// Get properties for move queue
	let sprite = subv[subv_card_num][type];
	let count  = card.num_subv(card_num);
	let offset = subv.offset(count - 1, count);
	let dest   = math.add_coords(card[card_num].sprite, offset);
	
	// Place sprite
	sprite.visible = true;
	sprite.alpha   = 1;
	env.place(sprite, dest);
	
	// Fade in
	let person = zone.find(card_num).person;
	if (skip_fade == null || !skip_fade) {
		sprite.alpha = 0;
		move.organize(person, zone.play);
		move.fade_in (sprite, card_num, offset);
		move.organize(person, zone.play);
	}
	
	// Apply traitorous
	if (type == subv.type.traitorous) {
		let loc = zone.find(card_num);
		zone.change(+(!loc.person), zone.play, card_num);
		move.organize(+(!person), zone.play);
	}
}

actions.surge = function(person) {
	
	// perform surge
	move   .combine_zone(person, zone.hand, zone.disc);
	actions.draw(person, 6);
	stats  .use_surge(person);
	
	// let player do more input
	if (person == player.you) {
		age.set_minor(-age.step.incr);
		controller.advance();
	}
	
	// Add to log
	log.add({event: log.event.surge, person: person});
}