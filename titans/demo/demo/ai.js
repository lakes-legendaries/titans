// ====================================================================================
// AI -- Opponent's choices
// ====================================================================================

// ====================================================================================
// Interface
var ai = {
	process_ability: function(ability, num_cards) {}, // ai processes ability & makes decisions
	end_of_turn    : function()                   {}, // ai prepares for next turn
	surge          : function()                   {}, // ai decides whether to surge
}

ai.process_ability = function(ability, num_cards) {
	switch (ability) {
		case abilities.summon:
			ai.play(1 + num_cards);
			break;
		case abilities.energy_evanesce:
			ai.buy();
			break;
		case abilities.flash:
			ai.play(num_cards);
			break;
		case abilities.purify:
			ai.purify(num_cards);
			break;
		case abilities.sacrifice:
			ai.sacrifice(num_cards);
			break;
		case abilities.substitute:
			ai.substitute(num_cards);
			break;
		case abilities.subvert_cave_in:
		case abilities.subvert_harmless:
		case abilities.subvert_mindless:
		case abilities.subvert_traitorous:
			ai.subvert(num_cards, subv.from_ability(ability));
			break;
	}
}

ai.end_of_turn = function() {
	ai.discard();
}

ai.surge = function() {
	if (age.major() != 0)                  {return;} // can only surge @ beginning of turn
	if (stats.temples   (player.opp) >= 2) {return;} // don't surge if early game
	if (stats.num_surges(player.opp) == 0) {return;} // can't surge if out of surges
	if (ai.num_elemental_in_hand()   >= 3) {return;} // don't surge if hand is interesting
	actions.surge(player.opp);                       // perform surge
	ai.surge();                                      // surge again?
}

// ====================================================================================
// Backend - All Actions

ai.play       = function(num_cards)       {} // play & buy section
ai.buy        = function()                {} // play & buy section
ai.discard    = function()                {} // basic action section
ai.purify     = function(num_cards)       {} // basic action section
ai.sacrifice  = function(num_cards)       {} // basic action section
ai.substitute = function(num_cards)       {} // basic action section
ai.subvert    = function(num_cards, type) {} // subvert section

// ====================================================================================
// Backend - Play & Buy

// ---------------------
// play helper functions
ai.num_in = function(person, place, name) {
	let count = 0;
	for (let card_num of zone.get(person, place)) {
		if (card[card_num].name == name && !actions.just_played[card_num]) {
			count++;
		}
	}
	return count;
}

ai.play_from_hand = function(name) {
	for (let card_num of zone.get(player.opp, zone.hand)) {
		if (card[card_num].name == name) {
			actions.play(player.opp, card_num);
			return true;
		}
	}
	return false;
}

ai.play_random = function() {
	if (zone.count(player.opp, zone.deck) > 0) {
		actions.play(player.opp, zone.get_last(player.opp, zone.deck));
	} else {
		actions.play(player.opp, zone.get_rand(player.opp, zone.hand));
	}
}

ai.num_elemental_in_hand = function() {
	let num_elemental = 0;
	for (let card_num of zone.get(player.opp, zone.hand)) {
		let is_forest = card[card_num].elem == card.elem.forest;
		let is_desert = card[card_num].elem == card.elem.desert;
		if (!is_forest && !is_desert) {
			num_elemental++;
		}
	}
	return num_elemental;
}

// ---------------------
// play functions
ai.play = function(num_cards) {
	
	// =====================================================================================================================
	// Play num_cards cards
	if (num_cards != null) {
		let num_played;
		for (num_played = 0; num_played < num_cards; num_played++) {
			let any_hand = zone.count(player.opp, zone.hand) > 0;
			let any_deck = zone.count(player.opp, zone.deck) > 0;
			if (!any_hand && !any_deck) {break;}
			ai.play();
		}
		if (num_played > 0) {move.organize(player.opp, zone.play);}
		return;
	}
	
	// =====================================================================================================================
	// First turn
	if (age.first_turn()) {
		let play_wizard = ai.num_in(player.you, zone.play, card.name.wizard) > ai.num_in(player.opp, zone.play, card.name.wizard);
		if (tut.active() && age.major() == 2) {play_wizard = true;}
		if (play_wizard) {
			if (ai.play_from_hand(card.name.wizard)) {return;}
		} else {
			if (ai.play_from_hand(card.name.monk  )) {return;}
		}
		ai.play_random();
		return;
	}
	
	// =====================================================================================================================
	// Flash
	if (Math.random() < 0.90 && ai.play_from_hand(card.name.winds_howl                )) {return;}
	if (Math.random() < 0.90 && ai.play_from_hand(card.name.living_volcano            )) {return;}
	if (Math.random() < 0.90 && ai.play_from_hand(card.name.return_of_the_frost_giants)) {return;}
	if (Math.random() < 0.90 && ai.play_from_hand(card.name.spine_splitter            )) {return;}
	
	// =====================================================================================================================
	// Late game - play most powerful
	if (stats.temples(player.opp) == 1 && age.major() > 0) {
		let choice = null;
		for (let card_num of zone.get(player.opp, zone.hand)) {
			if (choice == null || stats.card_power(card_num) > stats.card_power(choice)) {
				choice = card_num;
			}
		}
		if (choice == null || stats.card_power(choice) < 2) {
			ai.play_random();
		} else {
			actions.play(player.opp, choice);
		}
		return;
	}
	
	// =====================================================================================================================
	// Summon
	if (Math.random() < 0.90 && age.major() == 0 && ai.play_from_hand(card.name.nikolai_the_cursed)) {return;}
	
	// =====================================================================================================================
	// Purify
	if (Math.random() < 0.90 && ai.purify_choice() != null && ai.play_from_hand(card.name.zodiac_the_eternal)) {return;}
	
	// =====================================================================================================================
	// Subvert
	{
		let choice = ai.subvert_choice(subv.type.harmless);
		if (Math.random() < 0.90 && choice != null && stats.card_power(choice) >= 2 && ai.play_from_hand(card.name.zodiac_the_eternal)) {return;}
	}
	
	// =====================================================================================================================
	// Energy
	if (Math.random() < 0.90 && stats.temples(player.opp) >= 2 &&                                                      ai.play_from_hand(card.name.madness_of_1000_stars)) {return;}
	if (Math.random() < 0.90 && stats.temples(player.opp) >= 2 &&                                                      ai.play_from_hand(card.name.traveler             )) {return;}
	if (Math.random() < 0.90 && stats.temples(player.opp) >= 2 && stats.energy(player.opp) == 3 &&                     ai.play_from_hand(card.name.monk                 )) {return;}
	if (Math.random() < 0.75 && stats.temples(player.opp) >= 2 && stats.energy(player.opp) == 2 &&                     ai.play_from_hand(card.name.monk                 )) {return;}
	if (Math.random() < 0.75 && stats.temples(player.opp) == 3 && stats.energy(player.opp) == 1 && age.major() == 1 && ai.play_from_hand(card.name.monk                 )) {return;}
	if (Math.random() < 0.50 && stats.temples(player.opp) == 3 && stats.energy(player.opp) == 0 && age.major() == 0 && ai.play_from_hand(card.name.monk                 )) {return;}
	
	// =====================================================================================================================
	// Discard: All & Subvert: Traitorous
	if (Math.random() < 0.80 && ai.play_from_hand(card.name.final_judgment   )) {return;}
	if (Math.random() < 0.80 && ai.play_from_hand(card.name.hell_frozen_over )) {return;}
	
	// =====================================================================================================================
	// Draw
	if (Math.random() < 0.90 && age.major() <= 1 && ai.num_elemental_in_hand() <= 2 && ai.play_from_hand(card.name.akari_timeless_fighter)) {return;}
	if (Math.random() < 0.60 && age.major() <= 1 && ai.num_elemental_in_hand() <= 3 && ai.play_from_hand(card.name.akari_timeless_fighter)) {return;}
	
	// =====================================================================================================================
	// Bolster: Rivals
	if (Math.random() < 0.50 &&                     ai.play_from_hand(card.name.what_lies_beneath)) {return;}
	if (Math.random() < 0.95 && age.major() == 2 && ai.play_from_hand(card.name.what_lies_beneath)) {return;}
	
	// =====================================================================================================================
	// Dragons
	if (Math.random() < 0.90 &&                                                                                                      ai.play_from_hand(card.name.smoldering_dragon)) {return;}
	if (Math.random() < 0.90 && ai.num_in(player.none, zone.buy_top, card.name.ghost) &&                                             ai.play_from_hand(card.name.aurora_draco     )) {return;}
	if (Math.random() < 0.90 && stats.count(player.you, abilities.purify) == 0 && stats.count(player.you, abilities.protect) == 0 && ai.play_from_hand(card.name.frostbreath      )) {return;}
	if (Math.random() < 0.90 &&                                                                                                      ai.play_from_hand(card.name.caverns_defender )) {return;}
	
	// =====================================================================================================================
	// Power
	if (Math.random() < 0.90 && stats.temples(player.opp) <= 2 && ai.play_from_hand(card.name.akari_timeless_fighter)) {return;}
	if (Math.random() < 0.90 && stats.temples(player.opp) <= 2 && ai.play_from_hand(card.name.zodiac_the_eternal    )) {return;}
	if (Math.random() < 0.90 && stats.temples(player.opp) <= 2 && ai.play_from_hand(card.name.jace_winters_firstborn)) {return;}
	
	// =====================================================================================================================
	// Energy
	if (Math.random() < 0.90 && stats.temples(player.opp) == 3 && age.major() == 0 && ai.play_from_hand(card.name.monk)) {return;}
	
	// =====================================================================================================================
	// Play random
	ai.play_random();
	return;
}

// ---------------------
// buy helper functions
ai.num_owned_named = function(person, name) {
	let count = 0;
	for (let z = 0; z < 4; z++) {
		for (let card_num of zone.get(person, z)) {
			if (card[card_num].name == name) {
				count++;
			}
		}
	}
	return count;
}

ai.num_owned_elem = function(person, elem) {
	let count = 0;
	for (let z = 0; z < 4; z++) {
		for (let card_num of zone.get(person, z)) {
			if (card[card_num].elem == elem) {
				count++;
			}
		}
	}
	return count;
}

ai.from_list_get_named = function(candidates, name) {
	for (let card_num of candidates) {
		if (card[card_num].name == name) {
			return card_num;
		}
	}
	return null;
}

// ---------------------
// buy functions
ai.buy_choice = function() {
	// Buy most expensive possible
	for (let energy = stats.count(player.opp, abilities.energy) + stats.count(player.opp, abilities.energy_evanesce); energy > 0; energy--) {

		// Get candidates
		let candidates = [];
		for (let card_num of zone.get(player.none, zone.buy_top)) {
			if (card[card_num].name == card.name.ghost) {continue;}
			if (card[card_num].cost == energy) {candidates.push(card_num);}
		}
		if (candidates.length == 0) {continue;}
		
		// get a purify card, if player.you has subvert & you don't have purify
		if (energy == 1) {
			let zodiac   =  ai.from_list_get_named(candidates, card.name.zodiac_the_eternal);
			let opp_pure =  ai.num_owned_named(player.opp, card.name.zodiac_the_eternal    ) > 0;
			let you_subv =  ai.num_owned_named(player.you, card.name.jace_winters_firstborn) > 0 ||
							ai.num_owned_named(player.you, card.name.frostbreath           ) > 0 ||
							ai.num_owned_named(player.you, card.name.hell_frozen_over      ) > 0;
			if (zodiac != null && !opp_pure && you_subv) {
				return zodiac;
			}
		}
		
		// get a spine splitter card, if any ghosts in deck
		if (energy == 2) {
			let spine_splitter = ai.from_list_get_named(candidates, card.name.spine_splitter);
			let already_has    = ai.num_owned_named(player.opp, card.name.spine_splitter) > 0;
			let has_ghosts     = ai.num_owned_named(player.opp, card.name.ghost) > 0;
			if (spine_splitter != null && !already_has && has_ghosts) {
				return spine_splitter;
			}
		}
		
		// get a protect card, if player.you has subvert & you don't have protect (not 1st turn, tho)
		if (!age.first_turn() && energy == 3) {
			let sdragon  =  ai.from_list_get_named(candidates, card.name.smoldering_dragon);
			let opp_pure =  ai.num_owned_named(player.opp, card.name.smoldering_dragon     ) > 0;
			let you_subv =  ai.num_owned_named(player.you, card.name.frostbreath           ) > 0 ||
							ai.num_owned_named(player.you, card.name.hell_frozen_over      ) > 0;
			if (sdragon != null && !opp_pure && you_subv) {
				return sdragon;
			}
		}
		
		// Random, for 1 or 4 energy
		if (energy == 1 || energy == 4) {
			return math.shuffle(candidates)[0];
		}
		
		// Bias towards traveler, for 2 energy
		if (energy == 2) {
			let traveler = ai.from_list_get_named(candidates, card.name.traveler);
			if (Math.random() > 0.50 && traveler != null && ai.num_owned_named(player.opp, card.name.traveler) == 0) {
				return traveler;
			} else {
				return math.shuffle(candidates)[0];
			}
		}
		
		// Pick dragons based on what player.you has in their deck
		if (energy == 3) {
			let count = Array(4).fill(0);
			for (let elem = 0; elem < 4; elem++) {
				count[elem] = ai.num_owned_elem(player.you, card.elem.storm + elem);
			}
			for (let elem = 0; elem < 4; elem++) {
				let mf_elem = math.max_index(count); // most frequent element
				if (count[mf_elem] == 1) {return math.shuffle(candidates)[0];} // rand, if 1 of each
				let name;
				switch (mf_elem) {
					case 0: name = card.name.caverns_defender ; break;
					case 1: name = card.name.aurora_draco     ; break;
					case 2: name = card.name.smoldering_dragon; break;
					case 3: name = card.name.frostbreath      ; break;
				}
				let choice = ai.from_list_get_named(candidates, name);
				if (choice == null) {
					count[elem] = 0;
					continue;
				}
				return choice;
			}
		}
	}
	return null;
}

ai.buy = function() {
	let choice = ai.buy_choice();
	if (choice != null) {
		actions.buy(player.opp, choice);
	} else if (stats.energy(player.opp) > 0) {
		log.add({event: log.event.decline_buy, person: player.opp});
	}
}

// ====================================================================================
// Backend - Basic Actions

ai.discard = function() {
	let count = 0, max = zone.count(player.opp, zone.hand);
	for (let d = zone.count(player.opp, zone.hand) - 1; d >= 0; d--) {
		let card_num = zone.get(player.opp, zone.hand, d);
		if (card[card_num].elem == card.elem.forest || card[card_num].elem == card.elem.desert) {
			actions.discard(player.opp, card_num, {simultaneous: true});
			log.add({event: log.event.shuffle_discard, person: player.opp, card_num: card_num});
			count++;
		}
	}
	if (count == max) {
		log.add({event: log.event.shuffle_discard, person: player.opp, count: -1});
	}
	move.add_pause(1);
}

ai.purify_choice = function() {
	
	// Traitorous (powerful or interesting)
	for (let card_num of zone.get(player.you, zone.play)) {
		if (card[card_num].subv[subv.type.traitorous] != null) {
			if (stats.card_power(card_num) > 0 || card[card_num].name == card.name.what_lies_beneath) {
				return card_num;
			}
		}
	}
	
	// High-power harmless
	{
		let power = 1, choice = null;
		for (let card_num of zone.get(player.opp, zone.play)) {
			if (card[card_num].subv[subv.type.harmless] != null && card[card_num].power > power) {
				choice = card_num;
				power  = card[card_num].power;
			}
		}
		if (choice != null) {return choice;}
	}
	
	// Mindless
	for (let card_num of zone.get(player.opp, zone.play)) {
		if (card[card_num].subv[subv.type.mindless] != null && math.sum(card[card_num].active) > 0) {
			return card_num;
		}
	}
	
	// Ghost harmless (player.you)
	for (let card_num of zone.get(player.you, zone.play)) {
		if (card[card_num].subv[subv.type.harmless] != null && card[card_num].power < 0) {
			return card_num;
		}
	}
	
	// Low-power harmless
	for (let card_num of zone.get(player.opp, zone.play)) {
		if (card[card_num].subv[subv.type.harmless] != null && card[card_num].power > 0) {
			return card_num;
		}
	}
	
	// Traitorous (weak)
	for (let card_num of zone.get(player.you, zone.play)) {
		if (card[card_num].subv[subv.type.traitorous] != null) {
			if (stats.card_power(card_num) >= 0) {
				return card_num;
			}
		}
	}
	
	// Choose none
	return null;
}

ai.purify = function(num_cards) {
	// use purify_choice to decide
	for (let c = 0; c < num_cards; c++) {
		let choice = ai.purify_choice();
		if (choice != null) {
			actions.purify(choice, player.opp);
		} else {break;}
	}
}

ai.sacrifice = function(num_cards) {
	// sacrifice all ghosts, then all wizards, then all monks
	for (let x = 0; x < num_cards; x++) {
		let sacked = false;
		let choice = [card.name.ghost, card.name.wizard, card.name.monk];
		for (let c = 0; c < choice.length && !sacked; c++) {
			for (let card_num of zone.get(player.opp, zone.hand)) {
				if (card[card_num].name == choice[c]) {
					actions.sacrifice(card_num);
					sacked = true;
					break;
				}
			}
		}
	}
}

ai.substitute = function(num_cards) {
	
	// skip, if not time to substitute
	if (age.major() != age.substitute) {return;}
	
	// process each action separately
	for (let x = 0; x < num_cards; x++) {
		
		// get highest power card in your hand
		let in_choice = null;
		for (let card_num of zone.get(player.opp, zone.hand)) {
			if (in_choice == null || stats.card_power(card_num) > stats.card_power(in_choice)) {
				in_choice = card_num;
			}
		}
		
		// get lowest power card in play
		let out_choice = null;
		for (let card_num of zone.get(player.opp, zone.play)) {
			if (out_choice == null || stats.card_power(card_num) < stats.card_power(out_choice)) {
				out_choice = card_num;
			}
		}
		
		// choose course of action
		let from_hand    = in_choice != null && out_choice != null && stats.card_power(in_choice) > stats.card_power(out_choice);
		let from_deck    = !from_hand && out_choice != null && zone.count(player.opp, zone.deck) > 0 && stats.card_power(out_choice) < 2;
		
		// choose to not subst
		if (!from_hand && !from_deck) {
			log.add({event: log.event.decline, person: player.opp, ability: "Substitute"});
			return;
		}
		
		// sub w/ card from hand
		if (from_hand) {
			
			actions.discard(player.opp, out_choice);
			actions.play   (player.opp,  in_choice);
			log.add({event: log.event.substitute, person: player.opp, card_num: out_choice, card_num2: in_choice});
		
		// sub w/ top card
		} else {
			
			in_choice = zone.get_last(player.opp, zone.deck);
			actions.discard(player.opp, out_choice);
			actions.play   (player.opp,  in_choice);
			log.add({event: log.event.substitute, person: player.opp, card_num: out_choice, card_num2: in_choice});
		}
	}
}

// ====================================================================================
// Backend - Subversions

// checks if a card w/ name is in play & subvertible under player.you's control, returns card_num to that card (or null)
ai.subvertible = function(name, type) {
	for (let card_num of zone.get(player.you, zone.play)) {
		// check if in valid zone
		if (!card[card_num].ai_subvertible)     {continue;} // can't be subverted
		
		// make sure is actually played already
		if (actions.just_played[card_num]) {continue;}
		
		// check if already subverted
		if (type != subv.type.cave_in) {
			if (card[card_num].subv[type] != null) {continue;}
		} else {
			if (card[card_num].subv[subv.type.harmless] != null && card[card_num].subv[subv.type.mindless] != null) {continue;}
		}
		
		// check if name matches
		if (card[card_num].name == name) {
			return card_num;
		}
	}
	return null;
}

ai.subvert_choice = function(type) {
	
	// scope choice variable
	let choice;
	
	// =====================================================================================================================
	// Check for Cave-In
	if (type == subv.type.cave_in) {
		if (stats.count(player.you, abilities.purify) > 0) {return null;}
		if (age.major() + 1 < age.battle && (choice = ai.subvertible(card.name.final_judgment, subv.type.mindless)) != null) {
			for (let card_num of zone.get(player.opp, zone.hand)) {
				if (card[card_num].elem != card.elem.forest && card[card_num].elem != card.elem.desert) {
					return choice;
				}
			}
		}
		return null;
	}
	
	// =====================================================================================================================
	// Warriors (Fire)
	
	// Choose unused Purify
	if ((choice = ai.subvertible(card.name.zodiac_the_eternal, type)) != null && card[choice].age_played == age.major()) {
		return choice;
	}
	
	// =====================================================================================================================
	// Titans
	
	// Choose Bolster: Rivals
	if ((choice = ai.subvertible(card.name.what_lies_beneath, type)) != null) {
		return choice;
	}
	
	// Choose Discard: All if player.opp has elemental cards in their hand
	if (age.major() + 1 < age.battle && (choice = ai.subvertible(card.name.final_judgment, type)) != null) {
		for (let card_num of zone.get(player.opp, zone.hand)) {
			if (card[card_num].elem != card.elem.forest && card[card_num].elem != card.elem.desert) {
				return choice;
			}
		}
	}
	
	// Choose unused Energy: Evanesce
	if ((choice = ai.subvertible(card.name.madness_of_1000_stars, type)) != null && card[choice].age_played == age.major()) {
		return choice;
	}
	
	// =====================================================================================================================
	// Warriors (storm)
	
	// Choose Summon on first age
	if (age.major() == 0 && (choice = ai.subvertible(card.name.nikolai_the_cursed, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// Dragons (Powerful)
	
	// Choose Haunt
	if ((choice = ai.subvertible(card.name.aurora_draco, type)) != null && card[choice].age_played == age.major()) {
		return choice;
	}
	
	// Choose Powerful Bolsters
	if ((choice = ai.subvertible(card.name.aurora_draco     , type)) != null && stats.num_played_of_elem(player.opp, card.elem.fire) >= 2) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.smoldering_dragon, type)) != null && stats.num_played_of_elem(player.opp, card.elem.ice ) >= 2) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.frostbreath      , type)) != null && stats.num_played_of_elem(player.opp, card.elem.rock) >= 2) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.caverns_defender , type)) != null && stats.num_played_of_elem(player.opp, card.elem.storm ) >= 2) {
		return choice;
	}
	
	// =====================================================================================================================
	// Travelers (early game)
	if (stats.temples(player.opp) > 1 && (choice = ai.subvertible(card.name.traveler, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// Beasts
	if ((choice = ai.subvertible(card.name.winds_howl                , type)) != null && card[choice].active[abilities.flash    ] > 0) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.spine_splitter            , type)) != null && card[choice].active[abilities.sacrifice] > 0) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.return_of_the_frost_giants, type)) != null && card[choice].active[abilities.flash    ] > 0) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.living_volcano            , type)) != null && card[choice].active[abilities.flash    ] > 0) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.return_of_the_frost_giants, type)) != null) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.living_volcano            , type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// Warriors (Rock & Fire)
	if (age.major() + 1 < age.battle && (choice = ai.subvertible(card.name.akari_timeless_fighter, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// By Power
	
	if (type == subv.type.traitorous) {
		if ((choice = ai.subvertible(card.name.smoldering_dragon, type)) != null) {
			return choice;
		}
		if ((choice = ai.subvertible(card.name.caverns_defender, type)) != null) {
			return choice;
		}
		if ((choice = ai.subvertible(card.name.frostbreath, type)) != null) {
			return choice;
		}
		if ((choice = ai.subvertible(card.name.zodiac_the_eternal, type)) != null) {
			return choice;
		}
	}
	
	// =====================================================================================================================
	// Dragons (Weak)
	
	if ((choice = ai.subvertible(card.name.aurora_draco     , type)) != null && stats.num_played_of_elem(player.opp, card.elem.fire) >= 1) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.smoldering_dragon, type)) != null && stats.num_played_of_elem(player.opp, card.elem.ice ) >= 1) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.frostbreath      , type)) != null && stats.num_played_of_elem(player.opp, card.elem.rock) >= 1) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.caverns_defender , type)) != null && stats.num_played_of_elem(player.opp, card.elem.storm ) >= 1) {
		return choice;
	}
	
	// =====================================================================================================================
	// Warriors (storm)
	
	// Choose Summon on second age
	if (age.major() == 1 && (choice = ai.subvertible(card.name.nikolai_the_cursed, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// 1-power cards
	if ((choice = ai.subvertible(card.name.traveler, type)) != null) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.aurora_draco, type)) != null) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.hell_frozen_over, type)) != null) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.madness_of_1000_stars, type)) != null) {
		return choice;
	}
	if ((choice = ai.subvertible(card.name.jace_winters_firstborn, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// Monks (late game)
	if ((choice = ai.subvertible(card.name.monk, type)) != null) {
		return choice;
	}
	
	// =====================================================================================================================
	// Whatever's left
	for (let card_num of zone.get(player.you, zone.play)) {
		if (ai.subvertible(card[card_num].name)) {
			return card_num;
		}
	}
	
	// =====================================================================================================================
	// No available options
	return null;
}

ai.subvert = function(num_cards, type) {
	
	// error checking
	if (num_cards <= 0) {return;}
	
	// add protect to log
	if (stats.count(player.you, abilities.protect) > 0) {
		log.add({event: log.event.protect, person: player.you, ability: subv.to_text(type)});
		return;
	}
	
	// process subversion abilities, one-at-a-time
	for (let c = 0; c < num_cards; c++) {
		switch (type) {
			
			// Harmless
			case subv.type.harmless: {
				
				// get max-powered card
				let choice = null;
				for (let card_num of zone.get(player.you, zone.play)) {
					
					// check if subvertible
					if (card[card_num].subv[type] != null) {continue;}
					if (!card[card_num].ai_subvertible) {continue;}
					
					// check if max power seen so far
					if (choice == null) {
						choice = card_num;
					} else if (card[card_num].power > card[choice].power) {
						choice = card_num;
					}
				}
				
				// subvert
				if (choice == null) {return;}
				log.add({event: log.event.subvert_harmless, person: player.opp, card_num: choice});
				actions.subvert(choice, subv.type.harmless);
				break;
			}
			
			// Mindless / Traitorous / Cave In
			case subv.type.mindless:
			case subv.type.traitorous:
			case subv.type.cave_in: {
				
				// Get choice
				let choice = ai.subvert_choice(type);
				
				// Break out, if no valid choice
				if (choice == null) {
					if (type == subv.type.cave_in) {
						log.add({event: log.event.decline, person: player.opp, ability: "Subvert: Cave In"});
					}
					return;
				}
				
				// Subvert
				if (type == subv.type.cave_in) {
					actions.subvert(choice, subv.type.mindless);
					actions.subvert(choice, subv.type.harmless);
					actions.discard(player.opp, zone.get_last(player.opp, zone.play));
				} else {
					actions.subvert(choice, type);
				}
				
				// Add to log
				log.add({event: subv.to_event(type), person: player.opp, card_num: choice});
				
				// Exit
				break;
			}
		}
	}
}

