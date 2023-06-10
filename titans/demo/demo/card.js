// ====================================================================================
// Card sprites & properties
// ====================================================================================

// ====================================================================================
// Interface
var card = {
	
	// as array: ([card_num]) all cards
	// each elem of array has the following fields:
	//   abilities      - array [num_abilities]   - count of each of card's abilities (as written on card)
	//   active         - array [num_abilities]   - count of currently active abilities (i.e. unused, unremoved via subversions)
	//   subv           - array [num_subversions] - which subversion is currently attached (null = none)
	//   age_played     - int                     - age card was played (for subvert: cave in)
	//   ai_subvertible - boolean                 - whether ai can subvert this card (necessary b/c order decisions made in controller)
	//   name           - enum int                - card name
	//   elem           - enum int                - card elem
	//   spec           - enum int                - card species
	//   power          - int                     - card power (as listed on card)
	//   cost           - int                     - card cost
	
	// query current position of a card
	coords: function(card_num) {},
	
	// meta-data
	num     : 108,
	count   : function(name) {}, // count of any given name of card
	num_subv: function(card_num) {}, // number of subversions on a card
	
	// enums
	ordinal: {
		monk    :  0,
		wizard  : 16,
		traveler: 24,
		ghost   : 32,
		buy0    : 44,
	},
	name: {
		monk                      :  0,
		wizard                    :  1,
		traveler                  :  2,
		ghost                     :  3,
		nikolai_the_cursed        :  4,
		zodiac_the_eternal        :  5,
		jace_winters_firstborn    :  6,
		akari_timeless_fighter    :  7,
		winds_howl                :  8,
		living_volcano            :  9,
		return_of_the_frost_giants: 10,
		spine_splitter            : 11,
		aurora_draco              : 12,
		smoldering_dragon         : 13,
		frostbreath               : 14,
		caverns_defender          : 15,
		madness_of_1000_stars     : 16,
		final_judgment            : 17,
		hell_frozen_over          : 18,
		what_lies_beneath         : 19,
		num                       : 20,
		buy0                      :  4,
	},
	elem: {
		forest: 0,
		storm   : 1,
		fire  : 2,
		ice   : 3,
		rock  : 4,
		desert: 5,
		none  : 6,
	},
	species: {
		dweller: 0,
		warrior: 1,
		beast  : 2,
		dragon : 3,
		titan  : 4,
	},
	
	// for human interface -- name to str
	num2str : function(card_num) {return card.name2str(card[card_num].name);},
	name2str: function(name) {},
	
	// reset stats
	play   : function(card_num) {}, // set all abilities to active
	discard: function(card_num) {}, // unattach all subversions, reset all abilities
	discard: function()         {}, // discard(x) for all x
	
	// phaser plugins
	setup : function() {}, // create card objects, put in zones
}

card.coords = function(card_num) {
	let obj = card[card_num].sprite;
	return {x: obj.x, y: obj.y}
}

card.count = function(name) {
	if (name == card.name.monk    ) {return 16;}
	if (name == card.name.wizard  ) {return  8;}
	if (name == card.name.traveler) {return  8;}
	if (name == card.name.ghost   ) {return 12;}
	return 4;
}

card.num_subv = function(card_num) {
	let out = 0;
	for (let type = 0; type < subv.type.num; type++) {
		out += +(card[card_num].subv[type] != null);
	}
	return out;
}

card.name2str = function(name) {
	switch (name) {
		case card.name.monk                      : return "Monk";
		case card.name.wizard                    : return "Wizard";
		case card.name.traveler                  : return "Traveler";
		case card.name.ghost                     : return "Ghost";
		case card.name.nikolai_the_cursed        : return "Nikolai, The Cursed";
		case card.name.zodiac_the_eternal        : return "Zodiac, The Eternal";
		case card.name.jace_winters_firstborn    : return "Jace, Winter's Firstborn";
		case card.name.akari_timeless_fighter    : return "Akari, Timeless Fighter";
		case card.name.winds_howl                : return "Wind's Howl";
		case card.name.living_volcano            : return "Living Volcano";
		case card.name.return_of_the_frost_giants: return "Return Of The Frost Giants";
		case card.name.spine_splitter            : return "Spine Splitter";
		case card.name.aurora_draco              : return "Aurora Draco";
		case card.name.smoldering_dragon         : return "Smoldering Dragon";
		case card.name.frostbreath               : return "Frostbreath";
		case card.name.caverns_defender          : return "Cavern's Defender";
		case card.name.madness_of_1000_stars     : return "Madness Of 1,000 Stars";
		case card.name.final_judgment            : return "Final Judgment";
		case card.name.hell_frozen_over          : return "Hell, Frozen Over";
		case card.name.what_lies_beneath         : return "What Lies Beneath";
	}
	return null;
}

card.play = function(card_num) {
	card[card_num].active = card[card_num].abilities.slice();
}

card.discard = function(card_num) {
	
	// discard all, if card_num not provided
	if (card_num == null) {
		for (let card_num = 0; card_num < card.num; card_num++) {
			card.discard(card_num);
		}
		return;
	}
	
	// remove subversions
	for (let type = 0; type < subv.type.num; type++) {
		if (card[card_num].subv[type] == null) {continue;}
		subv.reset(card[card_num].subv[type], type);
		card[card_num].subv[type] = null;
	}
	
	// deactivate abilities
	card[card_num].active = Array(abilities.num).fill(0);
}

card.setup = function() {
	
	// Create cards & sprites
	for (let card_num = 0; card_num < card.num; card_num++) {
		
		// Create card
		card[card_num] = card.props(card_num);
		
		// Make sprite, attach to cards array
		card[card_num].sprite = env.add.sprite(env.nowhere, 'half cards');
		card[card_num].sprite.setFrame(card[card_num].sprite_frame);
	}
	
	// Build starting decks
	for (let card_num = 0; card_num < card.ordinal.traveler; card_num++) {
		let person = +(card_num < card.count(card.name.monk) / 2 || (card_num >= card.ordinal.wizard && card_num < card.ordinal.wizard + card.count(card.name.wizard) / 2));
		zone.add(person, zone.deck, card_num);
	}
	
	// Prepare starting decks
	// Make sure opp starts with a wizard & you have at least 3 monks (required for tutorial to work right)
	//   NOTE: In the real game, no matter what, you'll be able to play 3 monks on your 1st turn
	//         (b/c you'll always draw at least 2, and can play the top card of your deck,
	//			which will be guaranteed to be a monk then)
	let has_wizard, monk_count;
	do {
		// shuffle decks
		zone.shuffle();
		
		// opp wizard
		has_wizard = false;
		for (let d = 7; d < 12; d++) {
			let card_num = zone.get(player.opp, zone.deck, d);
			if (card[card_num].name == card.name.wizard) {
				has_wizard = true;
				break;
			}
		}
		
		// you monks
		monk_count = 0;
		for (let d = 7; d < 12; d++) {
			let card_num = zone.get(player.you, zone.deck, d);
			if (card[card_num].name == card.name.monk) {
				monk_count++;
			}
		}
	} while (!has_wizard || monk_count < 3);
	
	// Show starting decks
	for (let person = 0; person < player.num; person++) {
		for (let d = 0; d < zone.count(person, zone.deck); d++) {
			let card_num = zone.get(person, zone.deck, d);
			let sprite   = card[card_num].sprite;
			sprite.setFrame(0);
			env.place(sprite, zone.dest(person, zone.deck, d));
			env.to_front(sprite);
		}
	}
	
	// Show buy piles
	card.place(card.name.buy0    , card.ordinal.buy0);
	card.place(card.name.ghost   , card.ordinal.ghost   , 4, 0);
	card.place(card.name.traveler, card.ordinal.traveler, 4, 1);
}

// ====================================================================================
// Backend
card.place = function(first_card_name, first_card_ordinal, row, col) {
	let count = card.count(first_card_name);
	let limit = first_card_ordinal == card.ordinal.buy0? card.num: first_card_ordinal + count;
	for (let card_num = first_card_ordinal; card_num < limit; card_num++) {
		crow = row;
		ccol = col;
		if (crow == null) {crow = Math.floor(((card_num - first_card_ordinal) % Math.pow(count, 2))/count);}
		if (ccol == null) {ccol = Math.floor(( card_num - first_card_ordinal) / Math.pow(count, 2));}
		let pos    = (card_num - first_card_ordinal) % count;
		let coords = zone.buy_pos(crow, ccol, pos, count);
		let place  = (card_num - first_card_ordinal) % count == count - 1? zone.buy_top: zone.buy_other;
		env .place   (card[card_num].sprite, coords);
		env .to_front(card[card_num].sprite);
		zone.add  (player.none, place, card_num);
	}
}

card.props = function(card_num) {
	
	// Create card, initialize abilities & subversions
	let cur_card = {};
	cur_card.abilities = Array(abilities.num).fill(0); // card's abilities
	cur_card.active    = Array(abilities.num).fill(0); // abilities that are to be activated
	cur_card.subv      = [];                           // if card is subverted, this is the subversion card's number
	cur_card.subv[subv.type.harmless  ] = null;
	cur_card.subv[subv.type.mindless  ] = null;
	cur_card.subv[subv.type.traitorous] = null;
	cur_card.age_played  = null;
	cur_card.ai_subvertible = false;
	
	// Initialize card-specific properties
	if (card_num < card.ordinal.wizard) { // 16 monks
		cur_card.name             = card.name.monk;
		cur_card.elem             = card.elem.forest;
		cur_card.spec             = card.species.dweller;
		
		cur_card.power            =  0;
		cur_card.cost             =  0;
		
		cur_card.abilities[abilities.energy] = 1;
		
	} else if (card_num < card.ordinal.traveler) { // 8 wizards
		cur_card.name             = card.name.wizard;
		cur_card.elem             = card.elem.forest;
		cur_card.spec             = card.species.dweller;
		
		cur_card.power            =  1;
		cur_card.cost             =  0;
		
	} else if (card_num < card.ordinal.ghost) { // 8 travelers
		cur_card.name             = card.name.traveler;
		cur_card.elem             = card.elem.forest;
		cur_card.spec             = card.species.dweller;
		
		cur_card.power            =  1;
		cur_card.cost             =  2;
		
		cur_card.abilities[abilities.energy] = 2;
		
	} else if (card_num < card.ordinal.buy0) { // 12 ghosts
		cur_card.name             = card.name.ghost;
		cur_card.elem             = card.elem.desert;
		cur_card.spec             = card.species.dweller;
		
		cur_card.power            = -1;
		cur_card.cost             =  1;
		
	} else if (card_num < 48) { // 4 storm warriors
		cur_card.name             = card.name.nikolai_the_cursed;
		cur_card.elem             = card.elem.storm;
		cur_card.spec             = card.species.warrior;
		
		cur_card.power            =  0;
		cur_card.cost             =  1;
		
		cur_card.abilities[abilities.summon] = 1;
		
	} else if (card_num < 52) { // 4 fire warriors
		cur_card.name             = card.name.zodiac_the_eternal;
		cur_card.elem             = card.elem.fire;
		cur_card.spec             = card.species.warrior;
		
		cur_card.power            =  2;
		cur_card.cost             =  1;
		
		cur_card.abilities[abilities.purify] = 1;
		
	} else if (card_num < 56) { // 4 ice warriors
		cur_card.name             = card.name.jace_winters_firstborn;
		cur_card.elem             = card.elem.ice;
		cur_card.spec             = card.species.warrior;
		
		cur_card.power            =  1;
		cur_card.cost             =  1;
		
		cur_card.abilities[abilities.subvert_harmless] = 1;
		
	} else if (card_num < 60) { // 4 rock warriors
		cur_card.name             = card.name.akari_timeless_fighter;
		cur_card.elem             = card.elem.rock;
		cur_card.spec             = card.species.warrior;
		
		cur_card.power            =  2;
		cur_card.cost             =  1;
		
		cur_card.abilities[abilities.draw] = 2;
		
	} else if (card_num < 64) { // 4 storm beasts
		cur_card.name             = card.name.winds_howl;
		cur_card.elem             = card.elem.storm;
		cur_card.spec             = card.species.beast;
		
		cur_card.power            = -1;
		cur_card.cost             =  2;
		
		cur_card.abilities[abilities.flash] = 2;
		
	} else if (card_num < 68) { // 4 fire beasts
		cur_card.name             = card.name.living_volcano;
		cur_card.elem             = card.elem.fire;
		cur_card.spec             = card.species.beast;
		
		cur_card.power            =  0;
		cur_card.cost             =  2;
		
		cur_card.abilities[abilities.flash  ] = 1;
		cur_card.abilities[abilities.discard] = 2;
		
	} else if (card_num < 72) { // 4 ice beasts
		cur_card.name             = card.name.return_of_the_frost_giants;
		cur_card.elem             = card.elem.ice;
		cur_card.spec             = card.species.beast;
		
		cur_card.power            =  0;
		cur_card.cost             =  2;
		
		cur_card.abilities[abilities.flash     ] = 1;
		cur_card.abilities[abilities.substitute] = 1;
		
	} else if (card_num < 76) { // 4 rock beasts
		cur_card.name             = card.name.spine_splitter;
		cur_card.elem             = card.elem.rock;
		cur_card.spec             = card.species.beast;
		
		cur_card.power            =  0;
		cur_card.cost             =  2;
		
		cur_card.abilities[abilities.flash    ] = 1;
		cur_card.abilities[abilities.sacrifice] = 2;
		
	} else if (card_num < 80) { // 4 storm dragons
		cur_card.name             = card.name.aurora_draco;
		cur_card.elem             = card.elem.storm;
		cur_card.spec             = card.species.dragon;
		
		cur_card.power            =  1;
		cur_card.cost             =  3;
		
		cur_card.abilities[abilities.haunt       ] = 2;
		cur_card.abilities[abilities.bolster_fire] = 1;
		
	} else if (card_num < 84) { // 4 fire dragons
		cur_card.name             = card.name.smoldering_dragon;
		cur_card.elem             = card.elem.fire;
		cur_card.spec             = card.species.dragon;
		
		cur_card.power            =  3;
		cur_card.cost             =  3;
		
		cur_card.abilities[abilities.protect    ] = 1;
		cur_card.abilities[abilities.bolster_ice] = 1;
		
	} else if (card_num < 88) { // 4 ice dragons
		cur_card.name             = card.name.frostbreath;
		cur_card.elem             = card.elem.ice;
		cur_card.spec             = card.species.dragon;
		
		cur_card.power            =  2;
		cur_card.cost             =  3;
		
		cur_card.abilities[abilities.subvert_mindless] = 1;
		cur_card.abilities[abilities.bolster_rock    ] = 1;
		
	} else if (card_num < 92) { // 4 rock dragons
		cur_card.name             = card.name.caverns_defender;
		cur_card.elem             = card.elem.rock;
		cur_card.spec             = card.species.dragon;
		
		cur_card.power            =  3;
		cur_card.cost             =  3;
		
		cur_card.abilities[abilities.subvert_cave_in] = 1;
		cur_card.abilities[abilities.bolster_storm    ] = 1;
		
	} else if (card_num < 96) { // 4 storm titans
		cur_card.name             = card.name.madness_of_1000_stars;
		cur_card.elem             = card.elem.storm;
		cur_card.spec             = card.species.titan;
		
		cur_card.power            =  1;
		cur_card.cost             =  4;
		
		cur_card.abilities[abilities.energy_evanesce] = 4;
		
	} else if (card_num < 100) { // 4 fire titans
		cur_card.name             = card.name.final_judgment;
		cur_card.elem             = card.elem.fire;
		cur_card.spec             = card.species.titan;
		
		cur_card.power            =  2;
		cur_card.cost             =  4;
		
		cur_card.abilities[abilities.discard] = card.num;
		
	} else if (card_num < 104) { // 4 ice titans
		cur_card.name             = card.name.hell_frozen_over;
		cur_card.elem             = card.elem.ice;
		cur_card.spec             = card.species.titan;
		
		cur_card.power            =  1;
		cur_card.cost             =  4;
		
		cur_card.abilities[abilities.subvert_traitorous] = 1;
		
	} else if (card_num < 108) { // 4 rock titans
		cur_card.name             = card.name.what_lies_beneath;
		cur_card.elem             = card.elem.rock;
		cur_card.spec             = card.species.titan;
		
		cur_card.power            =  0;
		cur_card.cost             =  4;
		
		cur_card.abilities[abilities.bolster_rivals] = 1;
	}
	
	// Get sprite frame
	cur_card.sprite_frame = cur_card.name + 4;
	
	// Return
	return cur_card;
}