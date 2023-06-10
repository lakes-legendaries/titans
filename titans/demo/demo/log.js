// ====================================================================================
// Log -- reference card
// ====================================================================================

// ====================================================================================
// Interface
var log = {
	setup : function()    {}, // phaser plugin -- place button & add mouse-over action
	text  : function()    {}, // returns current log text
	add   : function(str) {}, // add str to log
	last  : function()    {}, // returns last entry in log
	cut   : function()    {}, // remove oldest entry
	active: false,            // whether log should be shown (buttons & other actions toggle this)
	event : {                 // type of thing added to log
		battle            :  0,
		buy               :  1,
		decline           :  2,
		decline_buy       :  3,
		discard           :  4,
		draw              :  5,
		flash             :  6,
		haunt             :  7,
		new_age           :  8,
		play              :  9,
		protect           : 10,
		purify            : 11,
		sacrifice         : 12,
		shuffle_discard   : 13,
		shuffle_draw      : 14,
		shuffle_traitorous: 15,
		substitute        : 16,
		subvert_cave_in   : 17,
		subvert_harmless  : 18,
		subvert_mindless  : 19,
		subvert_traitorous: 20,
		summon            : 21,
		surge             : 22,
	},
	unhide: function() {}, // remove the 'hide' field from hidden elements, making them visible
}

log.setup = function() {
	log.button = env.add.image(log.button_pos, 'log button');
	log.button.setInteractive();
	log.button.on('pointerdown', () => {log.active = !log.active;});
	
	// initialize log
	log.history = ["[color=purple][u]Age 1[/u][/color]"]
	log.add({event: log.event.new_age, age: 0});
	log.active  = false;
}

log.text = function() {
	let str = "";
	for (let g = 0; g < log.history.length; g++) {
		
		// get current elem
		let elem = log.history[g];
		if (elem.hide) {continue;}
		if (typeof elem == 'string') {continue;}
		
		// combine subsequent similar elements
		switch (elem.event) {
			case log.event.flash:
			case log.event.purify:
			case log.event.sacrifice:
			case log.event.shuffle_discard:
			case log.event.substitute:
			case log.event.subvert_harmless:
			case log.event.subvert_mindless:
			case log.event.subvert_traitorous:
			case log.event.subvert_cave_in:
			case log.event.summon:
				while (g + 1 < log.history.length && elem.event == log.history[g+1].event && elem.person == log.history[g+1].person) {
					
					// change to array, add extra card
					if (!Array.isArray(elem.card_num)) {
						elem.card_num = [elem.card_num];
					}
					elem.card_num.push(log.history[g+1].card_num);
					
					// add secondary card
					if (elem.card_num2 != null) {
						if (!Array.isArray(elem.card_num2)) {
							elem.card_num2 = [elem.card_num2];
						}
						elem.card_num2.push(log.history[g+1].card_num2);
					}
					
					// change to discard all
					if (elem.event == log.event.shuffle_discard && log.history[g+1].count == -1) {
						elem.count = -1;
					}
					
					// remove extra event
					log.history.splice(g+1, 1);
				}
				break;
		}
		
		// process
		switch (elem.event) {
			case log.event.battle: {
				switch (elem.person) {
					case null:
						str += "The battle was fought to a draw.";
						break;
					default:
						str += log.subject(elem.person) + " won the battle, ";
						str += elem.capture? "capturing": "destroying";
						str += " 1 of " + log.possessive_noun(+(!elem.person)) + " temples.";
						break;
				}
				break;
			}
			case log.event.buy: {
				str += log.subject(elem.person) + " used " + log.ability("Energy") + " to awaken " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.decline: {
				str += log.subject(elem.person) + " chose to not use " + log.ability(elem.ability) + ".";
				break;
			}
			case log.event.decline_buy: {
				str += log.subject(elem.person) + " chose to not awaken a card.";
				break;
			}
			case log.event.discard: {
				str += log.subject(elem.person) + " used " + log.ability("Discard") + " to force " + log.direct_object(+(!elem.person)) + " to discard " + elem.count.toString() + " randomly-chosen cards.";
				break;
			}
			case log.event.draw: {
				str += log.subject(elem.person) + " used " + log.ability("Draw") + " to draw " + elem.count.toString() + " cards.";
				break;
			}
			case log.event.flash: {
				str += log.subject(elem.person) + " used " + log.ability("Flash") + " to play " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.haunt: {
				str += log.subject(elem.person) + " used " + log.ability("Haunt") + " to force " + log.direct_object(+(!elem.person)) + " to gain " + elem.count.toString() + " " + log.card("Ghost" + (elem.count > 1? "s": "")) + " into play (" + log.ability("Subverted") + "with " + log.subv("Harmless") + ").";
				break;
			}
			case log.event.new_age: {
				let name;
				if (elem.age == age.battle) {
					name = "End Of Turn";
				} else {
					name = "Age " + (elem.age + 1).toString();
				}
				str += "[color=purple][u]" + name + "[/u][/color]";
				break;
			}
			case log.event.play: {
				str += log.subject(elem.person) + " played " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.protect: {
				str += log.subject(elem.person) + " blocked " + log.possessive_noun(+(!elem.person)) + " " + log.ability(elem.ability) + " with " + log.ability("Protect") + ".";
				break;
			}
			case log.event.purify: {
				str += log.subject(elem.person) + " used " + log.ability("Purify") + " to remove all " + log.subv("subversions") + " from " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.sacrifice: {
				str += log.subject(elem.person) + " used " + log.ability("Sacrifice") + " to permanently remove " + log.possessive(elem.person) + " " + log.card(elem.card_num) + " from the game.";
				break;
			}
			case log.event.shuffle_discard: {
				str += log.subject(elem.person) + " discarded ";
				if (elem.count == -1) {
					str += log.possessive(elem.person) + " entire hand.";
				} else {
					str += log.card(elem.card_num) + ".";
				}
				break;
			}
			case log.event.shuffle_draw: {
				str += log.subject(elem.person) + " drew " + elem.count.toString() + " cards for next turn.";
				break;
			}
			case log.event.shuffle_traitorous: {
				str += log.card(elem.card_num) + " had " + log.subv("Traitorous") + " removed from it, returning it to " + log.possessive_noun(elem.person) + " deck.";
				break;
			}
			case log.event.substitute: {
				str += log.subject(elem.person) + " used " + log.ability("Substitute") + " to replace " + log.card(elem.card_num) + " with " + log.card(elem.card_num2) + ".";
				break;
			}
			case log.event.subvert_cave_in: {
				str += log.subject(elem.person) + " used " + log.ability("Subvert: Cave In") + " to attach " + log.subv("Harmless & Mindless") + " to " + log.possessive_noun(+(!elem.person)) + " " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.subvert_harmless: {
				str += log.subject(elem.person) + " used " + log.ability("Subvert: Harmless") + " to attach " + log.subv("Harmless") + " to " + log.possessive_noun(+(!elem.person)) + " " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.subvert_mindless: {
				str += log.subject(elem.person) + " used " + log.ability("Subvert: Mindless") + " to attach " + log.subv("Mindless") + " to " + log.possessive_noun(+(!elem.person)) + " " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.subvert_traitorous: {
				str += log.subject(elem.person) + " used " + log.ability("Subvert: Traitorous") + " to attach " + log.subv("Traitorous") + " to " + log.possessive_noun(+(!elem.person)) + " " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.summon: {
				str += log.subject(elem.person) + " used " + log.ability("Summon") + " to play " + log.card(elem.card_num) + ".";
				break;
			}
			case log.event.surge: {
				str += log.subject(elem.person) + " used " + log.ability("Surge") + " to draw a new hand.";
				break;
			}
		}
		
		// end log entry
		str += "\n";
	}
	
	// move punctuation to w/in bbcode blocks (to stop being pushed to the next line)
	{
		let cmd   = '[/color]';
		let punct = ',.()!?';
		for (let c = 0; c + cmd.length < str.length; c++) {
			if (!str.substring(c, c + cmd.length).localeCompare(cmd)) {
				for (let p of punct) {
					if (str[c+cmd.length] == p) {
						str = str.substring(0, c) + p + str.substring(c, c + cmd.length) + str.substring(c+cmd.length+1, str.length);
					}
				}
			}
		}
	}
	
	// return
	return str;
}

log.last = function() {
	return log.history[log.history.length-1];
}

log.add = function(obj) {
	log.history.push(obj);
	while (log.history.length > log.record_length) {
		log.cut();
	}
}

log.cut = function() {
	log.history.splice(0, 1);
}

log.unhide = function() {
	for (let g = 0; g < log.history.length; g++) {
		if (log.history[g].hide) {
			log.history[g].hide = false;
		}
	}
}

// ====================================================================================
// Backend
log.button_pos    = {x: 1582, y: 146};
log.record_length = 20;

log.subject = function(person) {
	return "[color=red]" + (person == player.you? "You": "Karen") + "[/color]";
}
log.direct_object = function(person) {
	return "[color=red]" + (person == player.you? "you": "Karen") + "[/color]";
}
log.possessive = function(person) {
	return "[color=red]" + (person == player.you? "your": "her") + "[/color]";
}
log.possessive_noun = function(person) {
	return "[color=red]" + (person == player.you? "your": "Karen's") + "[/color]";
}
log.ability = function(ability) {
	return "[color=green]" + ability + "[/color]";
}
log.card = function(card_num) {
	
	// process several, if is array
	if (Array.isArray(card_num)) {
		let name = "";
		for (let g = 0; g < card_num.length; g++) {
			name += log.card(card_num[g]);
			if (g + 1 == card_num.length) {continue;}       // no trailing characters
			if (card_num.length > 2)      {name += ",";}    // add comma
			if (g + 2 == card_num.length) {name += " and";} // add 'and'
			name += " ";                                    // add trailing space
		}
		return name;
	}
	
	// process single
	{
		let name;
		if (typeof card_num == 'string') {
			name = card_num;
		} else {
			name = card.num2str(card_num);
		}
		return "[color=blue]" + name + "[/color]";
	}
}
log.subv = function(subv) {
	return "[color=teal]" + subv + "[/color]";
}