// ====================================================================================
// Stats - temples, surges, power
// ====================================================================================

// ====================================================================================
// Interface
var stats = {
	// phaser plugins
	setup       : function() {},       // phaser plugin - create text sprites
	update      : function() {},       // phaser plugin - update text sprites, perform battle (if after 3rd age)
	
	// update once move queue is empty
	queue_update: function() {},       // update() once the move queue is empty
	
	// battle stats
	winner      : function() {},         // winner of battle
	capture     : function() {},         // true if winner captures loser's temple
	card_power  : function(card_num) {}, // current power of given card
	
	// temple count
	temples     : function(person) {return stats.num.temples[person];}, // returns # of temples
	
	// current power
	power       : function(person) {return stats.num.power[person];},
	
	// surge interface
	num_surges  : function(person) {return stats.num.surges[person];},
	use_surge   : function(person) {stats.num.surges[person]--;},
	
	// player's abilities interface
	count    : function(person, ability)            {}, // count of a given ability (only active, in play, count)
	energy   : function(person)                     {}, // adds energy & energy_evanesce for given person
	decrement: function(person, ability, num_times) {}, // deactivate a given ability (num_times = -1 -> deactivate all)
	
	// is game over?
	game_over: function() {return stats.temples(player.you) == 0 || stats.temples(player.opp) == 0;},
}

stats.setup = function() {
	
	// initial states
	stats.num = {
		temples: [3, 3],
		surges : [2, 2],
		power  : [0, 0],
	}
	stats.perform_update = false;
	
	stats.sprites = {
		ages   : env.add.sprite(stats.position.ages, 'age text'),
		temples: [],
		surges : [],
		power  : [[], []],
		delta  : [],
	};
	for (let person = 0; person < player.num; person++) {
		// Create sprites
		stats.sprites.temples[person] = env.add.sprite(stats.position.temples[person], 'numbers');
		stats.sprites.surges [person] = env.add.sprite(stats.position.surges [person], 'numbers');
		stats.sprites.delta  [person] = env.add.sprite(stats.position.delta  [person], 'delta');
		for (let digit = 0; digit < stats.position.power[person].length; digit++) {
			stats.sprites.power[person][digit] = env.add.sprite(stats.position.power[person][digit], 'numbers');
		}
		
		// Set defaults
		stats.sprites.temples[person].setFrame(3);
		stats.sprites.surges [person].setFrame(2);
		stats.sprites.delta  [person].setFrame(2);
		stats.sprites.power  [person][0].setFrame(0);
		for (let digit = 1; digit < stats.position.power[person].length; digit++) {
			stats.sprites.power[person][digit].setFrame(11);
		}
	}
}

stats.update = function() {
	
	// Check if update should be performed
	if (!stats.perform_update || move.in_progress()) {return;}
	stats.perform_update = false;
	
	// Update age
	stats.sprites.ages.setFrame(age.major());
	
	// Update temples
	for (let person = 0; person < player.num; person++) {
		stats.sprites.temples[person].setFrame(stats.num.temples[person]);
	}
	
	// Update surge
	for (let person = 0; person < player.num; person++) {
		stats.sprites.surges[person].setFrame(stats.num.surges[person]);
	}
	
	// Update power
	stats.update_power();
	for (let person = 0; person < player.num; person++) {
		
		// Blank-out sprite
		let sprite = stats.sprites.power[person];
		for (let digit = 0; digit < stats.sprites.power[person].length; digit++) {
			sprite[digit].setFrame(11);
		}
		
		// Update text
		let power             = stats.num.power[person];
		let use_minus_sign    = power < 0;
		let use_double_digits = power >= 10 || power <= -10;
		if (use_minus_sign) {
			sprite[0].setFrame(10);
		}
		if (use_double_digits) {
			sprite[+use_minus_sign  ].setFrame(Math.floor(Math.abs(power) / 10));
			sprite[+use_minus_sign+1].setFrame(Math.abs(power) % 10);
		} else {
			sprite[+use_minus_sign  ].setFrame(Math.abs(power));
		}
	}
	
	// Resolve battle
	if (age.major() == age.battle) {
		
		// Resolve battle
		let winner  = stats.winner ();
		if (winner != null) {
			let loser   = +(winner == 0);
			
			// Loser loses a temple
			stats.num.temples [loser]--;
			stats.sprites.delta[loser].setFrame(0);
			
			// Winner captures a temple
			if (stats.capture()) {
				stats.num.temples[winner]++;
				stats.sprites.delta[winner].setFrame(1);
			}
		}
		
	// No on-going battle; blank out delta sprites
	} else {
		for (let person = 0; person < player.num; person++) {
			stats.sprites.delta[person].setFrame(2);
		}
	}
}

stats.queue_update = function() {
	stats.perform_update = true;
}

stats.capture = function() {
	let winner = stats.winner();
	if (winner == null) {return false;}
	return stats.num.temples[winner] == 1;
}

stats.winner = function() {
	stats.update_power();
	let dif = stats.num.power[player.you] - stats.num.power[player.opp];
	if (dif >=  2) {return player.you;}
	if (dif <= -2) {return player.opp;}
	return null;
}

stats.card_power = function(card_num) {
	
	// get base power
	let power = card[card_num].subv[subv.type.harmless] == null? card[card_num].power: 0;
	
	// add in bolster
	if (card[card_num].subv[subv.type.mindless] == null) {
		let person  = zone.find(card_num).person;
		let ability = [abilities.bolster_storm, abilities.bolster_fire, abilities.bolster_ice, abilities.bolster_rock, abilities.bolster_rivals];
		let elem    = [card.elem.storm, card.elem.fire, card.elem.ice, card.elem.rock, null];
		for (let a = 0; a < ability.length; a++) {
			if (!card[card_num].active[ability[a]]) {continue;}
			let add   = stats.num_played_of_elem(+(!person), elem[a]);
			if (elem[a] == null && add > 6) {add = 6;}
			power += add;
		}
	}
	
	// return
	return power;
}

stats.count = function(person, ability) {
	let count = 0;
	for (let card_num of zone.get(person, zone.play)) {
		if (card[card_num].subv[subv.type.mindless] != null) {continue;}
		count += card[card_num].active[ability];
	}
	return count;
}

stats.energy = function(person) {
	return stats.count(person, abilities.energy) + stats.count(person, abilities.energy_evanesce);
}

stats.decrement = function(person, ability, num_times) {
	let num_decremented = 0;
	for (let card_num of zone.get(person, zone.play)) {
		num_active = card[card_num].active[ability];
		for (let a = 0; a < num_active && (num_times == -1 || num_decremented < num_times); a++, num_decremented++) {
			card[card_num].active[ability]--;
		}
	}
}

// ====================================================================================
// Backend
stats.position = {temples: [], surges: [], power: [[], []], delta: []}
stats.position.ages                   = {x: 1770, y:  345}
stats.position.temples[player.you]    = {x: 1425, y:  925};
stats.position.temples[player.opp]    = {x: 1425, y:   42};
stats.position.surges [player.you]    = {x: 1425, y:  985};
stats.position.surges [player.opp]    = {x: 1425, y:  102};
stats.position.delta  [player.you]    = {x: 1475, y:  925};
stats.position.delta  [player.opp]    = {x: 1475, y:   42};
stats.position.power  [player.you][0] = {x: 1425, y: 1045};
stats.position.power  [player.you][1] = {x: 1460, y: 1045};
stats.position.power  [player.you][2] = {x: 1495, y: 1045};
stats.position.power  [player.opp][0] = {x: 1425, y:  162};
stats.position.power  [player.opp][1] = {x: 1460, y:  162};
stats.position.power  [player.opp][2] = {x: 1495, y:  162};

stats.num_played_of_elem = function(person, elem) {
	let num = 0;
	for (let d = 0; d < zone.count(person, zone.play); d++) {
		let card_num = zone.get(person, zone.play, d);
		if (elem == null || card[card_num].elem == elem) {
			num++;
		}
	}
	return num;
}

stats.update_power = function() {
	for (let person = 0; person < player.num; person++) {
		// Update number
		stats.num.power[person] = 0;
		for (let card_num of zone.get(person, zone.play)) {
			if (card[card_num].subv[subv.type.harmless] != null) {continue;}
			stats.num.power[person] += card[card_num].power;
		}
		
		// Add in bolsters
		let ability = [abilities.bolster_storm, abilities.bolster_fire, abilities.bolster_ice, abilities.bolster_rock, abilities.bolster_rivals];
		let elem    = [card.elem.storm, card.elem.fire, card.elem.ice, card.elem.rock, null];
		for (let a = 0; a < ability.length; a++) {
			let count = stats.count(person, ability[a]);
			if (count == 0) {continue;}
			let add   = stats.num_played_of_elem(+(!person), elem[a]);
			if (elem[a] == null && add > 6) {add = 6;}
			stats.num.power[person] += count * add;
		}
	}
}