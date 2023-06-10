// ====================================================================================
// Subversions - each card's status
// ====================================================================================

// ====================================================================================
// Interface
var subv = {
	
	// as 2D array -- [subv_card_num][type]
	
	// unattach all
	reset: function(subv_card_num, type) {}, // card_num = null -> reset all
	
	// enum for subversions
	type: {
		harmless  : 0,
		mindless  : 1,
		traitorous: 2,
		num       : 3,
		cave_in   : 3,
	},
	
	// convert
	to_ability  : function(type)    {},
	to_age      : function(type)    {},
	to_text     : function(type)    {return abilities.to_text(subv.to_ability(type));},
	to_event    : function(type)    {},
	from_ability: function(ability) {},
	from_age    : function(step)    {},
	
	// meta-data
	num   : 30, // count of each type of card spawned
	offset: function(pos_in_seq, num_attached)  {}, // returns offset
	next  : function(type) {}, // returns index of next unattached subversion of given type, marks attached
	
	// Phaser plugins
	setup: function() {},
}

subv.reset = function(subv_card_num, type) {
	
	// Place card off to the side
	if (subv_card_num != null) {
		env.place(subv[subv_card_num][type], env.nowhere);
		return;
	}
	
	// Process all
	
	// Mark all as unattached
	subv.attached = Array(subv.type.num).fill(-1);
	
	// Process each card
	for (let s = 0; s < subv.num; s++) {
		for (let type = 0; type < subv.type.num; type++) {
			subv.reset(s, type);
		}
	}
}

subv.offset = function(pos_in_seq, num_attached) {
	if (pos_in_seq   == null) {pos_in_seq   = 0;}
	if (num_attached == null) {num_attached = 1;}
	return {
		x:  (pos_in_seq + 1) / num_attached * 10,
		y: -(pos_in_seq + 1) / num_attached * 50,
	};
}

subv.to_ability = function(type) {
	switch (type) {
		case subv.type.harmless  : return abilities.subvert_harmless;
		case subv.type.mindless  : return abilities.subvert_mindless;
		case subv.type.traitorous: return abilities.subvert_traitorous;
		case subv.type.cave_in   : return abilities.subvert_cave_in;
	}
	return null;
}

subv.to_age = function(type) {
	switch (type) {
		case subv.type.harmless  : return age.step.subvert_harmless;
		case subv.type.mindless  : return age.step.subvert_mindless;
		case subv.type.traitorous: return age.step.subvert_traitorous;
		case subv.type.cave_in   : return age.step.subvert_cave_in;
	}
	return null;
}

subv.to_event = function(type) {
	switch (type) {
		case subv.type.harmless  : return log.event.subvert_harmless;
		case subv.type.mindless  : return log.event.subvert_mindless;
		case subv.type.traitorous: return log.event.subvert_traitorous;
		case subv.type.cave_in   : return log.event.subvert_cave_in;
	}
	return null;
}

subv.from_ability = function(ability) {
	switch (ability) {
		case abilities.subvert_harmless  : return subv.type.harmless;
		case abilities.subvert_mindless  : return subv.type.mindless;
		case abilities.subvert_traitorous: return subv.type.traitorous;
		case abilities.subvert_cave_in   : return subv.type.cave_in;
	}
	return null;
}

subv.from_age = function(step) {
	switch (step) {
		case age.step.subvert_harmless  : return subv.type.harmless;
		case age.step.subvert_mindless  : return subv.type.mindless;
		case age.step.subvert_traitorous: return subv.type.traitorous;
		case age.step.subvert_cave_in   : return subv.type.cave_in;
	}
	return null;
}

subv.next = function(type) {
	return ++subv.attached[type];
}

subv.setup = function() {
	
	// Initialize card arrays
	for (let s = 0; s < subv.num; s++) {
		subv[s] = [];
	}
	
	// Make card sprites
	for (let type = subv.type.num - 1; type >= 0; type--) {
		for (let s = 0; s < subv.num; s++) {
			subv[s][type] = env.add.sprite(env.nowhere, 'half cards');
			subv[s][type].setFrame(1 + type);
			subv[s][type].setInteractive();
			env.to_front(subv[s][type]);
		}
	}
	
	// Move to the side
	subv.reset();
}