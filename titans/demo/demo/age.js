// ====================================================================================
// Age (phase) of the turn
// ====================================================================================

// ====================================================================================
// Interface
var age = {
	
	// set points
	battle    : 3, // age # corresponding to battle (last age of turn)
	substitute: 2, // age that substitute trigger
	
	// enum
	step  : {                     // steps of each non-battle age
		play              : 0.00, // play a card
		subvert_harmless  : 0.01, // resolve subversions
		subvert_mindless  : 0.02, // resolve subversions
		subvert_traitorous: 0.03, // resolve subversions
		subvert_cave_in   : 0.04, // resolve subversions
		flash             : 0.05, // flash in additional cards
		haunt             : 0.06, // haunt your opponent
		sacrifice         : 0.07, // sacrifice cards from your hand
		purify            : 0.08, // purify cards in play
		buy               : 0.09, // buy a card
		substitute_out    : 0.10, // substitute a card out of play (3rd age only)
		substitute_in     : 0.11, // substitute a card into   play (3rd age only)
	},
	
	// modifiers
	advance   : function()      {}, // increment age counter
	set_minor : function(minor) {}, // change minor age to provided value
	end       : function()      {}, // mark the end of this age (age.cur -> age.step.end + age.step.incr)
	
	// queries
	major     : function() {},                         // get current age (integer)
	minor     : function() {},                         // get step of the age (play-buy)
	first_turn: function() {return age.is_first_turn}, // returns false after first turn completed
	
	// phaser plugins
	setup: function() {},
}

age.major = function() {
	return Math.floor(age.cur);
}

age.minor = function() {
	let inv_step = Math.round(1.0 / age.step.incr);
	return Math.round(inv_step * (age.cur % 1)) / inv_step;
}

age.advance = function() {
	if (age.cur == age.battle) {
		age.cur = 0;
		age.is_first_turn = false;
	} else {
		age.cur += age.step.incr;
		if (age.minor() > age.step.end) {
			age.cur = Math.ceil(age.cur);
		}
	}
}

age.set_minor = function(minor) {
	age.cur = age.major() + minor;
}

age.end = function() {
	age.cur = age.major() + age.step.end + age.step.incr;
}

age.setup = function() {
	age.cur           =    0;
	age.is_first_turn = true;
}

// ====================================================================================
// Backend
age.step.incr     = 0.01;
age.step.end      = age.step.substitute_in;