// ====================================================================================
// Basic enum for abilities
// ====================================================================================

// ====================================================================================
// Interface
var abilities = {
	
	// enum
	bolster_enemies    :  0,
	bolster_fire       :  1,
	bolster_ice        :  2,
	bolster_rivals     :  3,
	bolster_rock       :  4,
	bolster_storm        :  5,
	discard            :  6,
	draw               :  7,
	energy             :  8,
	energy_evanesce    :  9,
	flash              : 10,
	haunt              : 11,
	protect            : 12,
	purify             : 13,
	sacrifice          : 14,
	substitute         : 15,
	subvert_cave_in    : 16,
	subvert_harmless   : 17,
	subvert_mindless   : 18,
	subvert_traitorous : 19,
	summon             : 20,
	num                : 21,
	
	// meta-data
	first_subvert: function() {return abilities.subvert_cave_in;},    // starting point for iterating through subversions
	last_subvert : function() {return abilities.subvert_traitorous;}, // ending   point for iterating through subversions
	num_subvert  : function() {return abilities.last_subvert() - abilities.first_subvert() + 1;}, // count of subv abilities
	
	// convert between enums
	from_age: function(minor_age) {}, // call w/out argument to use current age
	to_text : function(ability)   {}, // convert to string
}

abilities.from_age = function(minor_age) {
	switch (minor_age == null? age.minor(): minor_age) {
		case age.step.play:
			return abilities.summon;
		case age.step.subvert_harmless:
			return abilities.subvert_harmless;
		case age.step.subvert_mindless:
			return abilities.subvert_mindless;
		case age.step.subvert_traitorous:
			return abilities.subvert_traitorous;
		case age.step.subvert_cave_in:
			return abilities.subvert_cave_in;
		case age.step.flash:
			return abilities.flash;
		case age.step.haunt:
			return abilities.haunt;
		case age.step.sacrifice:
			return abilities.sacrifice;
		case age.step.purify:
			return abilities.purify;
		case age.step.buy:
			return abilities.energy_evanesce;
		case age.step.substitute_out:
			return abilities.substitute;
		case age.step.substitute_in:
			return abilities.substitute;
	}
	return null;
}

abilities.to_text = function(ability) {
	switch (ability) {
		case abilities.bolster_enemies    : return "Bolster: Enemeies";
		case abilities.bolster_fire       : return "Bolster: Fire";
		case abilities.bolster_ice        : return "Bolster: Ice";
		case abilities.bolster_rivals     : return "Bolster: Rivals";
		case abilities.bolster_rock       : return "Bolster: Rock";
		case abilities.bolster_storm        : return "Bolster: storm";
		case abilities.discard            : return "Discard";
		case abilities.draw               : return "Draw";
		case abilities.energy             : return "Energy";
		case abilities.energy_evanesce    : return "Energy: Evanesce";
		case abilities.flash              : return "Flash";
		case abilities.haunt              : return "Haunt";
		case abilities.protect            : return "Protect";
		case abilities.purify             : return "Purify";
		case abilities.sacrifice          : return "Sacrifice";
		case abilities.substitute         : return "Substitute";
		case abilities.subvert_cave_in    : return "Subvert: Cave In";
		case abilities.subvert_harmless   : return "Subvert: Harmless";
		case abilities.subvert_mindless   : return "Subvert: Mindless";
		case abilities.subvert_traitorous : return "Subvert: Traitorous";
		case abilities.summon             : return "Summon";
	}
	return null;
}

// ====================================================================================
// Backend
Object.freeze(abilities); // set immutable