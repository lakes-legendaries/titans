// ====================================================================================
// Zone (location) of cards
// ====================================================================================

// ====================================================================================
// Interface
var zone = {
	
	// enum
	hand     : 0,
	play     : 1,
	deck     : 2,
	disc     : 3,
	buy_top  : 4,
	buy_other: 5, // anywhere in buy pile, except on top
	
	// cards in each zone
	contains: function(person, place, card_num)   {}, // check if card exists in given zone
	find    : function(card_num)                  {}, // find current person, zone, and index for a card
	get     : function(person, place, position)   {}, // get card_num in contents (zone). if position is blank, return all
	get_last: function(person, place)             {}, // get last (top) card_num in contents (zone)
	get_rand: function(person, place)             {}, // get random card_num from zone
	count   : function(person, place)             {}, // count of cards in each zone
	dest    : function(person, place, pos_in_seq) {}, // coords for a card NOT in a buy pile
	buy_pos : function(row, col, count, pos)      {}, // coords for a card in buy pile
	
	// zone modifiers
	add    : function(person, place, card_num) {}, // add card to zone
	change : function(person, place, card_num) {}, // move card from one zone to another
	remove : function(card_num)                {}, // remove card so it's not in any zone
	shuffle: function()                        {}, // shuffle decks
	
	// phaser plugins
	setup: function() {}, // build out contents array
}

zone.contains = function(person, place, card_num) {
	return math.contains(zone.contents[person][place], card_num);
}

zone.find = function(card_num) {
	let person, place, index;
	for (person = 0; person <= player.num; person++) {
		for (place = 0; place < zone.num; place++) {
			index = math.index_of(zone.contents[person][place], card_num);
			if (index != null) {
				return {person: person, place: place, index: index};
			}
		}
		if (index != null) {break;}
	}
	return null;
}

zone.get = function(person, place, position) {
	if (position == null) {return zone.contents[person][place];}
	return zone.contents[person][place][position];
}

zone.get_last = function(person, place) {
	return math.last(zone.contents[person][place]);
}

zone.get_rand = function(person, place) {
	return zone.contents[person][place][math.rand(zone.count(person, place))];
}

zone.count = function(person, place) {
	return zone.contents[person][place].length;
}

zone.dest = function(person, place, pos_in_seq) {
	
	// Get count & coords
	let count  = zone.count(person, place);
	let coords = zone.coords[person][place];
	if (pos_in_seq == null) {pos_in_seq = zone.count(person, place);}
	
	// Get position
	if (coords.x0 + (count - 1) * coords.xspace > coords.xmax || coords.xmax + (count - 1) * coords.xspace < coords.x0) {
		x = coords.x0 + pos_in_seq * (coords.xmax - coords.x0) / (count - 1);
	} else {x = coords.x0 + pos_in_seq * coords.xspace;}
	if (coords.y0 + (count - 1) * coords.yspace > coords.ymax || coords.ymax + (count - 1) * coords.yspace < coords.y0) {
		y = coords.y0 + pos_in_seq * (coords.ymax - coords.y0) / (count - 1);
	} else {y = coords.y0 + pos_in_seq * coords.yspace;}
	
	// Return
	return {x: x, y: y}
}

zone.buy_pos = function(row, col, pos, count) {
	let x0 = zone.coords.buy.x0 + row * zone.coords.buy.x_space;
	let y0 = zone.coords.buy.y0 + col * zone.coords.buy.y_space;
	let offset = pos * zone.coords.buy.breadth / (count - 1);
	return {x: x0 + offset, y: y0 + offset};
}

zone.add = function(person, place, card_num) {
	zone.contents[person][place].push(card_num);
}

zone.change = function(person, place, card_num) {
	zone.remove(card_num);
	zone.add   (person, place, card_num);
}

zone.remove = function(card_num) {
	let cur = zone.find(card_num);
	zone.contents[cur.person][cur.place].splice(cur.index, 1);
}

zone.shuffle = function() {
	for (let person = 0; person < player.num; person++) {
		zone.contents[person][zone.deck] = math.shuffle(zone.contents[person][zone.deck]);
	}
}

zone.setup = function() {
	zone.contents = [];
	for (person = 0; person <= player.num; person++) {
		zone.contents[person] = [];
		for (place = 0; place < zone.num; place++) {
			zone.contents[person][place] = [];
		}
	}
}

// ====================================================================================
// Backend

// holder of card_num's
zone.contents = []

// enum for # of zones
zone.num = 6;

// Coordinates for each zone ([person][place])
zone.coords   = [[], []]
zone.coords[player.you][zone.hand] = {x0:  130, xmax: 1050, xspace: 175, y0:  980, ymax:  980, yspace:  0}
zone.coords[player.opp][zone.hand] = {x0:  130, xmax: 1050, xspace: 175, y0:  100, ymax:  100, yspace:  0}
zone.coords[player.you][zone.play] = {x0:  150, xmax: 1000, xspace: 230, y0:  730, ymax:  730, yspace:  0}
zone.coords[player.opp][zone.play] = {x0:  150, xmax: 1000, xspace: 230, y0:  350, ymax:  350, yspace:  0}
zone.coords[player.you][zone.deck] = {x0: 1225, xmax: 1275, xspace:   5, y0:  730, ymax:  780, yspace:  5}
zone.coords[player.opp][zone.deck] = {x0: 1275, xmax: 1225, xspace:  -5, y0:  350, ymax:  300, yspace: -5}
zone.coords[player.you][zone.disc] = {x0: 1500, xmax: 1500, xspace:   0, y0:  730, ymax:  780, yspace: 10}
zone.coords[player.opp][zone.disc] = {x0: 1500, xmax: 1500, xspace:   0, y0:  300, ymax:  350, yspace: 10}

// Buy pile coords
zone.coords.buy = {x0: 2045, y0: 135, breadth: 60, x_space: 300, y_space: 250}