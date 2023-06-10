// ====================================================================================
// Move sprites around
// ====================================================================================

// ====================================================================================
// Interface
var move = {
	fade_in       : function(obj, root, offset)              {}, // sprite fades to visible   (can pass sprite or card_num)
	fade_out      : function(card_num)                       {}, // sprite fades to invisible (can pass sprite or card_num)
	add_pause     : function(pause_len)                      {}, // add a pause to the move queue
	card          : function(person, place, card_num, props) {}, // move card_num to person, place w/ props
	combine_zone  : function(person, source, dest)           {}, // all cards in source zone -> dest zone
	organize      : function(person, place, props)           {}, // redistribute cards in a zone
	in_progress   : function()                               {}, // false if the move queue is empty
	fading        : function()                               {}, // returns true if the current move is a fade
	setup         : function()                               {}, // phaser plugin
	update        : function()                               {}, // phaser plugin
	default_speed : 3000,                                        // speed objects move
	queue_off     : false,                                       // for debugging -- cancel all movements
	all_faceup    : false,                                       // for debugging -- make all cards visible
}

move.fade_in  = function(obj, root, offset) {
	move.queue.push({fade: obj, out: false, root: root, offset: offset});
}

move.fade_out = function(card_num) {
	move.queue.push({fade: card_num, out: true});
}

move.add_pause = function(pause_len) {
	if (pause_len > 0) {
		move.queue.push({pause: pause_len});
	}
}

move.card = function(person, place, card_num, props) {
	
	// Set props
	if (props == null) {props = {};}
	if (props.card_num      == null) {props.card_num      = card_num;}
	if (props.destination   == null) {props.destination   = zone.dest(person, place);}
	if (props.flip_facedown == null) {props.flip_facedown = place != zone.disc;}
	if (props.flip_faceup   == null) {props.flip_faceup   = place == zone.disc || (person == player.you && place == zone.hand);}
	props = move.get_default_props(props);
	
	// Debugging -- make cards be faceup
	if (move.all_faceup) {
		props.flip_faceup = true;
	}
	
	// Add to queue
	move.queue.push(Object.assign({}, props));
	
	// Move attached subversions w/ cards in play
	if (place == zone.play) {
		let dest = props.destination;
		props.bring_to_front = false;
		for (let type = 0, pos = 0; type < subv.type.num; type++) {
			let subv_card_num = card[card_num].subv[type];
			if (subv_card_num == null) {continue;}
			props.card_num    = subv[subv_card_num][type];
			props.destination = math.add_coords(dest, subv.offset(pos++, card.num_subv(card_num)));
			move.queue.push(Object.assign({}, props));
		}
	
	// Remove subversions
	} else if (place == zone.disc) {
		card.discard(card_num);
	}
	
	// Change zone
	if (!props.skip_move && person != null && place != null) {
		zone.change(person, place, card_num)
	}
}

move.combine_zone = function(person, source, dest) {
	
	// If person argument is null, do both
	if (person == null) {
		for (let person = 0; person < player.num; person++) {
			move.combine_zone(person, source, dest);
		}
		return;
	}
	
	// Move cards, one at a time, from source to dest
	while (zone.count(person, source) > 0) {
		let card_num = zone.get_last(person, source);
		move.card(person, dest, card_num, {simultaneous: true});
	}
	
	// Organize cards
	move.organize(person, dest);
}

move.organize = function(person, place, props) {
	
	// If person argument is null, do both
	if (person == null) {
		for (let person = 0; person < player.num; person++) {
			move.organize(person, place, props);
		}
		return;
	}
	
	// Add pause
	move.add_pause(1);
	
	// Set props
	if (props == null) {props = {};}
	if (props.bring_to_front == null) {props.bring_to_front = false;}
	if (props.flip_facedown  == null) {props.flip_facedown  = false;}
	if (props.flip_faceup    == null) {props.flip_faceup    = false;}
	if (props.simultaneous   == null) {props.simultaneous   =  true;}
	props = move.get_default_props(props);
	
	// Organize cards
	for (let d = 0; d < zone.count(person, place); d++) {
		props.card_num    = zone.get (person, place, d);
		props.destination = zone.dest(person, place, d);
		move.card(null, place, props.card_num, props);
	}
	
	// Add pause
	move.add_pause(1);
}

move.in_progress = function() {
	return move.queue.length > 0;
}

move.fading = function() {
	return move.in_progress? move.queue[0].fade != null: false;
}

move.setup = function() {	
	move.queue = []
	move.prev_dist = [];
}

// ====================================================================================
// Backend

move.get_default_props = function(props) {
	if (props                == null) {props = {}}
	if (props.bring_to_front == null) {props.bring_to_front =  true;}
	if (props.flip_facedown  == null) {props.flip_facedown  =  true;}
	if (props.flip_faceup    == null) {props.flip_faceup    =  true;}
	if (props.move_speed     == null) {props.move_speed     = move.default_speed;}
	if (props.simultaneous   == null) {props.simultaneous   = false;}
	if (props.skip_move      == null) {props.skip_move      = false;}
	return props;
}

move.process_pause = function() { // returns true if currently paused
	if (move.queue[0].pause != null) {
		if (move.queue[0].pause-- == 0) {
			move.queue.shift();
		}
		return true;
	}
	return false;
}

move.process_fade = function() {
	let props    = move.queue[0];
	let card_num = props.fade;
	if (card_num != null) {
		
		// Get sprite
		let is_card = !isNaN(card_num); // we can pass sprites & other objects to this method, as card_num
		let sprite  = is_card? card[card_num].sprite: card_num;
		
		// Flip fade-out card faceup
		let fading_out = props.out;
		if (is_card && fading_out && sprite.frame.name == 0) {
			sprite.setFrame(card[card_num].sprite_frame);
			env.to_front(sprite);
			move.queue.unshift({pause: 30});
			return;
		}
		
		// Place fade-in card
		if (is_card && !fading_out) {
			let dest = math.add_coords(card[props.root].sprite, props.offset);
			if (dest.x != sprite.x || dest.y != sprite.y) {
				env.place(sprite, dest);
				sprite.alpha = 0;
			}
		}
		
		// Make visible if not
		if (!fading_out && sprite.visible == false && !sprite.kill) {
			sprite.visible = true;
			sprite.alpha   = 0;
		}
		
		// Change transparency
		sprite.alpha += (fading_out? -1: 1) * 0.05;
		let done = fading_out? sprite.alpha <= 0: sprite.alpha >= 1;
		
		// Finish fade
		if (done) {
			if (fading_out) {
				env.place(sprite, env.nowhere);
			}
			move.queue.shift();
		}
		return true;
	}
	return false;
}

move.start = function(queue_position = 0) {
	
	// Check if a new move is to be started
	if (queue_position >= move.queue.length) {return;} // invalid queue position
	if (move.prev_dist.length > 0)           {return;} // moves ongoing
	
	// See if should be skipped
	let props  = move.queue[queue_position];
	if (props.pause != null)  {return;} // called through simultaneous -- can't simultaneous a pause, though
	if (props.fade  != null)  {return;} // called through simultaneous -- can't simultaneous a fade , though
	
	// Get sprite
	let sprite;
	let is_card = !isNaN(props.card_num); // false if is subversion
	if (is_card) {
		sprite = card[props.card_num].sprite;
	} else {sprite = props.card_num;}
	
	// Bring to front, flip facedown
	if (props.bring_to_front) {env.to_front(sprite);}
	if (props.flip_facedown && is_card) {sprite.setFrame(0);}
	
	// Start additional, simultaneous moves
	if (props.simultaneous) {move.start(queue_position + 1);}
	
	// Send
	if (!props.skip_move) {
		env.physics.physics.moveTo(sprite, props.destination.x, props.destination.y, props.move_speed);
		move.prev_dist[queue_position] = math.dist({x: sprite.x, y: sprite.y}, props.destination);
	} else {move.prev_dist[queue_position] = 0;}
}

move.finish = function(queue_position = 0) {
	
	// Check if a move can be finished
	if (queue_position >= move.queue.length) {return;} // invalid queue position
	if (move.prev_dist.length == 0)          {return;} // no ongoing moves
	
	// See if should be skipped
	let props = move.queue[queue_position];
	if (props.pause != null)  {return;} // called through simultaneous -- can't simultaneous a pause, though
	if (props.fade  != null)  {return;} // called through simultaneous -- can't simultaneous a fade , though
	
	// Finish additional, simultaneous moves
	if (props.simultaneous) {move.finish(queue_position + 1);}
	
	// Get sprite
	let sprite;
	let is_card = !isNaN(props.card_num); // false if is subversion
	if (is_card) {
		sprite = card[props.card_num].sprite;
	} else {sprite = props.card_num;}
	
	// Check if a moving object has reached its destination
	let cur_dist   = math.dist(sprite, props.destination);
	let at_endpt   = cur_dist == 0;
	let past_endpt = cur_dist > move.prev_dist[queue_position];
	if (at_endpt || past_endpt || props.skip_move) {
		if (!props.skip_move) {env.place(sprite, props.destination);} // anchor to endpoint
		if (props.flip_faceup && is_card) {                           // flip faceup
			sprite.setFrame(card[props.card_num].sprite_frame);
		}
		move.prev_dist[queue_position] = 0;                           // mark move completed
	} else {move.prev_dist[queue_position] = cur_dist;}               // log new distance
}

move.clean_up = function() {
	if (move.prev_dist.length == 0) {return;}    // no ongoing moves
	if (math.sum(move.prev_dist) != 0) {return;} // all moves haven't yet been completed
	for (let g = 0; g < move.prev_dist.length; g++) {
		move.queue.shift();
	}
	move.prev_dist = [];
}

move.update = function() {
	if (move.queue_off) {move.queue = [];}
	if (move.queue.length == 0) {return;}
	if (move.process_pause())   {return;}
	if (move.process_fade ())   {return;}
	if (move.prev_dist.length == 0) {
		move.start   ();
	} else {
		move.finish  ();
		move.clean_up();
	}
}