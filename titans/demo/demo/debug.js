// ====================================================================================
// Debug -- have game play itself, endlessly, w/ random inputs
// ====================================================================================

// ====================================================================================
// Interface
var debug = {
	all_faceup: false,         // turn all cards faceup
	play_self : false,         // set to true to run
	queue_off :  true,         // turn off move queue (if play_self) (finds errors fast)
	up_energy :  true,         // make energy plentiful (if play_self)
	update    : function() {}, // phaser plugin
}

debug.setup = function() {
	if (debug.all_faceup) {
		move.all_faceup = true;
		move.organize(player.opp, zone.hand);
		move.organize(player.opp, zone.deck);
		move.organize(player.you, zone.deck);
	}
	if (debug.play_self) {
		move.default_speed = 10E3;               // make everything move faster
		move.queue_off     = debug.queue_off;    // turn off move queue
		if (debug.up_energy) {
			for (let card_num = 0; card_num < card.ordinal.traveler; card_num++) {
				card[card_num].abilities[abilities.energy] = math.rand(3);
			}
		}
		console.log("Game " + ++debug.game_num); // log game #
	}
}

debug.update = function() {
	if (debug.play_self && !move.in_progress()) {
		// play the game
		if (!controller.game_over) {
			// select a card
			if ((age.first_turn() && age.major() < age.battle) || math.rand(10) > 0) {
				for (let g = 0; g < 100; g++) {
					let choice, is_forest, is_desert;
					do {
						choice = math.rand(card.num);
						is_forest = card[choice].elem == card.elem.forest;
						is_desert = card[choice].elem == card.elem.desert;
					} while ((is_forest || is_desert) && math.rand(10) > 0);
					if (input.select(choice)) {break;}
				}
			// push a button
			} else {
				if (Math.random() > 0.5) {
					button.click.top();
				} else {
					button.click.bot();
				}
			}
			// toggle the log
			if (math.rand(100) == 0) {
				log.active = !log.active;
			}
		// restart the game
		} else {
			game.setup();
		}
	}
}

// ====================================================================================
// Back-end
debug.game_num = 0;