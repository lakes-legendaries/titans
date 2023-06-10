// ====================================================================================
// Instructions: Help for the user
// ====================================================================================

// ====================================================================================
// Interface
var instr = {
	preload: function() {}, // phaser plugin - load rexbbcodetext plugin
	setup  : function() {}, // phaser plugin - initial text, add hover-over instructions for card
	update : function() {}, // phaser plugin
	pause  : false,         // when true, don't update
	num_sub: 0,             // # can be subverted (set by controller)
	game_over: false,       // mark true when game's over to change instr box
}

instr.preload = function() {
	let url = 'https://raw.githubusercontent.com/rexrainbow/phaser3-rex-notes/b0ae5f3c53d9e35dccf5ab2dde4ceefde9a05f0a/dist/rexbbcodetextplugin.min.js';
	env.physics.load.plugin('rexbbcodetextplugin', url, true);
}

instr.setup = function() {
	instr.beginning    = true;
	instr.last_message = "";
	instr.text = env.physics.add.rexBBCodeText(instr.pos.x, instr.pos.y, "", instr.format).setOrigin(0.5, 0);
	ref.button.on('pointerdown', () => {log.active = false;});
	log.button.on('pointerdown', () => {ref.hide();});
	instr.show_surge = false;
	instr.game_over  = false;
	instr.jace_read  = false;
}

instr.update = function() {
	
	// Show log
	if (log.active) {
		while (true) {
			instr.show(log.text(), false, 16);
			if (instr.text.height > instr.max_height) {
				log.cut();
			} else {break;}
		}
		return;
	}
	
	// Show ref sheet abilities timing
	if (ref.active()) {
		instr.show("[u][color=purple]Ability Timing[/color][/u]\nWhen there are multiple abilities in play that have not yet been activated, first activate the ability among them that has the highest priority (as listed on the Reference Sheet). This may force you to backtrack on the priority stack as additional cards are played, like in this example:\n\nYou play a card with [color=green]Flash[/color] (priority 5) at the same time your opponent plays a card with [color=green]Haunt[/color] (priority 6). Then, you use [color=green]Flash[/color] to play a card with [color=green]Mimic[/color] (priority 4). Next, your [color=green]Mimic[/color] activates (before your opponent's [color=green]Haunt,[/color] even though your opponent played the card with [color=green]Haunt[/color] before you played the card with [color=green]Mimic[/color]).\n\nIt doesn’t matter what order cards were played in: All that matters is which ability has the highest priority.");
		return;
	}
	
	// Skip if tutorial
	if (tut.block_instr) {
		instr.show("");
		return;
	}
	
	// Show hover text
	if (full_card.active()) {
		instr.hover_text(full_card.showing[1], full_card.showing_subv);
		return;
	}
	
	// Don't show after-battle text, if in tutorial
	if (tut.active()) {
		instr.show("");
		return;
	}
	
	// Show tutorial text
	if (instr.beginning && age.major() == 0 && age.minor() == 0) {
		instr.show("Welcome to the online demo for Titans of Eden, an upcoming tabletop card game that will launch on KickStarter in 2021.\n\nFirst time playing? Click the Tutorial button below.\n\nTo start the game, play a card. Your opponent will play a card at the same time you do.");
		return;
	} else {instr.beginning = false;}
	
	// Don't show if moving
	if (move.in_progress()) {instr.show(""); return;}
	
	// Show main text
	if (age.major() < age.battle) {
		switch (age.minor()) {
			case age.step.play: {
				
				// get age
				let str = "It's the ";
				switch (age.major()) {
					case 0: str += "1st"; break;
					case 1: str += "2nd"; break;
					case 2: str += "3rd"; break;
				}
				str += " Age.\n\n";
				
				// write summon text
				let sum_count = stats.count(player.you, abilities.summon);
				if (sum_count > 0) {
					str += "Because you have ";
					if (sum_count > 1) {str += sum_count + " ";}
					str += "[color=blue]Summon[/color]";
					if (sum_count > 1) {str += " abilities";}
					str += ", you get to play " + (sum_count + 1) + " cards.\n\nOnce all cards have been selected, all cards will be flipped up simultaneously.";
				
				// write normal text
				} else {
					str += "Play a card facedown. You can play from your hand, or the top card of your deck.\n\nOnce you and your opponent have each selected a card, both cards will be flipped faceup simultaneously.";
					
					// Show surge text
					if (age.major() == 0 && stats.temples(player.you) < 3 && stats.num_surges(player.you) > 0) {
						str += "\n\n[color=purple][u]Surge[/u][/color]\nIf you don't like the hand you drew, you can use [color=green]Surge[/color] to discard it and draw a new one. (You can only do this twice per game, and you cannot do this if you haven't lost a temple yet this game.)";
					}
				}
				
				instr.show(str);
				break;
			}
			case age.step.subvert_harmless: {
				let sub_count = stats.count(player.you, abilities.subvert_harmless);
				instr.show("[u][color=purple]Subvert[/u][/color]\nChoose " + sub_count + " of your opponent's cards to [color=green]Subvert[/color] with [color=teal]Harmless.[/color] The card you choose will have 0 base power for the rest of the turn. (It can still gain power through [color=green]Bolster.[/color])");
				break;
			}
			case age.step.subvert_mindless: {
				let sub_count = stats.count(player.you, abilities.subvert_mindless);
				instr.show("[u][color=purple]Subvert[/u][/color]\nChoose " + sub_count + " of your opponent's cards to [color=green]Subvert[/color] with [color=teal]Mindless.[/color] The card you choose will have no abilities for the rest of the turn. (This ability cannot cancel another [color=green]Subvert[/color] ability, because all [color=green]Subvert[/color] abilities activate and are resolved simultaneously.)");
				break;
			}
			case age.step.subvert_traitorous: {
				let sub_count = stats.count(player.you, abilities.subvert_traitorous);
				instr.show("[u][color=purple]Subvert[/u][/color]\nChoose " + sub_count + " of your opponent's cards to [color=green]Subvert[/color] with [color=teal]Traitorous.[/color] The card you choose will be yours to control for the rest of the turn. (Before taking control of that card, your opponent will get to use any [color=green]Subvert[/color] abilities on that card, because all [color=green]Subvert[/color] abilities activate and are resolved simultaneously.)");
				break;
			}
			case age.step.subvert_cave_in: {
				let sub_count = stats.count(player.you, abilities.subvert_cave_in);
				instr.show("[u][color=purple]Subvert[/u][/color]\nYou may choose to [color=green]Subvert[/color] " + sub_count + " of your opponent's cards that were played this age with both [color=teal]Harmless[/color] and [color=teal]Mindless,[/color] which will cause the chosen card to have 0 power and no abilities for the rest of the turn. However, if you choose to use this ability, you must discard your [color=blue]Cavern's Defender[/color] from play.\n\n(This ability cannot cancel another [color=green]Subvert[/color] ability, because all [color=green]Subvert[/color] abilities activate and are resolved simultaneously.)\n\nIf you don't want to use [color=green]Subvert: Cave In,[/color] click the button below.");
				break;
			}
			case age.step.flash: {
				let flash_count = stats.count(player.you, abilities.flash);
				instr.show("[u][color=purple]Flash[/u][/color]\nPlay " + flash_count + " additional card" + (flash_count > 1? "s": "") + " now.");
				break;
			}
			case age.step.sacrifice: {
				let sac_count = stats.count(player.you, abilities.sacrifice);
				instr.show("[u][color=purple]Sacrifice[/u][/color]\nYou may permanently remove " + sac_count + " card" + (sac_count > 1? "s": "") + " in your hand from the game.\n\nSelect the cards you want to [color=green]Sacrifice,[/color] then click the button below.\n\nWe recommend first removing your Wizards from your hand, and then your Monks. (This will cause you to draw your better cards more often.)");
				break;
			}
			case age.step.purify: {
				let purify_count = stats.count(player.you, abilities.purify);
				instr.show("[u][color=purple]Purify[/u][/color]\nYou may remove all subversions from any " + purify_count + " card" + (purify_count > 1? "s": "") + ".\n\nIf you don't want to use [color=green]Purify,[/color] click the button below.");
				break;
			}
			case age.step.buy: {
				let energy = stats.energy(player.you);
				let str = "[u][color=purple]Awakening[/u][/color]\nYou may use your [color=green]Energy[/color] abilities to awaken a card from the ritual piles. The card you awaken will be added to your discard, and then shuffled into your deck at the end of the turn.\n\n[color=green]You have " + energy + " Energy.[/color]\n\nIf you don't want to awaken a card, click the button below."
				if (energy > 4) {
					str += "\n\nEven though you have more than 4 [color=green]Energy,[/color] you can still only awaken 1 card.";
				}
				instr.show(str);
				break;
			}
			case age.step.substitute_out: {
				let str = "[u][color=purple]Substitute Out[/u][/color]\nYou may discard " + instr.num_sub + " card";
				if (instr.num_sub > 1) {str += "s";}
				str += " from play. If you do, you may play 1 card";
				if (instr.num_sub > 1) {str += " for each card discarded";}
				str += ".\n\nNow abilities (i.e. any ability that starts with the word Now) will not activate on the card";
				if (instr.num_sub > 1) {str += "s";}
				str += " you play.\n\nIf you don't want to use [color=green]Substitute,[/color] click the button below."
				instr.show(str);
				break;
			}
			case age.step.substitute_in: {
				instr.show("[u][color=purple]Substitute In[/u][/color]\nPlay a card (to replace the card you just [color=green]Substituted[/color] out of play). You can play from your hand, or the top card of your deck.\n\nNow abilities (i.e. any ability that starts with the word Now) will not activate on the card you play.");
				break;
			}
		}
	} else {
		
		// Get stats
		let winner  = stats.winner ();
		let capture = stats.capture();
		
		// Get text
		let str = "";
		if (winner == player.you) {
			str += "[color=purple]You win the battle![/color]\n\n" + (capture? "Capture": "Destroy") + " 1 of your opponent's temples.";
		} else if (winner == player.opp) {
			str += "[color=purple]You lose the battle.[/color]\n\nYour opponent " + (capture? "captures": "destroys") + " 1 of your temples.";
		} else {
			str += "[color=purple]The battle is fought to a draw.[/color]";
		}
		if (!instr.game_over) {
			str += "\n\nSelect any number of cards in your hand to discard, then click one of the buttons below.\n\nAfter discarding, we'll shuffle all your cards together, and you'll draw until you have 6 cards in your hand.";
		}
		
		// Show instructions
		instr.show(str);
	}
}

// ====================================================================================
// Backend
instr.pos = {x: 1770, y:  415}
instr.max_height = 450;
instr.format = {
	backgroundColor: null,
	fontFamily: "Roboto",
	fontSize  : "22px",
	color     : "black",
	align     : "center",
	underline : {
		color    : "purple",
		thickness: 2,
		offset   : 5,
	},
	wrap: {
		mode: 'word',
		width: 250,
	},
	stroke: 'black',
	strokeThickness: 2,
}

instr.show = function(str, do_shrinking = true, font_size = 22, force = false) {
	
	// See if an update is required
	if (!instr.last_message.localeCompare(str) && !force) {return;}
	instr.last_message = str;
	
	// Update text
	instr.text.setText(str);
	
	// Update font size
	instr.text.setFontSize(font_size.toString() + "px");
	
	// Shrink if overflowing
	if (do_shrinking && instr.text.height > instr.max_height) {
		instr.show(str, do_shrinking, font_size-0.5, true);
	}
}

instr.hover_text = function(card_num, subversion_frame) {
	
	// check if card is being sacrificed
	if (zone.find(card_num) == null) {return;}
	
	// Show subversion
	if (subversion_frame != null) {
		let str;
		switch (subversion_frame) {
			case subv.type.harmless:
				str = "[color=teal]Harmless[/color] reduces the subverted card's base power to 0 for the rest of the turn. (It’s as if the power printed on the card is 0.)\n\nThe subverted card can still gain power through [color=green]Bolster.[/color]";
				break;
			case subv.type.mindless:
				str = "[color=teal]Mindless[/color] removes all abilities from the subverted card for the rest of the turn.";
				break;
			case subv.type.traitorous:
				str = "[color=teal]Traitorous[/color] lets you gain control of an opponent's card for the rest of the turn. (That card behaves as if it was 1 of your cards you played this turn.)";
				break;
		}
		instr.show(str);
		return;
	}
	
	// See if card can be seen
	if (card[card_num].sprite.frame.name == 0) {return;}
	
	// Show help text
	let str;
	switch (card[card_num].name) {
		case card.name.monk:
			str = "[color=blue]Monks[/color] have 0 power and the [color=green]Energy[/color] ability, which lets you awaken cards. (Whenever you awaken a card, it gets added to your discard pile, and then shuffled into your deck at the end of the turn.)\n\n[color=blue]Monks[/color] won't help you in battle, but they'll let you add elemental cards with powerful abilties to your deck.";
			break;
		case card.name.wizard:
			str = "[color=blue]Wizards[/color] have 1 power and no abilities.\n\nThey'll help you in battle, but they're not strong enough to win this war themselves.";
			if (age.first_turn()) {
				str += "\n\n[color=red]We [i]highly[/i] recommend that you only play Monks on your first turn.[/color]";
			}
			break;
		case card.name.traveler:
			str = "[color=blue]Travelers[/color] give you 2 [color=green]Energy[/color] each age, which lets you awaken better and more costly cards than you could with [color=blue]Monks[/color] alone.\n\nAdditionally, [color=blue]Travelers[/color] have 1 power, so they contribute to the fight (instead of idly standing by).";
			break;
		case card.name.ghost:
			str = "[color=blue]Ghosts[/color] have negative power and no abilities. Don’t awaken [color=blue]Ghosts[/color]: They’re bad. They’re only in the game so you can make your opponent gain them (with the [color=green]Haunt[/color] ability.";
			break;
		case card.name.nikolai_the_cursed:
			str = "[color=blue]Nikolai, The Cursed[/color] has [color=green]Summon[/color], which lets you play 2 cards, instead of 1, each age.\n\nSo, if you play [color=blue]Nikolai, The Cursed[/color] during the 1st Age, then you get to play 2 cards, simultaneously, during each of the 2nd and 3rd Ages. (On the other hand, if you play [color=blue]Nikolai, The Cursed[/color] during the 3rd Age, it has no effect.)\n\n[color=blue]Nikolai, The Cursed[/color] has 0 power, but this is more than made up for by its awesome [color=green]Summon[/color] ability.";
			break;
		case card.name.zodiac_the_eternal:
			str = "";
			if (!instr.jace_read || age.first_turn()) {
				str += "(If this is your first game, you should read [color=blue]Jace, Winter's Firstborn[/color] before reading this card. [color=blue]Jace[/color] is the Ice Warrior, located one card to the right of this card.)\n\n";
			}
			str += "[color=blue]Zodiac, The Eternal[/color] lets you respond to your opponent’s [color=green]Subvert[/color] abilities with vengeance and fury. When you play [color=blue]Zodiac, The Eternal,[/color] you get to remove all [color=teal]subversions[/color] from any 1 card.\n\nSo, if your opponent makes your card [color=teal]Harmless[/color] (with [color=blue]Jace, Winter's Firstborn[/color]), you can turn right around and remove that [color=teal]subversion.[/color] With 2 power, [color=blue]Zodiac, The Eternal[/color] is a force to be reckoned with.";
			break;
		case card.name.jace_winters_firstborn:
			str = "[color=blue]Jace, Winter's Firstborn[/color] has [color=green]Subvert: Harmless,[/color] which lets you attach the [color=teal]Harmless Subversion[/color] to any 1 of your opponent's cards.\n[color=teal]Subversions[/color] are effects that temporarily modify cards – they stay attached to a card until the end of the turn.\n\nThe [color=teal]Harmless Subversion[/color] reduces a card's base power to 0 for the rest of the turn. (It’s as if the power printed on the card is 0.) So, if you play this card, you'll have 1 power [color=blue](Jace's)[/color], and your opponent will have 0. That’s a pretty good trade-off for you.";
			instr.jace_read = true;
			break;
		case card.name.akari_timeless_fighter:
			str = "[color=blue]Akari, Timeless Fighter[/color] lets you [color=green]Draw[/color] 2 cards at the start of each age, before any cards are played.\n\nIf you play [color=blue]Akari, Timeless Fighter[/color] during the 1st Age, you’ll draw 2 cards at the start of the 2nd and 3rd Ages. However, if you play [color=blue]Akari, Timeless Fighter[/color] during the 3rd Age, this ability will have no effect.\n\n[color=blue]Akari[/color] has a lot of power, and helps you find the other powerful cards in your deck.";
			break;
		case card.name.winds_howl:
			str = "[color=blue]Wind's Howl[/color] has [color=green]Flash 2,[/color] which lets you play 2 additional cards right after you play it.\n\nIf you and your opponent both play cards with [color=green]Flash,[/color] then you and your opponent play all additional cards simultaneously. If any of the additional cards you play have [color=green]Flash,[/color] you repeat this process.\n\nBe careful with [color=blue]Wind's Howl[/color]: It has negative power, which you'll need to make up for with the cards you play.";
			break;
		case card.name.living_volcano:
			str = "[color=blue]Living Volcano[/color] has [color=green]Flash,[/color] which lets you play 1 additional card right after you play it. It also has [color=green]Discard 2,[/color] which forces your opponent to discard 2 random cards at the start of each age (limiting what they are able to play).";
			break;
		case card.name.return_of_the_frost_giants:
			str = "[color=blue]Return of the Frost Giants[/color] has [color=green]Flash,[/color] which lets you play 1 additional card right after you play it. It also has [color=green]Substitute,[/color] which lets you replace 1 card you have in play with another 1 of your cards right before the battle.\n\n[color=indigo]Now abilities[/color], which are abilities that start with the word [color=indigo]Now[/color], don't activate on replacement cards.\n\n[color=green]Substitute[/color] lets you play a [color=blue]Monk[/color] to awaken cards with, then replace it before battle with a more powerful card.";
			break;
		case card.name.spine_splitter:
			str = "[color=blue]Spine Splitter[/color] has [color=green]Flash,[/color] which lets you play 1 additional card right after you play it. It also has [color=green]Sacrifice 2,[/color] which lets you permanently remove 2 cards in your hand from the game.\n\nThis is extremely powerful ability because it lets you remove your low-value cards from your deck, making it so that you'll draw your high-value cards more often.\n\nWe recommend removing [color=blue]Wizards[/color] first, and then [color=blue]Monks.[/color]";
			break;
		case card.name.aurora_draco:
			str = "[color=blue]Aurora Draco[/color] has [color=green]Haunt 2[/color], which forces your opponent to gain 2 [color=blue]Ghosts[/color] into play. These [color=blue]Ghosts[/color] will be [color=green]Subverted[/color] with [color=teal]Harmless[/color] – they have 0 power this turn. At the end of the turn, they'll get shuffled into your opponent’s deck, which is where they'll start to wreak their havoc: Your opponent might draw these [color=blue]Ghosts[/color] (instead of better cards), and might even accidently play 1 of them from the top of their deck.\n\n[color=blue]Aurora Draco[/color] has [color=green]Bolter: Fire,[/color] which gives it +1 power for each fire card your opponent has in play.";
			break;
		case card.name.smoldering_dragon:
			str = "[color=blue]Smoldering Dragon[/color] has [color=green]Protect[/color], which saves your cards from being [color=green]Subverted[/color] and you from being [color=green]Haunted[/color] this age.\n\n[color=green]Protect[/color] does not work retro-actively: All [color=teal]subversions[/color] that have already been attached this turn remain attached.\n\n[color=green]Protect[/color] does not last the whole turn: It wears off at the end of the age [color=blue]Smoldering Dragon[/color] is played.\n\n[color=blue]Smoldering Dragon[/color] has [color=green]Bolter: Ice,[/color] which gives it +1 power for each ice card your opponent has in play.";
			break;
		case card.name.frostbreath:
			str = "[color=blue]Frostbreath[/color] has [color=green]Subvert: Mindless[/color], which lets you attach [color=teal]Mindless[/color] to an opponent’s card. [color=teal]Mindless[/color] removes all abilities on the subverted card, including [color=green]Bolster[/color] and [color=green]Energy[/color].\n\n[color=blue]Frostbreath[/color] has [color=green]Bolter: Rock,[/color] which gives it +1 power for each rock card your opponent has in play.";
			break;
		case card.name.caverns_defender:
			str = "When you play [color=blue]Cavern’s Defender[/color], you get to choose: 1) Do nothing, or 2) Discard [color=blue]Cavern's Defender[/color] from play and attach [color=teal]Harmless[/color] and [color=teal]Mindless[/color] to an opponent's card played this age. ([color=teal]Harmless[/color] and [color=teal]Mindless[/color] remove all power and abilities from a card.)\n\nThe choice is yours: Do you want [color=blue]Cavern's Defender's[/color] power, or do you want to cancel out an opponent's card?\n\n[color=blue]Cavern's Defender has [color=green]Bolter: storm,[/color] which gives it +1 power for each storm card your opponent has in play.";
			break;
		case card.name.madness_of_1000_stars:
			str = "[color=blue]Madness Of 1,000 Stars[/color] is the titan that just keeps giving: When you play it, you get [color=green]4 Energy[/color] this age (and only this age -- not every age). So, whenever you play [color=blue]Madness Of 1,000 Stars,[/color] you can get another titan.";
			break;
		case card.name.final_judgment:
			str = "[color=blue]Final Judgment[/color] makes your opponent [color=green]Discard[/color] all of the cards in their hand. For the rest of the turn, they'll be forced to play random cards from the top of their deck.";
			break;
		case card.name.hell_frozen_over:
			str = "[color=blue]Hell, Frozen Over[/color] lets you attach [color=teal]Traitorous[/color] to an opponent's card, which gives you control over that card for the rest of the turn. (You get to use its power and abilities.)";
			break;
		case card.name.what_lies_beneath:
			str = "[color=blue]What Lies Beneath[/color] is one of the most powerful cards in this game. It gets +1 power for each card your opponent has in play, for up to 6 total power.";
			break;
	}
	
	// add live stats for in play
	let loc = zone.find(card_num);
	if (loc.place == zone.play) {
		str += "\n\n[color=orangered]Current Power: " + stats.card_power(card_num).toString() + "[/color]";
	}
	
	// show
	instr.show(str);
}